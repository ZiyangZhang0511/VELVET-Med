import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07, learnable_temp=True, alpha_i2t=0.5):
        super().__init__()

        if learnable_temp:
            self.temperature = nn.Parameter(temperature*torch.ones([]))
        else:
            self.temperature = torch.as_tensor(temperature)

        self.alpha_i2t = alpha_i2t

    def forward(self, embed_vis, embed_txt, temperature=None, alpha_i2t=None):

        if not alpha_i2t:
            alpha_i2t = self.alpha_i2t

        if not temperature:
            temperature = self.temperature.to(embed_vis.device)
        
        batch_size = embed_vis.shape[0]

        img_embeds_norm = F.normalize(embed_vis, dim=-1)
        txt_embeds_norm = F.normalize(embed_txt, dim=-1)
        # print(img_embeds_norm.size(), txt_embeds_norm.size())

        similarity_matrix_it = img_embeds_norm @ txt_embeds_norm.t()
        similarity_matrix_ti = txt_embeds_norm @ img_embeds_norm.t()

        logits_i2t = similarity_matrix_it / temperature
        logits_t2i = similarity_matrix_ti / temperature
        # print(logits.size())

        # labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.long), num_classes=batch_size).float().to(embed_vis.device)
        # labels = labels.to(self.device)
        # print(labels)
        # loss_i2t = -(labels * F.log_softmax(logits_i2t, dim=-1)).sum() / batch_size
        # loss_t2i = -(labels * F.log_softmax(logits_t2i, dim=-1)).sum() / batch_size
        # print(labels * F.log_softmax(logits_i2t, dim=-1))
        # print(labels * F.log_softmax(logits_t2i, dim=-1))

        targets = torch.linspace(0, batch_size-1, batch_size, dtype=int).to(embed_vis.device)
        loss_i2t = F.cross_entropy(logits_i2t, targets, label_smoothing=0.1)
        loss_t2i = F.cross_entropy(logits_t2i, targets, label_smoothing=0.1)

        loss_itc = alpha_i2t * loss_i2t +\
                   (1-alpha_i2t) * loss_t2i

        return loss_itc
    
    
class VisionContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.5, learnable_temp=False):
        super().__init__()

        if learnable_temp:
            self.temperature = nn.Parameter(temperature*torch.ones([]))
        else:
            self.temperature = torch.as_tensor(temperature)

    def forward(self, embed_1, embed_2):

        temperature = self.temperature.to(embed_1.device)
        batch_size = embed_1.shape[0]

        negatives_mask = ~torch.eye(batch_size*2, batch_size*2, dtype=torch.bool)
        negatives_mask = torch.clone(negatives_mask.type(torch.float)).to(embed_1.device)

        embeds = torch.cat([embed_1, embed_2], dim=0)
        sim_matrix = F.cosine_similarity(embeds.unsqueeze(1), embeds.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives/temperature)
        denominator = negatives_mask * torch.exp(sim_matrix/temperature)

        loss_partial = -torch.log(nominator/torch.sum(denominator, dim=1))

        return torch.sum(loss_partial) / (2*batch_size)


class HierarchicalCLIPLoss(nn.Module):

    def __init__(self, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"):
        super().__init__()
        
        self.agg = agg
        self.temp1 = temp1
        self.temp2 = temp2
        self.temp3 = temp3

    def forward(self, vis_embeds, txt_embeds_list):
        """
        vis_embeds [bs, h, w, d, embed_size]
        txt_embeds_list: a list of bs tensors [token_len (word or sentence), embed_size]
        """

        att_maps = []
        similarities = []

        batch_size = vis_embeds.shape[0]
        # [bs, h, w, d, embed_size]
        vis_embeds = F.normalize(vis_embeds, dim=-1)
        # [bs, embed_size, h, w, d]
        vis_embeds = vis_embeds.permute(0, 4, 1, 2, 3).contiguous()

        for i in range(batch_size):
            # [bs, num_tokens, embed_size]
            txt_embeds = txt_embeds_list[i].unsqueeze(0).repeat(batch_size, 1, 1).contiguous()
            # if torch.isnan(txt_embeds).any():
            #     pass
            #     print(txt_embeds[0, -1, :])
            txt_embeds = F.normalize(txt_embeds, dim=-1)
            txt_embeds = txt_embeds.permute(0, 2, 1).contiguous() # [bs, embed_size, num_tokens]
            # print(vis_embeds.size(), txt_embeds.size())

            num_tokens = txt_embeds.shape[-1]

            weiContext, attn = self.attention_fn(
                txt_embeds, vis_embeds, self.temp1
            )  # [48, 512, 25], [48, 25, 6, 6, 6]
            # print(weiContext.size(), attn.size())
            # return

            att_maps.append(
                attn[i].unsqueeze(0).contiguous()
            )  # add attention for curr index  [25, 6, 6, 6]
            txt_embeds = txt_embeds.transpose(1, 2).contiguous()  # [48, 25, 512]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 512]

            txt_embeds = txt_embeds.view(batch_size * num_tokens, -1)  # [1200, 512]
            weiContext = weiContext.view(batch_size * num_tokens, -1)  # [1200, 512]

            row_sim = self.custom_cosine_similarity(txt_embeds, weiContext)
            row_sim = row_sim.view(batch_size, num_tokens)  # [48, 25]

            row_sim.mul_(self.temp2).exp_()
            if self.agg == "sum":
                row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
            else:
                row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
            row_sim = torch.log(row_sim)

            # if torch.isnan(row_sim).any():
            #     print(row_sim)
            #     # print(txt_embeds[0])
            #     # print(weiContext[0])
            #     pass

            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)  #
        similarities = similarities * self.temp3
        similarities1 = similarities.transpose(0, 1)  # [48, 48]
        # print(similarities.size())

        labels = torch.LongTensor(range(batch_size)).to(similarities.device)

        loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        # if torch.isnan(loss0) or torch.isnan(loss1):
            # print(similarities)
            # print(loss0, loss1)
            # pass
        loss = (loss0 + loss1) / 2.0
        return loss, att_maps


    @staticmethod
    def custom_cosine_similarity(x1, x2, dim=1, eps=1e-8):
        """Returns cosine similarity between x1 and x2, computed along dim."""
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


    @staticmethod
    def attention_fn(query, context, temp1):
        """
        query: batch x ndf x queryL
        context: batch x ndf x h x w, d (sourceL=h x w x d)
        mask: batch_size x sourceL
        """
        batch_size, queryL = query.size(0), query.size(2)
        h, w, d = context.size(2), context.size(3), context.size(4)
        sourceL = h * w * d

        # --> batch x sourceL x ndf
        context = context.view(batch_size, -1, sourceL)
        contextT = torch.transpose(context, 1, 2).contiguous()

        # Get attention
        # (batch x sourceL x ndf)(batch x ndf x queryL)
        # -->batch x sourceL x queryL
        attn = torch.bmm(contextT, query)
        # --> batch*sourceL x queryL
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax(dim=-1)(attn)

        # --> batch x sourceL x queryL
        attn = attn.view(batch_size, sourceL, queryL)
        # --> batch*queryL x sourceL
        attn = torch.transpose(attn, 1, 2).contiguous()
        attn = attn.view(batch_size * queryL, sourceL)

        attn = attn * temp1
        attn = nn.Softmax(dim=-1)(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # (batch x ndf x sourceL)(batch x sourceL x queryL)
        # --> batch x ndf x queryL
        weightedContext = torch.bmm(context, attnT)
        if torch.isnan(weightedContext).any():
            # print(contextT[0, :, 0])
            # print(attnT[0, 0, :])
            # print(query[0, :, -1])
            pass

        return weightedContext, attn.view(batch_size, -1, h, w, d).contiguous()


if __name__ in "__main__":

    device = "cuda"
    contras_criterion = VisionContrastiveLoss(temperature=0.5, learnable_temp=True)
    embed_1 = torch.randn(32, 512).to(device)
    embed_2 = torch.randn(32, 512).to(device)

    loss = contras_criterion(embed_1, embed_2)
    print(loss, loss.device, contras_criterion.temperature)


