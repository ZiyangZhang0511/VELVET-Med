import math

import torch
import torch.nn as nn

from transformers import BertConfig
from transformers.activations import ACT2FN
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.bert.modeling_bert import  BertPreTrainedModel

from typing import List, Optional, Tuple, Union


class DeconvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_upsample=5):
        super(DeconvBlock, self).__init__()

        self.up_blocks = nn.ModuleList([
            nn.ConvTranspose3d(
                in_channels//(2**(i)), in_channels//(2**(i+1)), kernel_size=2, stride=2
            )
            for i in range(num_upsample-1)
        ])

        self.out_layer = nn.ConvTranspose3d(
                in_channels//(2**(num_upsample-1)), out_channels, kernel_size=2, stride=2
            )

    def forward(self, x):

        for block in self.up_blocks:
            x = block(x)

        return self.out_layer(x)


class UNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_upsample=5):
        super(UNetBlock, self).__init__()

        self.up_blocks = nn.ModuleList([
            self.make_up_layer(in_channels//(2**(i)), in_channels//(2**(i+1)))
            for i in range(num_upsample)
        ])

        self.out_layer = nn.Conv3d(in_channels//(2**(num_upsample)), out_channels, kernel_size=1, stride=1)

    def make_up_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )

    def forward(self, x):

        for block in self.up_blocks:
            x = block(x)

        return self.out_layer(x)
        


class SentBertEmbeddings(nn.Module):
    def __init__(self, config):
        super(SentBertEmbeddings, self).__init__()
        # print("sent bert emb", config)

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"], padding_idx=config["pad_token_id"])
        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], config["hidden_size"], padding_idx=0)

        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.register_buffer(
            "position_ids", torch.arange(config["max_position_embeddings"]).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.ones(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids,
        token_type_ids,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
            # print("input_shape in inputs_embeds", inputs_embeds.size())
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length+past_key_values_length]
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings # [bs, seq_len, hidden_size]


class SentBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class SentBertSelfAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config["hidden_size"] % config["num_attention_heads"] != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config['hidden_dize']}) is not a multiple of the number of attention "
                f"heads ({config['num_attention_heads']})"
            )

        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = int(config["hidden_size"] / config["num_attention_heads"])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config["hidden_size"], self.all_head_size)
        self.key = nn.Linear(config["hidden_size"], self.all_head_size)
        self.value = nn.Linear(config["hidden_size"], self.all_head_size)

        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config["max_position_embeddings"]
            self.distance_embedding = nn.Embedding(2*config["max_position_embeddings"]-1, self.attention_head_size)

        self.is_decoder = config["is_decoder"]

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: x of shape (bs, seq_len, hz)
        output: shape (bs, num_head, seq_len, hz_each_head)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape).contiguous()
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor, # (bs, seq_len, hz)
        attention_mask: Optional[torch.FloatTensor] = None, # (bs, seq_len)
        head_mask: Optional[torch.FloatTensor] = None, # (num_heads,)
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # (bs, seq_len_sec, hz)
        encoder_attention_mask: Optional[torch.FloatTensor] = None, # (bs, seq_len_sec)
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None, # a tuple of 2 tensors of shape (bs, num_heads, seq_len, hz_each_head)
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states) # shape (bs, seq_len, hz)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0] # a tensor of shape (bs, num_heads, seq_len, hz_each_head)
            value_layer = past_key_value[1] # a tensor of shape (bs, num_heads, seq_len, hz_each_head)
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        # query_layer of shape (bs, num_head, seq_len, hz_each_head)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores of shape (bs, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # attention_scores of shape (bs, num_heads, q_len, k_len)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs of shape (bs, num_heads, q_len, k_len)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context_layer of shape (bs, num_heads, q_len, hz_each_head)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape (bs, q_len, hz)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape).contiguous()

        # outputs of a tuple of context_layer (bs, q_len, hz), 
        # attention_probs (bs, num_heads, q_len, k_len)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        # outputs: (context_layer, attention_probs, past_key_value(tuple of 2 tensors))
        return outputs


class SentBertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # self.self = BERT_SELF_ATTENTION_CLASSES[config._attn_implementation](
        #     config, position_embedding_type=position_embedding_type
        # )
        self.self = SentBertSelfAttention(
            config, position_embedding_type=position_embedding_type
        )

        self.output = SentBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor, # (bs, seq_len, hz)
        attention_mask: Optional[torch.FloatTensor] = None,# (bs, seq_len)
        head_mask: Optional[torch.FloatTensor] = None, # (num_heads,)
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # (bs, seq_len_enc, hz)
        encoder_attention_mask: Optional[torch.FloatTensor] = None, # (bs, seq_len_enc)
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None, # a tuple of 2 tensors of shape (bs, num_heads, seq_len, hz_each_head)
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # self_outputs: a tuple: context_layer (bs, q_len, hz), 
        # attention_probs (bs, num_heads, q_len, k_len), past_key_value (tuple of 2 tensors (bs, num_heads, seq_len, hz_each_head))
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # attention_output shape of (bs, seq_len, hz)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # outputs: (attention_output, attention_probs, past_key_value(tuple of 2 tensors))
        return outputs


class SentBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["intermediate_size"])
        if isinstance(config["hidden_act"], str):
            self.intermediate_act_fn = ACT2FN[config["hidden_act"]]
        else:
            self.intermediate_act_fn = config["hidden_act"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SentBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SentBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config["chunk_size_feed_forward"]
        self.seq_len_dim = 1
        # self.attention is for self-attention mechanism, without encoder_hidden_states info
        self.attention = SentBertAttention(config) # to see if sentence modeling is needed !!!
        self.is_decoder = config["is_decoder"]
        self.add_cross_attention = config["add_cross_attention"]
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = SentBertAttention(config, position_embedding_type="absolute")
        self.intermediate = SentBertIntermediate(config)
        self.output = SentBertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor, # (bs, seq_len, hidden_size)
        attention_mask: Optional[torch.FloatTensor] = None, # (bs, seq_len)
        head_mask: Optional[torch.FloatTensor] = None, # (num_heads,)
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # (bs, seq_len_enc, hidden_size)
        encoder_attention_mask: Optional[torch.FloatTensor] = None, # (bs, seq_len_enc)
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None, # a tuple of 4 tensors of shape (bs, num_heads, seq_len, hz_each_head)
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # past_key_value (tuple) only for self-attention mechanism
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        ### specify the shape of self_attention_outputs ###
        # self_attention_outputs: (
        #   attention_output, (bs, seq_len, hz)
        #   attention_probs, (bs, num_heads. q_len, k_len)
        #   past_key_value(tuple of 2 tensors), (bs, num_heads, k\v_len, hz_each_head)
        #)
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value, # a tuple of 2 tensors of shape (bs, num_heads, seq_len, hz_each_head)
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1] # attention_probs
            present_key_value = self_attention_outputs[-1] # a tuple of 2 tensors
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            
            # cross_attention_outputs: (
            #   attention_output, (bs, seq_len, hz)
            #   attention_probs, (bs, num_heads. q_len, k_len)
            #   past_key_value (tuple of 2 tensors), (bs, num_heads, k\v_len, hz_each_head)
            #)
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0] # attention_output, (bs, seq_len, hz)
            # outputs: (self-atten probs, cross-atten probs)
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            # present_key_value a tuple of 4 tensors (self-atten key_value, cross-atten key_value)
            present_key_value = present_key_value + cross_attn_present_key_value

        # layer_output of shape (bs, seq_len, hz)
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # outputs: (layer_output, self-atten probs, or plus cross-atten probs)
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



class SentBertEncoder(nn.Module):

    def __init__(self, config):
        super(SentBertEncoder, self).__init__()

        self.config = config
        self.gradient_checkpointing = False

        if config["num_decoder_layers"] > 0:
            self.layer = nn.ModuleList([])
            for i in range(config["num_hidden_layers"]):
                if i < (config["num_hidden_layers"] - config["num_decoder_layers"]):
                    ### add only self-attention layer
                    self.layer.append(SentBertLayer(config))
                    
                else:
                    ### add self-attention plus cross-attention layer
                    decoder_layer_config = config.copy()
                    decoder_layer_config["add_cross_attention"] = True
                    decoder_layer_config["is_decoder"] = True
                    self.layer.append(SentBertLayer(decoder_layer_config))
        else:
            self.layer = nn.ModuleList([SentBertLayer(config) for _ in range(config["num_hidden_layers"])])
            
    
    def forward(
        self,
        hidden_states: torch.Tensor, # (bs, seq_len, hidden_size)
        attention_mask: Optional[torch.FloatTensor] = None, # (bs, seq_len)
        head_mask: Optional[torch.FloatTensor] = None, # (num_layers, num_heads)
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # (bs, seq_len_enc, hidden_size)
        encoder_attention_mask: Optional[torch.FloatTensor] = None, # (bs, seq_len_enc)
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, # each sub-tuple having 4 tensors of shape (bs, num_heads, seq_len-1, embed_size_per_head)
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        start_layer=0,
        end_layer=11,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        # all_cross_attentions = () if output_attentions and self.config["add_cross_attention"] else None
        all_cross_attentions = () if output_attentions else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None

        # for i, layer_module in enumerate(self.layer):
        for i in range(start_layer, end_layer+1):
            layer_module = self.layer[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # layer_head_mask of shape [1(bs), num_heads, 1(seq_len), 1(seq_len)]
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # past_key_value at i-th layer, a tuple of 4 tensors of shape (bs, num_heads, seq_len-1, embed_size_per_head)
            past_key_value = past_key_values[i] if past_key_values is not None else None

            ### specify shape of layer_outputs ###
            # layer_outputs: (
            #   hidden_states: (bs, seq_len, hz), 
            #   self-atten probs (bs, num_heads. q_len, k_len)), 
            #   or plus cross-atten probs,
            #   a tuple of (self-atten present_key_value (bs, num_heads, k\v_len, hz_each_head) , or plus cross-atten present_key_value),
            # )
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value, # tuple of 4 tensors
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config["add_cross_attention"] or len(layer_outputs) >= 3:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



class SentBertModel(BertPreTrainedModel):

    def __init__(self, config):

        official_config = BertConfig()
        super(SentBertModel, self).__init__(official_config)

        self.config = config
        self.embeddings = SentBertEmbeddings(config)
        self.encoder = SentBertEncoder(config)
        self.pooler = None

        if config["sentence_modeling"]:
            print("enable sentence tokens in text model...")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None, # (bs, seq_len)
        attention_mask: Optional[torch.Tensor] = None, # (bs, seq_len)
        token_type_ids: Optional[torch.Tensor] = None, # (bs, seq_len)
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None, # (num_heads,)
        inputs_embeds: Optional[torch.Tensor] = None, # (bs, seq_len, hz)
        encoder_hidden_states: Optional[torch.Tensor] = None, # (bs, seq_len_enc, hz)
        encoder_attention_mask: Optional[torch.Tensor] = None,# (bs, seq_len_enc)
        past_key_values: Optional[List[torch.FloatTensor]] = None, # 12 tuples of a tuple of 4 tensors (self-atten k, v or plus cross-atten k, v)  (bs, num_heads, k\v_len, hz_each_head)
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mode="txt_mlm",
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config["output_attentions"]
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config["output_hidden_states"]
        )
        return_dict = return_dict if return_dict is not None else self.config["use_return_dict"]

        # set use_cache to Fasle so that I don't need to store past_key_values during training and inference
        if self.config["is_decoder"]:
            use_cache = use_cache if use_cache is not None else self.config["use_cache"]
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.ones(input_shape, dtype=torch.long, device=device)

        # embedding_output of shape (bs, seq_len, hz)
        if inputs_embeds is not None:
            # print("No use of embedding layers...")
            embedding_output = inputs_embeds
        else:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        # print("embeddings size:", embedding_output.size())


        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length+past_key_values_length), device=device)

        # use_sdpa_attention_masks = (
        #     self.attn_implementation == "sdpa"
        #     and self.position_embedding_type == "absolute"
        #     and head_mask is None
        #     and not output_attentions
        # )
        use_sdpa_attention_masks = None

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.config["is_decoder"]:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            
            # attention_mask of shape (bs, seq_len), input_shape (bs, seq_len)
            # extended_attention_mask of (bs, 1, 1, seq_len) with value 0.0 or -10000.0
            # !!! add a new mask mechanism into get_extended_attention_mask() for sentence modeling
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, token_type_ids)


        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config["is_decoder"] or encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

            # print(encoder_extended_attention_mask.size())
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask of shape [num_hidden_layers, 1(bs), num_heads, 1(seq_len), 1(seq_len)]
        head_mask = self.get_head_mask(head_mask, self.config["num_hidden_layers"])


        if mode == "txt_mlm":
            start_layer = 0
            end_layer = self.config["layer_for_txt_ssl"]-1
        elif mode == "mm_match":
            start_layer = self.config["layer_for_txt_ssl"]
            end_layer = self.config["num_hidden_layers"]-1
        elif mode == "mm_mlm":
            start_layer = 0
            end_layer = self.config["num_hidden_layers"]-1

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor, # shape (bs, seq_len)
        input_shape: Tuple[int], # (bs, seq_len)
        token_type_ids:torch.Tensor = None, # shape (bs, seq_len)
        device: torch.device = None,
        dtype: torch.float = None,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.config["is_decoder"]):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model need to model [SENT] tokens, apply a customized mask !!!
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config["is_decoder"]:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            # !!! TO DO !!!
            elif self.config["sentence_modeling"]:
                # shape: (bs, 1, seq_len, seq_len)
                extended_attention_mask = self.create_extended_attention_mask_for_sentence_modeling(
                    attention_mask,
                    token_type_ids,
                    device,
                )
                # print("sentence_modeling", extended_attention_mask.size())
                # print(attention_mask[0])
                # print(token_type_ids[0])
                # print(extended_attention_mask[0][0][10])
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
                # print("without sentence_modeling")
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask


    @staticmethod
    def create_extended_attention_mask_for_sentence_modeling(
        attention_mask: torch.Tensor, # shape (bs, seq_len),
        token_type_ids: torch.Tensor, # shape (bs, seq_len),
        device: torch.device = None,
    ):

        batch_size, seq_len = token_type_ids.size()
        extented_attention_mask = attention_mask[:, None, :]
        
        device = attention_mask.device

        sentence_mask = []
        for i in range(batch_size):
            cur_token_type_id = token_type_ids[i]
            cur_attention_mask = extented_attention_mask[i]
            cur_sentence_mask = torch.ones((seq_len, seq_len), dtype=attention_mask.dtype, device=device)

            diff_mask = cur_token_type_id[1:] != cur_token_type_id[:-1]
            transition_indices = diff_mask.nonzero(as_tuple=True)[0] + 1
            transition_indices = torch.cat([torch.tensor([1], device=device), transition_indices], dim=0)
            transition_indices[-1] = seq_len
            # print(transition_indices)

            for j in range(len(transition_indices)-1):
                start_idx = transition_indices[j]
                end_idx = transition_indices[j+1]
                cur_sentence_mask[start_idx][1:start_idx] = 0
                cur_sentence_mask[start_idx][end_idx:] = 0

            cur_sentence_mask *= cur_attention_mask
            sentence_mask.append(cur_sentence_mask)
            # print(cur_sentence_mask[14])
            # return

        sentence_mask = torch.stack(sentence_mask, dim=0) # shape (bs, seq_len, seq_len) 
        # print(sentence_mask.size())

        sentence_mask = sentence_mask[:, None, :, :]

        return sentence_mask

class MLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(config["hidden_size"], config["hidden_size"]),
            ACT2FN[config["hidden_act"]],
            nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"]),
        )
        self.output = nn.Linear(config["hidden_size"], config["vocab_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.fc(x))


if __name__ == "__main__":
    # x = torch.randn((4, 768, 3, 3, 3))
    # net = DeconvBlock(768, 1,  num_upsample=5)
    # print(net(x).shape)
    # print(net)

    token_type_id = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 0, 0, 0, 0, 0])
    token_type_ids = token_type_id[None, :].expand(10, -1)
    attention_mask = (token_type_ids != 0).int()
    SentBertModel.create_extended_attention_mask_for_sentence_modeling(
        attention_mask,
        token_type_ids,
    )