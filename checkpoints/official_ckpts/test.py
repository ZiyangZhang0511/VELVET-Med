import torch



if __name__ == "__main__":
    # print(torch.hub.list("pytorch/vision"))

    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased') 
    # print(model)
    torch.save(model.state_dict(), "./checkpoints/official_ckpts/bert_base_uncased.pth")