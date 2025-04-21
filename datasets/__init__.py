import torch


def collate_fn(batch):

    ### aggregate each item of input list by its key to form a new batch [dict] ###
    final_dict = {}
    for single_dict in batch:
        for key, value in single_dict.items():
            if key not in final_dict.keys():
                final_dict[key] = list()
            final_dict[key].append(value)

    ### stack value under each key by its data type ###
    for key, value in final_dict.items():
        if "volume" in key:
            final_dict[key] = torch.stack(value)
        elif "answer" in key:
            pass
            # final_dict[key] = torch.stack(value)
        else:
            final_dict[key] = torch.cat(value, dim=0)

    return final_dict