import torch
import torch.nn.functional as F

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)

from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def compute_score_matrix(vis_embeds, txt_embeds, k_test=256, requires_itm=False):
    
    num_vis_embeds, num_txt_embeds = vis_embeds.shape[0], txt_embeds.shape[0]

    vis_embeds_normalized = F.normalize(vis_embeds, dim=-1)
    txt_embeds_normalized = F.normalize(txt_embeds, dim=-1)

    ### vision to text score matrix ###
    sims_matrix_v2t = vis_embeds_normalized @ txt_embeds_normalized.t()
    # print(sims_matrix_v2t.shape)
    score_matrix_v2t = torch.full((num_vis_embeds, num_txt_embeds), -100.0).to(vis_embeds.device)
    for i, sims in enumerate(sims_matrix_v2t):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

        if requires_itm:
            match_score = 0.
        else:
            match_score = 0.

        score_matrix_v2t[i, topk_idx] = topk_sim + match_score

    ### text to vision score matrix ###
    sims_matrix_t2v = sims_matrix_v2t.t()
    score_matrix_t2v = torch.full((num_txt_embeds, num_vis_embeds), -100.0).to(vis_embeds.device)
    for i, sims in enumerate(sims_matrix_t2v):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        
        if requires_itm:
            match_score = 0.
        else:
            match_score = 0.

        score_matrix_t2v[i, topk_idx] = topk_sim + match_score

    return score_matrix_v2t.numpy(), score_matrix_t2v.numpy()



# retrieval metric R@k and P@K
def compute_RatK(sim_matrix, k=1):
    N = sim_matrix.shape[0]
    correct_matches = 0
    
    for i in range(N):
        sorted_indices = np.argsort(-sim_matrix[i])
        
        if i in sorted_indices[:k]:
            correct_matches += 1
            
    return correct_matches / N
    

def compute_PatK(score_matrix, mm_dict, k=1):
    numcases = score_matrix.shape[0]
    res = []
    
    for index, score in enumerate(score_matrix):
        # order = ord[i]
        inds = np.argsort(score)[::-1]
        correct_indices = mm_dict[index]

        p = 0.0
        r = 0.0
        for j in range(k):
            if inds[j] in correct_indices:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]

    return np.mean(res)

def multiclass_cls_metrics(probs, ground_truth):
    
    # Convert probabilities into predicted classes
    preds = np.argmax(probs, axis=1)
    # print(preds)

    accuracy = accuracy_score(ground_truth, preds)

    # AUC (One-vs-Rest, assuming multiclass problem)
    try:
        auc = roc_auc_score(ground_truth, probs, multi_class='ovr')
    except ValueError:
        auc = None 

    f1 = f1_score(ground_truth, preds, average='macro')
    conf_matrix = confusion_matrix(ground_truth, preds)
    sensitivity, specificity, ppv, npv = [], [], [], []
    
    for i in range(conf_matrix.shape[0]):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)

        sens = TP / (TP + FN) if TP + FN > 0 else 0
        spec = TN / (TN + FP) if TN + FP > 0 else 0
        ppv_val = TP / (TP + FP) if TP + FP > 0 else 0
        npv_val = TN / (TN + FN) if TN + FN > 0 else 0

        sensitivity.append(sens)
        specificity.append(spec)
        ppv.append(ppv_val)
        npv.append(npv_val)

    sensitivity = np.mean(sensitivity)
    specificity = np.mean(specificity)
    ppv = np.mean(ppv)
    npv = np.mean(npv)

    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
    }

    return metrics

def binary_cls_metrics(probs, ground_truth):

    preds = (probs > 0.5).astype(int)

    accuracy = accuracy_score(ground_truth, preds)

    auc = roc_auc_score(ground_truth, probs)

    f1 = f1_score(ground_truth, preds)

    conf_matrix = confusion_matrix(ground_truth, preds)
    TN, FP, FN, TP = conf_matrix.ravel()

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
    }

    return metrics

def nlg_metrics(pred_captions, gt_captions):

    N = len(pred_captions)
    predictions = {i: [pred_captions[i]] for i in range(N)}

    if isinstance(gt_captions[0], str):
        ground_truths = {i: [gt_captions[i]] for i in range(N)}
    elif isinstance(gt_captions[0], list):
        ground_truths = {i: [*gt_captions[i]] for i in range(N)}

    B4 = Bleu(n=4)  # To calculate BLEU-1 to BLEU-4
    C = Cider()
    # S = Spice()
    M = Meteor()
    R = Rouge()


    b4_score, _ = B4.compute_score(ground_truths, predictions)
    c_score, _ = C.compute_score(ground_truths, predictions)
    # s_score, _ = S.compute_score(ground_truths, predictions)
    m_score, _ = M.compute_score(ground_truths, predictions)
    r_score, _ = R.compute_score(ground_truths, predictions)

    return {
        "B@1": b4_score[0],
        "B@2": b4_score[1],
        "B@3": b4_score[2],
        "B@4": b4_score[3],
        "CIDEr": c_score,
        # "SPICE": s_score,
        "METEOR": m_score,
        "ROUGE_L": r_score
    }
    