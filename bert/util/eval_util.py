
import logging
logger = logging.getLogger(__name__)

def evaluate_ori(pred, label, topk):
    """
    dimension of pred and label should be equal.
    :param pred: a list of prediction
    :param label: a list of true label
    :param topk:
    :return: a dictionary: {'precision': pre_k, 'recall': rec_k, 'f1': f1_k}
    """
    top_idx_list = sorted(range(len(pred)), key=lambda i: pred[i])[-topk:]
    num_of_true_in_topk = len([idx for idx in top_idx_list if label[idx] == 1])
    # precision@k = #true label in topk / k
    pre_k = num_of_true_in_topk / float(topk)
    # recall@k = #true label in topk / #true label
    num_of_true_in_all = sum(label)
    if num_of_true_in_all == 0:
        return -1,-1,-1
    if num_of_true_in_all > topk:
        rec_k = num_of_true_in_topk / float(topk)
    else:
        if num_of_true_in_all == 0:
            return -1,-1,-1
        rec_k = num_of_true_in_topk / float(num_of_true_in_all)
    # f1@k = 2 * precision@k * recall@k / (precision@k + recall@k)
    if pre_k == 0 and rec_k == 0:
        f1_k = 0.0
    else:
        f1_k = 2 * pre_k * rec_k / (pre_k + rec_k)
    # return {'precision': pre_k, 'recall': rec_k, 'f1': f1_k}
    return pre_k, rec_k, f1_k


def evaluate_batch(pred, label, topk_list=[1, 2, 3, 4, 5]):
    pre = [0.0] * len(topk_list)
    rc = [0.0] * len(topk_list)
    f1 = [0.0] * len(topk_list)
    cnt = 0
    flag = 0
    for i in range(0, len(pred)):
        for idx, topk in enumerate(topk_list):
            pre_val, rc_val, f1_val = evaluate_ori(
                pred=pred[i], label=label[i], topk=topk)
            if pre_val == -1:
                cnt -= 1
                break
            pre[idx] += pre_val
            rc[idx] += rc_val
            f1[idx] += f1_val
        cnt += 1
    pre[:] = [x / cnt for x in pre]
    rc[:] = [x / cnt for x in rc]
    f1[:] = [x / cnt for x in f1]
    return [pre, rc, f1, cnt]
