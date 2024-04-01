import argparse
import logging
import os
import sys
import time
import logging
import os
import sys
sys.path.append("..")
sys.path.append("../..")
import torch
from transformers import BertConfig
from util.util import get_files_paths_from_directory
from model.model import TBertT,TBertSI, TBertTNoText
from util.data_util import get_tag_encoder, get_fixed_tag_encoder, load_data_to_dataset, get_dataloader, load_tenor_data_to_dataset
from torch.utils.data import DataLoader
import numpy as np
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import csv

to_csv = []
def evaluate_ori(pred, label, topk,mlb=None):
    """
    dimension of pred and label should be equal.
    :param pred: a list of prediction
    :param label: a list of true label
    :param topk:
    :return: a dictionary: {'precision': pre_k, 'recall': rec_k, 'f1': f1_k}
    """
    top_idx_list = sorted(range(len(pred)), key=lambda i: pred[i])[-topk:]
    top_idx_one_hot = [0] * len(pred)
    for idx in top_idx_list:
        top_idx_one_hot[idx] = 1
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
    # new_dict = dict()
    # new_dict['top'] = topk
    # new_dict['precision'] = pre_k
    # new_dict['recall'] = rec_k
    # new_dict['f1'] = f1_k
    # new_dict['predict_tag'] = mlb.inverse_transform(np.array([top_idx_one_hot]))
    # new_dict['true_tag'] = mlb.inverse_transform(np.array([label]))
    # to_csv.append(new_dict)
    # logger.info("Loging dict ---> {0}".format(new_dict))
    return pre_k, rec_k, f1_k


def evaluate_batch(pred, label, topk_list=[1, 2, 3, 4, 5], mlb=None):
    pre = [0.0] * len(topk_list)
    rc = [0.0] * len(topk_list)
    f1 = [0.0] * len(topk_list)
    cnt = 0
    for i in range(0, len(pred)):
        for idx, topk in enumerate(topk_list):
            pre_val, rc_val, f1_val = evaluate_ori(
                pred=pred[i], label=label[i], topk=topk,mlb=mlb)
            if pre_val == -1:
                cnt -= 1
            pre[idx] += pre_val
            rc[idx] += rc_val
            f1[idx] += f1_val
        cnt += 1
    pre[:] = [x / cnt for x in pre]
    rc[:] = [x / cnt for x in rc]
    f1[:] = [x / cnt for x in f1]
    return [pre, rc, f1, cnt]


def avg(data):
    import numpy as np
    a = np.array(data)
    res = np.average(a, axis=0)
    return res
def test(args, model, test_set,mlb):
    test_data_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             )
    with torch.no_grad():
        model.eval()
        fin_pre = []
        fin_rc = []
        fin_f1 = []
        fin_cnt = 0
        for batch_idx, data in enumerate(test_data_loader, 0):
            fin_outputs = []
            fin_targets = []
            title_ids = data['titile_ids'].to(
                args.device, dtype=torch.long)
            title_mask = data['title_mask'].to(
                args.device, dtype=torch.long)
            code_ids = data['code_ids'].to(
                args.device, dtype=torch.long)
            code_mask = data['code_mask'].to(
                args.device, dtype=torch.long)
            targets = data['labels'].to(
                args.device, dtype=torch.float)

            outputs = model(title_ids=title_ids,
                            title_attention_mask=title_mask,
                            code_ids=code_ids,
                            code_attention_mask=code_mask)

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(
                outputs).cpu().detach().numpy().tolist())
            [pre, rc, f1, cnt] = evaluate_batch(
                fin_outputs, fin_targets, [1, 2, 3, 4, 5],mlb)
            fin_pre.append(pre)
            fin_rc.append(rc)
            fin_f1.append(f1)
            fin_cnt += cnt
            logger.info("Final F1 Score = {}".format(f1))
            logger.info("Final Recall Score  = {}".format(rc))
            logger.info("Final Precision Score  = {}".format(pre))
            logger.info("Final Count  = {}".format(cnt))
        avg_pre = avg(fin_pre)
        avg_rc = avg(fin_rc)
        avg_f1 = avg(fin_f1)
        logger.info("Final File F1 Score = {}".format(avg_f1))
        logger.info("Final File Recall Score  = {}".format(avg_rc))
        logger.info("Final File Precision Score  = {}".format(avg_pre))
        logger.info("Final File Count  = {}".format(fin_cnt))
    return [avg_pre, avg_rc, avg_f1, fin_cnt]

def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="../../data/test", type=str,
        help="The input test data dir.")
    
    parser.add_argument("--model_path", default="../../data/results/microsoft/codebert-base_01-02-03-14-13_code/epoch-0-file-499/t_bert.pt", help="The model to evaluate")
    # parser.add_argument("--model_path", default="../../data/results/triplet_12-07 15-29-36_/final_model-199/t_bert.pt", help="The model to evaluate")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--no_code", action="store_true", help="Whether to include code in the model")
    parser.add_argument("--vocab_file", default="../../data/tags/commonTags_post2vec.csv", type=str,
                        help="The tag vocab data file.")
    parser.add_argument("--verbus", action="store_true", help="show more logs")
    parser.add_argument("--mlb_latest", action="store_true", help="use the latest mlb")
    parser.add_argument("--test_batch_size", default=500, type=int,help="batch size used for testing")
    parser.add_argument("--output_dir", default="./logs", help="directory to store the results")
    parser.add_argument("--code_bert", default='microsoft/codebert-base',
                        choices=['microsoft/codebert-base', 'huggingface/CodeBERTa-small-v1',
                                 'codistai/codeBERT-small-v2', 'albert-base-v2','jeniya/BERTOverflow', 'roberta-base',
                                 'bert-base-uncased'])
    parser.add_argument("--log_result", action="store_true", help="wheather to store the test result in a csv file")
    parser.add_argument("--model_type", default="triplet", choices=["triplet","siamese"])
    args = parser.parse_args()
    return args
def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_eval_args()
    logging.info("Start Testing") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    # get the encoder for tags
    if args.mlb_latest == True:
        logger.info("use new mlb tagger")
        mlb, num_class = get_fixed_tag_encoder(args.vocab_file)
    else:
        mlb, num_class = get_tag_encoder(args.vocab_file)
    args.mlb = mlb
    args.num_class = num_class
    
    if args.model_type == "triplet":
        model = TBertTNoText(BertConfig(), args.code_bert, num_class)
    elif args.model_type == "siamese":
        model = TBertSI(BertConfig(), args.code_bert, num_class)
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    if args.code_bert == "microsoft/codebert-base":
        args.model_path = "../../data/results/microsoft/codebert-base_01-05-20-50-50_text/epoch-0-file-499/t_bert.pt"

    if args.model_path and os.path.exists(args.model_path):
        model_path = os.path.join(args.model_path, )
        model.load_state_dict(torch.load(model_path)) 
    logger.info("model loaded")

    fin_pre = []
    fin_rc = []
    fin_f1 = []
    fin_cnt = 0
    files = get_files_paths_from_directory(args.data_dir)

    logger.info("***** Running testing *****")
    logger.info("device %s",args.device)
    
    for file_cnt in range(len(files)):
        logger.info("load file {}".format(file_cnt))
        test_set = load_tenor_data_to_dataset(args.mlb, files[file_cnt])
        [pre, rc, f1, cnt] = test(args, model, test_set, mlb)
        fin_pre.append(pre)
        fin_rc.append(rc)
        fin_f1.append(f1)
        fin_cnt += cnt 
    
    avg_pre = avg(fin_pre)
    avg_rc = avg(fin_rc)
    avg_f1 = avg(fin_f1)
    logger.info("Final F1 Score = {}".format(avg_f1))
    logger.info("Final Recall Score  = {}".format(avg_rc))
    logger.info("Final Precision Score  = {}".format(avg_pre))
    logger.info("Final Count  = {}".format(fin_cnt))
    logger.info("Test finished")
    # keys = to_csv[0].keys()

    # with open('./logs/result.csv', 'w', newline='') as output_file:
    #     dict_writer = csv.DictWriter(output_file, keys)
    #     dict_writer.writeheader()
    #     dict_writer.writerows(to_csv)
if __name__ == "__main__":
    main()