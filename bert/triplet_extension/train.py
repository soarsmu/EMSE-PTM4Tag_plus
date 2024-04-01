import sys
sys.path.append("../")
sys.path.append("/usr/src/bert")
import torch
from torch.optim import AdamW
from transformers import BertConfig, get_linear_schedule_with_warmup
from transformers import AlbertConfig, RobertaConfig
from datetime import datetime
from model.model import TBertT,TBertSI, TBertTNoCode,TBertTNoTitle, TBertTNoText
import logging
import argparse
from util.data_util import get_fixed_tag_encoder
from util.util import seed_everything

logger = logging.getLogger(__name__)

def get_exe_name(args):
    exe_name = "{}_{}_{}"
    time = datetime.now().strftime("%m-%d-%H-%M-%S")
    component = args.remove_component
    return exe_name.format(args.code_bert, time, component)
def get_optimizer(args,model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer
def get_optimizer_scheduler(args, model, train_steps):
    optimizer = get_optimizer(args, model)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=train_steps
    )
    return optimizer, scheduler

def init_train_env(args, tbert_type):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    seed_everything(args.seed)
    
    # get the encoder for tags
    mlb, num_class = get_fixed_tag_encoder(args.vocab_file)
    args.mlb = mlb
    args.num_class = num_class
    
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
        
    logger.info("tbert_type architectue {}".format(tbert_type))
    if args.remove_component == "title":
        logger.info("No title model")
        model = TBertTNoTitle(BertConfig(), args.code_bert, args.num_class)
    elif args.remove_component == "text":
        logger.info("No text model")
        model = TBertTNoText(BertConfig(), args.code_bert, args.num_class)
    elif args.remove_component == "code":
        logger.info("No code model")
        model = TBertTNoCode(BertConfig(), args.code_bert, args.num_class)
    elif tbert_type == 'triplet':
        logger.info("model with all components")
        model = TBertT(BertConfig(), args.code_bert, args.num_class)
    elif tbert_type == 'siamese':
        model = TBertSI(BertConfig(), args.code_bert, args.num_class)
    elif tbert_type == 'single':
        model = TBertSI(BertConfig(), args.code_bert, args.num_class)
    else:
        raise Exception("TBERT type not found")
    
    args.tbert_type = tbert_type
    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
        
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    if args.fp16:
        try:
            import apex
            # apex.amp.register_half_function(torch, "einsum")
            apex.amp.register_float_function(torch, 'sigmoid')
            apex.amp.register_float_function(torch, 'tanh')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    return model


def get_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", default="../../data/tensor_data", type=str,
                        help="The direcoty of the input training data files.")
    parser.add_argument("--test_data_folder", default="../../data/test", type=str,
                        help="The direcoty of the input training data files.")
    parser.add_argument("--vocab_file", default="../../data/tags/commonTags_post2vec.csv", type=str,
                        help="The tag vocab data file.")
    parser.add_argument("--train_numbers", default=10279014, type=int,
                        help="The total number of training samples")
    parser.add_argument(
        "--model_path", default="", type=str,
        help="path of checkpoint and trained model, if none will do training from scratch")
    parser.add_argument("--logging_steps", type=int,
                        default=500, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--valid_num", type=int, default=200,
                        help="number of instances used for evaluating the checkpoint performance")
    parser.add_argument("--valid_step", type=int, default=500,
                        help="obtain validation accuracy every given steps")
    parser.add_argument("--per_gpu_train_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_evalute_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--exp_name", type=str, help="name of this execution"
    )
    parser.add_argument(
        "--output_dir", default="../results", type=str,
        help="The output directory where the model checkpoints and predictions will be written.", )
    parser.add_argument("--learning_rate", default=1e-6,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--bert_type", default='triplet',
                        choices=['triplet', 'siamese','single'])
    parser.add_argument("--remove_component", default="", choices=['title', 'text','code'])
    parser.add_argument("--code_bert", default='microsoft/codebert-base',
                        choices=['microsoft/codebert-base', 'huggingface/CodeBERTa-small-v1',
                                 'codistai/codeBERT-small-v2', 'Salesforce/codet5-small','jeniya/BERTOverflow', 'roberta-base',
                                 'bert-base-uncased', 'Salesforce/codet5-base', 'razent/cotext-2-cc','facebook/bart-base',
                                 't5-base'])
    parser.add_argument(
        "--fp16", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    args = parser.parse_args()
    return args