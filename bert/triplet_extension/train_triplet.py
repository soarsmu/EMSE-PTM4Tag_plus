import sys
sys.path.append("../")
sys.path.append("../..")
sys.path.append("/usr/src/bert")
import numpy as np
from datetime import datetime
import gc
import logging
import os
import pandas as pd
import torch
from data_structure.question import Question, QuestionDataset,TensorQuestionDataset
from util.util import get_files_paths_from_directory, save_check_point, load_check_point,seed_everything,write_tensor_board
from util.eval_util import evaluate_batch
from util.data_util import get_dataloader, get_distribued_dataloader, load_tenor_data_to_dataset,load_data_to_dataset
from model.loss import loss_fn
from train import get_optimizer, get_optimizer_scheduler,get_train_args, init_train_env, get_exe_name
from apex.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import BertConfig, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
# from accelerate import Accelerator

logger = logging.getLogger(__name__)

def log_train_info(args):
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)


def train(args, model):
    logger.info("GET ARGS")
    logger.info(args)
    files = get_files_paths_from_directory(args.data_folder)
    
    if not args.exp_name:
        exp_name = get_exe_name(args)
    else:
        exp_name = args.exp_name
    # total training examples 10279014
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_numbers = args.train_numbers
    epoch_batch_num = train_numbers / args.train_batch_size
    t_total = epoch_batch_num // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer = get_optimizer(args,model)
    
    # make output directory
    args.output_dir = os.path.join(args.output_dir, exp_name)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir="../runs/{}".format(exp_name))
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    logger.info("n_gpu: {}".format(args.n_gpu))
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = DDP(model, delay_allreduce=True)


    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        logger.info("model loaded")
    log_train_info(args)
    args.global_step = 0
        
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    for epoch in range(args.num_train_epochs):
        logger.info(
            '############# Epoch {}: Training Start   #############'.format(epoch))
        for file_cnt in range(len(files)):
            # Load dataset and dataloader
            train_dataset = load_tenor_data_to_dataset(args.mlb, files[file_cnt])     
            if args.local_rank == -1:
                train_data_loader = get_dataloader(
                    train_dataset, args.train_batch_size)
            else: 
                train_data_loader = get_distribued_dataloader(
                    train_dataset, args.train_batch_size)    
            tr_loss = 0
            model.train()
            model.zero_grad()
            for step, data in enumerate(train_data_loader):
                title_ids = data['titile_ids'].to(
                    args.device, dtype=torch.long)
                title_mask = data['title_mask'].to(
                    args.device, dtype=torch.long)
                text_ids = data['text_ids'].to(args.device, dtype=torch.long)
                text_mask = data['text_mask'].to(args.device, dtype=torch.long)
                code_ids = data['code_ids'].to(args.device, dtype=torch.long)
                code_mask = data['code_mask'].to(args.device, dtype=torch.long)
                targets = data['labels'].to(args.device, dtype=torch.float)
                outputs = model(title_ids=title_ids,
                                title_attention_mask=title_mask,
                                text_ids=text_ids,
                                text_attention_mask=text_mask,
                                code_ids=code_ids,
                                code_attention_mask=code_mask)

                loss = loss_fn(outputs, targets)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    try:
                        from apex import amp
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    except ImportError:
                        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                else:
                    loss.backward()
                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    args.global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and args.global_step % args.logging_steps == 0:
                        tb_data = {
                            'lr': scheduler.get_last_lr()[0],
                            'loss': tr_loss / args.logging_steps
                        }
                        write_tensor_board(tb_writer, tb_data, args.global_step)
                        logger.info("tb_data {}".format(tb_data))
                        logger.info(
                            'Epoch: {}, Batch: {}， Loss:  {}'.format(epoch, step, tr_loss / args.logging_steps))
                        tr_loss = 0.0
            logger.info(
                '############# FILE {}: Training End     #############'.format(file_cnt))
            
            ### Save Model
            if args.local_rank in [-1, 0]:
                model_output = os.path.join(
                    args.output_dir, "epoch-{}-file-{}".format(epoch,file_cnt))
                save_check_point(model, model_output, args,
                                optimizer, scheduler)
    if args.local_rank in [-1, 0]:
        tb_writer.close()


def main():
    args = get_train_args()
    model = init_train_env(args, tbert_type=args.bert_type)
    train(args, model)

if __name__ == "__main__":
    main()
