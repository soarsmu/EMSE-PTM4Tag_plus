
import shutil
import sys
import torch
from torch import optim
import argparse
import datetime
import logging
import multiprocessing
import os
import sys
import random
import numpy as np
logger = logging.getLogger(__name__)

MODEL_FNAME = "t_bert.pt"
OPTIMIZER_FNAME = "optimizer.pt"
SCHED_FNAME = "scheduler.pt"
ARG_FNAME = "training_args.bin"

def avg(data):
    import numpy as np
    a = np.array(data)
    res = np.average(a, axis=0)
    return res


def write_tensor_board(tb_writer, data, step):
    for att_name in data.keys():
        att_value = data[att_name]
        tb_writer.add_scalar(att_name, att_value, step)


def get_files_paths_from_directory(input_dir):
    file_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))
    file_paths.sort()
    return file_paths


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def save_check_point(model, ckpt_dir, args, optimizer, scheduler):
    logger.info("Saving checkpoint to %s", ckpt_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, MODEL_FNAME))
    torch.save(args, os.path.join(ckpt_dir, ARG_FNAME))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, OPTIMIZER_FNAME))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, SCHED_FNAME))


def load_check_point(model, ckpt_dir, optimizer, scheduler):
    logger.info(
        "Loading checkpoint from {}, remove optimizer and scheduler if you do not want to load them".format(ckpt_dir))
    optmz_path = os.path.join(ckpt_dir, OPTIMIZER_FNAME)
    sched_path = os.path.join(ckpt_dir, SCHED_FNAME)
    model_path = os.path.join(ckpt_dir, MODEL_FNAME)
    arg_path = os.path.join(ckpt_dir, ARG_FNAME)

    model.load_state_dict(torch.load(model_path))
    if os.path.isfile(optmz_path):
        logger.info("Loading optimizer...")
        optimizer.load_state_dict(torch.load(optmz_path))
    if os.path.isfile(sched_path):
        logger.info("Loading scheduler...")
        scheduler.load_state_dict(torch.load(sched_path))

    args = None
    if os.path.isfile(arg_path):
        args = torch.load(os.path.join(ckpt_dir, ARG_FNAME))
    return {'model': model, "optimizer": optimizer, "scheduler": scheduler, "args": args}
