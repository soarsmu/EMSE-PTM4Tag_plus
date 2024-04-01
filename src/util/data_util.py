import sys
sys.path.append("../")
sys.path.append("/usr/src/bert")
import pandas as pd
from sklearn import preprocessing
from data_structure.question import Question, QuestionDataset,TensorQuestionDataset, TestQuestionDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

def get_fixed_tag_encoder(vocab_file):
    tab_vocab_path = vocab_file
    tag_vocab = pd.read_csv(tab_vocab_path,keep_default_na=False)
    tag_list = tag_vocab["tag"].astype(str).tolist()
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit([tag_list])
    return mlb, len(mlb.classes_)

def get_tag_encoder(vocab_file):
    tab_vocab_path = vocab_file
    tag_vocab = pd.read_csv(tab_vocab_path)
    tag_list = tag_vocab["tag"].astype(str).tolist()
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit([tag_list])
    return mlb, len(mlb.classes_)

def load_tenor_data_to_dataset(mlb, file):
    train = pd.read_pickle(file)
    training_set = TensorQuestionDataset(train, mlb)
    return training_set

def load_data_to_dataset_for_test(mlb, file, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, local_files_only=True)
    train = pd.read_pickle(file)
    training_set = TestQuestionDataset(train, mlb, tokenizer)
    return training_set

def load_data_to_dataset(mlb, file):
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/codebert-base", local_files_only=True)
    train = pd.read_pickle(file)
    training_set = QuestionDataset(train, mlb, tokenizer)
    return training_set

def get_dataloader(dataset, batch_size):
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             )
    return data_loader


def get_distribued_dataloader(dataset, batch_size):
    sampler = DistributedSampler(dataset)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             )
    return data_loader
