from torch.utils.data import Dataset
import torch
import numpy as np
import ast
from nltk.tokenize.treebank import TreebankWordDetokenizer


class Question:
    # use __slots to decrease the memory usage
    __slots__ = ['qid', 'title', 'desc_text',
                 'desc_code', 'creation_date', 'tags', 'combine']

    def __init__(self, qid=None, title=None, desc_text=None, desc_code=None, creation_date=None, tags=None):
        self.qid = qid
        self.title = title
        self.desc_text = desc_text
        self.desc_code = desc_code
        self.creation_date = creation_date
        self.tags = tags
        self.combine = None

    def get_comp_by_name(self, comp_name):
        if comp_name == "qid":
            return self.qid
        if comp_name == "title":
            return TreebankWordDetokenizer().detokenize(self.title)
        if comp_name == "desc_text":
            return TreebankWordDetokenizer().detokenize(self.desc_text)
        if comp_name == "desc_code":
            return TreebankWordDetokenizer().detokenize(self.desc_code)
        if comp_name == "creation_date":
            return self.creation_date
        if comp_name == "tags":
            return self.tags
        if comp_name == "combine":
            return self.combine


class NewQuestion:
    # use __slots to decrease the memory usage
    __slots__ = ['qid', 'title', 'text',
                 'code', 'creation_date', 'tags', ]

    def __init__(self, qid=None, title=None, text=None,
                 code=None, creation_date=None, tags=None):
        self.qid = qid
        self.title = title
        self.text = text
        self.code = code
        self.creation_date = creation_date
        self.tags = tags

    def get_qid(self):
        return self.qid

    def get_creation_date(self):
        return self.creation_date

    def get_title(self):
        return self.title

    def get_text(self):
        return self.text

    def get_code(self):
        return self.code

    def get_tag(self):
        return self.tags

    def get_comp_by_name(self, comp_name):
        if comp_name == "qid":
            return self.qid
        if comp_name == "creation_date":
            return self.creation_date

class QuestionDataset(Dataset):

    def __init__(self, questions, mlb, tokenizer):
        self.questions = questions
        self.tokenizer = tokenizer
        self.mlb = mlb
        self.length = 0

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        title = question.get_comp_by_name("title")
        text = question.get_comp_by_name("desc_text")
        code = question.get_comp_by_name("desc_code")
        labels = set(question.get_comp_by_name("tags"))
        ret = self.mlb.transform([labels])

        title_feat = self._gen_feature(title)
        text_feat = self._gen_feature(text)
        code_feat = self._gen_feature(code)

        return {
            'titile_ids': title_feat['input_ids'],
            'title_mask': title_feat['attention_mask'],
            'text_ids': text_feat['input_ids'],
            'text_mask': text_feat['attention_mask'],
            'code_ids': code_feat['input_ids'],
            'code_mask': code_feat['attention_mask'],
            'labels': torch.from_numpy(ret[0]).type(torch.FloatTensor)
        }

    def _gen_feature(self, tokens):

        feature = self.tokenizer(tokens, max_length=512,
                                 padding='max_length', return_attention_mask=True,
                                 return_token_type_ids=False, truncation=True,
                                 return_tensors='pt')
        res = {
            "input_ids": feature["input_ids"].flatten(),
            "attention_mask": feature["attention_mask"].flatten()}
        return res


class TestQuestionDataset(Dataset):
    
    def __init__(self, questions, mlb, tokenizer):
        self.questions = questions
        self.tokenizer = tokenizer
        self.mlb = mlb
        self.length = 0

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        title_feat = question.get_title()
        text_feat = question.get_text()
        code_feat = question.get_code()
        labels = set(question.get_tag())
        ret = self.mlb.transform([labels])

        # title_feat = self._gen_feature(title)
        # text_feat = self._gen_feature(text)
        # code_feat = self._gen_feature(code)

        return {
            'title': title,
            'text': text,
            'code': code,
            'titile_ids': title_feat['input_ids'],
            'title_mask': title_feat['attention_mask'],
            'text_ids': text_feat['input_ids'],
            'text_mask': text_feat['attention_mask'],
            'code_ids': code_feat['input_ids'],
            'code_mask': code_feat['attention_mask'],
            'labels': torch.from_numpy(ret[0]).type(torch.FloatTensor)
        }

    def _gen_feature(self, tokens):

        feature = self.tokenizer(tokens, max_length=512,
                                 padding='max_length', return_attention_mask=True,
                                 return_token_type_ids=False, truncation=True,
                                 return_tensors='pt')
        res = {
            "input_ids": feature["input_ids"].flatten(),
            "attention_mask": feature["attention_mask"].flatten()}
        return res



class TensorQuestionDataset(Dataset):

    def __init__(self, questions, mlb):
        self.questions = questions
        self.mlb = mlb

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        title_feat = question.get_title()
        text_feat = question.get_text()
        code_feat = question.get_code()
        labels = set(question.get_tag())
        # labels = set(ast.literal_eval(question.get_tag()))
        # print("tags:", question.get_tag())
        ret = self.mlb.transform([labels])

        return {
            'titile_ids': title_feat['input_ids'],
            'title_mask': title_feat['attention_mask'],
            'text_ids': text_feat['input_ids'],
            'text_mask': text_feat['attention_mask'],
            'code_ids': code_feat['input_ids'],
            'code_mask': code_feat['attention_mask'],
            'labels': torch.from_numpy(ret[0]).type(torch.FloatTensor)
        }