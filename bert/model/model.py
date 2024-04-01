import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, T5EncoderModel, PLBartModel
from transformers.models.bart.modeling_bart import BartEncoder
from transformers.models.plbart.modeling_plbart import PLBartEncoder
from model.loss import loss_fn


class AvgPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class MaxPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveMaxPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class ClassifyHeader(nn.Module):
    """
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config, num_class):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.title_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)
        self.code_pooler = AvgPooler(config)

        # self.dense = nn.Linear(config.hidden_size * 5, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, num_class)

    def forward(self, title_hidden, text_hidden, code_hidden):
        pool_title_hidden = self.title_pooler(title_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        pool_code_hidden = self.code_pooler(code_hidden)

        # concatenates the given sequence of tensors in the given dimension
        concated_hidden = torch.cat((pool_title_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, pool_code_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class ClassifyHeaderNoTitle(nn.Module):
    """
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config, num_class):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.text_pooler = AvgPooler(config)
        self.code_pooler = AvgPooler(config)

        # self.dense = nn.Linear(config.hidden_size * 5, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, num_class)

    def forward(self, text_hidden, code_hidden):
        pool_text_hidden = self.text_pooler(text_hidden)
        pool_code_hidden = self.code_pooler(code_hidden)

        # concatenates the given sequence of tensors in the given dimension
        concated_hidden = torch.cat((pool_text_hidden, pool_code_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class ClassifyHeaderNoText(nn.Module):
    """
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config, num_class):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.title_pooler = AvgPooler(config)
        self.code_pooler = AvgPooler(config)

        # self.dense = nn.Linear(config.hidden_size * 5, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, num_class)

    def forward(self, title_hidden, code_hidden):
        pool_title_hidden = self.title_pooler(title_hidden)
        pool_code_hidden = self.code_pooler(code_hidden)

        # concatenates the given sequence of tensors in the given dimension
        concated_hidden = torch.cat((pool_title_hidden, pool_code_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
class ClassifyHeaderNoCode(nn.Module):
    """
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config, num_class):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.title_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, num_class)

    def forward(self, title_hidden, text_hidden):
        pool_title_hidden = self.title_pooler(title_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        concated_hidden = torch.cat((pool_title_hidden, pool_text_hidden), 1)
        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

class TBertT(PreTrainedModel):
    def __init__(self, config, code_bert, num_class):
        super().__init__(config)
        if "bart" in code_bert:	
            self.tbert = BartEncoder.from_pretrained(code_bert)	
            self.nbert = BartEncoder.from_pretrained(code_bert)	
            self.cbert = BartEncoder.from_pretrained(code_bert)	
            print("BART EncoderModel")	
        	
        if "t5" in code_bert:	
            self.tbert = T5EncoderModel.from_pretrained(code_bert)	
            self.nbert = T5EncoderModel.from_pretrained(code_bert)	
            self.cbert = T5EncoderModel.from_pretrained(code_bert)	
            print("T5EncoderModel")	
        	
        elif "cotext" in code_bert:	
            self.tbert = T5EncoderModel.from_pretrained(code_bert)	
            self.nbert = T5EncoderModel.from_pretrained(code_bert)	
            self.cbert = T5EncoderModel.from_pretrained(code_bert)	
            print("Cotext Model")
        elif "plbart" in code_bert:
            self.tbert = PLBartEncoder.from_pretrained(code_bert)
            self.nbert = PLBartEncoder.from_pretrained(code_bert)
            self.cbert = PLBartEncoder.from_pretrained(code_bert)
        else:
            self.tbert = AutoModel.from_pretrained(code_bert)
            self.nbert = AutoModel.from_pretrained(code_bert)
            self.cbert = AutoModel.from_pretrained(code_bert)

        self.cls = ClassifyHeader(config, num_class=num_class)

    def forward(
            self,
            title_ids=None,
            title_attention_mask=None,
            text_ids=None,
            text_attention_mask=None,
            code_ids=None,
            code_attention_mask=None,
    ):
        t_hidden = self.tbert(
            title_ids, attention_mask=title_attention_mask)[0]
        n_hidden = self.nbert(text_ids, attention_mask=text_attention_mask)[0]
        c_hidden = self.cbert(code_ids, attention_mask=code_attention_mask)[0]

        logits = self.cls(title_hidden=t_hidden,
                          text_hidden=n_hidden, code_hidden=c_hidden)
        return logits


class TBertTNoTitle(PreTrainedModel):
    def __init__(self, config, code_bert, num_class):
        super().__init__(config)
        if "t5" in code_bert:	
            self.nbert = T5EncoderModel.from_pretrained(code_bert)	
            self.cbert = T5EncoderModel.from_pretrained(code_bert)	
        else: 
            self.nbert = AutoModel.from_pretrained(code_bert)
            self.cbert = AutoModel.from_pretrained(code_bert)

        self.cls = ClassifyHeaderNoTitle(config, num_class=num_class)

    def forward(
            self,
            text_ids=None,
            text_attention_mask=None,
            code_ids=None,
            code_attention_mask=None,
    ):
        n_hidden = self.nbert(text_ids, attention_mask=text_attention_mask)[0]
        c_hidden = self.cbert(code_ids, attention_mask=code_attention_mask)[0]

        logits = self.cls(text_hidden=n_hidden, code_hidden=c_hidden)
        return logits


class TBertTNoText(PreTrainedModel):
    def __init__(self, config, code_bert, num_class):
        super().__init__(config)
        if "t5" in code_bert:	
            self.nbert = T5EncoderModel.from_pretrained(code_bert)	
            self.cbert = T5EncoderModel.from_pretrained(code_bert)	
        else: 
            self.tbert = AutoModel.from_pretrained(code_bert)
            self.cbert = AutoModel.from_pretrained(code_bert)

        self.cls = ClassifyHeaderNoText(config, num_class=num_class)

    def forward(
            self,
            title_ids=None,
            title_attention_mask=None,
            code_ids=None,
            code_attention_mask=None,
    ):
        t_hidden = self.tbert(
            title_ids, attention_mask=title_attention_mask)[0]
        c_hidden = self.cbert(code_ids, attention_mask=code_attention_mask)[0]

        logits = self.cls(title_hidden=t_hidden,code_hidden=c_hidden)
        return logits

class TBertTNoCode(PreTrainedModel):
    def __init__(self, config, code_bert, num_class):
        super().__init__(config)
        if "t5" in code_bert:	
            self.nbert = T5EncoderModel.from_pretrained(code_bert)	
            self.cbert = T5EncoderModel.from_pretrained(code_bert)	
        else: 
            self.tbert = AutoModel.from_pretrained(code_bert)
            self.nbert = AutoModel.from_pretrained(code_bert)

        self.cls = ClassifyHeaderNoCode(config, num_class=num_class)

    def forward(
            self,
            title_ids=None,
            title_attention_mask=None,
            text_ids=None,
            text_attention_mask=None,
    ):
        t_hidden = self.tbert(
            title_ids, attention_mask=title_attention_mask)[0]
        n_hidden = self.nbert(text_ids, attention_mask=text_attention_mask)[0]

        logits = self.cls(title_hidden=t_hidden, text_hidden=n_hidden)
        return logits


class TBertPLBart(PreTrainedModel):
    def __init__(self, config, code_bert, num_class):
        super().__init__(config)
        self.tbert = PLBartModel.from_pretrained(code_bert)
        self.nbert = PLBartModel.from_pretrained(code_bert)
        self.cbert = PLBartModel.from_pretrained(code_bert)

        self.cls = ClassifyHeader(config, num_class=num_class)

    def forward(
            self,
            title_ids=None,
            title_attention_mask=None,
            text_ids=None,
            text_attention_mask=None,
            code_ids=None,
            code_attention_mask=None,
    ):
        t_hidden = self.tbert(
            input_ids=title_ids, attention_mask=title_attention_mask)[0]
        n_hidden = self.nbert(input_ids=text_ids, attention_mask=text_attention_mask)[0]
        c_hidden = self.cbert(input_ids=code_ids, attention_mask=code_attention_mask)[0]

        logits = self.cls(title_hidden=t_hidden,
                          text_hidden=n_hidden, code_hidden=c_hidden)
        return logits
