import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SSSUnit(nn.Module):
    """
    Shift Scale Shift Unit
    """

    def __init__(self, input_dim, output_dim):
        super(SSSUnit, self).__init__()
        self.lin = nn.Linear(output_dim, input_dim * 3)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, y):
        pre_shift, log_scale, post_shift = self.lin(y).chunk(3, dim=-1)
        return (x + pre_shift) * torch.exp(log_scale) + post_shift


class Bert_Wrapper(nn.Module):
    """Wrapper for Auto Model"""
    
    def __init__(self, bert_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(bert_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
    

    def forward(self, text):
        device = next(self.model.parameters()).device

        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        tokenized = { # move to device (CUDA, CPU, whatever)
            k: v.to(device)
            for k,v in tokenized.items()
        }

        
        outputs = self.model(**tokenized)

        return self._mean_pooling(model_output=outputs, attention_mask=tokenized['attention_mask'])


    # https://datascience.stackexchange.com/questions/107212/get-sentence-embeddings-of-transformer-based-models
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask



