import math
import copy

import torch
import torch.nn as nn

from config.defaults import cfg


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    print('A1 ', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    print('A2 ', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # 3 for q k v transform, 1 for result transform
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.key_memory = torch.empty(size=(1, 0, self.h, self.d_k))
        self.value_memory = torch.empty(size=(1, 0, self.h, self.d_k))

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        key, value = torch.cat((key, self.key_memory), dim=1), torch.cat((value, self.value_memory), dim=1)

        check memory
        if self.key_memory.shape[1] > cfg.MODEL.TRANSFORMER_MEMORY_SIZE:
            self.key_memory = self.key_memory[0][1:]
            self.value_memory = self.value_memory[0][1:]

        # # distinguish memory and non memory attention by mask, memorize in encoder
        ####

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
                .contiguous()
                .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        torch.cuda.empty_cache()
        return self.linears[-1](x)

    def remove_memory(self):
        print('removed')
        del self.key_memory
        self.key_memory = torch.empty(size=(1, 0, self.h, self.d_k))
        del self.value_memory
        self.value_memory = torch.empty(size=(1, 0, self.h, self.d_k))
