import torch
import torch.nn as nn
from config.defaults import cfg as cfg
from model.Attention import *
from model.Transformer import *
from config.defaults import cfg


class CumulativeTransformer(nn.Module):
    def __init__(self, in_dim, head, out_dim, device):
        super(CumulativeTransformer, self).__init__()
        N = 6
        c = copy.deepcopy
        ff = PositionwiseFeedForward(in_dim, cfg.MODEL.TRANSFORMER_FEED_FORWARD_HIDDEN, cfg.MODEL.DROPOUT)
        attn = MultiHeadedAttention(head, in_dim)
        self.device = device
        self.attn_list = [c(attn), c(attn), c(attn)]
        self.encoder_decoder = EncoderDecoder(
            Encoder(EncoderLayer(in_dim, self.attn_list[0], c(ff), cfg.MODEL.DROPOUT), N),
            Decoder(DecoderLayer(in_dim, self.attn_list[1], self.attn_list[2], c(ff), cfg.MODEL.DROPOUT), N),
            Generator(in_dim, out_dim),
        )
        # self.output_embedder = Embeddings(in_dim, cfg.MODEL.NUM_CLASSES)
        self.output_embedder = Embeddings(in_dim, cfg.MODEL.GCAT_OUT_DIM)
        self.dropout = nn.Dropout(p=0.65)

    def forward(self, graph_feature, previous_output, edge_num=None):
        previous_output = previous_output.float()
        previous_output = self.dropout(previous_output)
        previous_output = previous_output.long()
        previous_output = self.output_embedder(previous_output)
        subsequent_mask = torch.triu(torch.ones(graph_feature.shape[0], graph_feature.shape[0]), diagonal=1).type(torch.uint8).to(self.device)
        ones = torch.ones(size=(1, graph_feature.shape[1])).expand(edge_num, graph_feature.shape[1]).type(torch.uint8)
        padding = torch.zeros(size=(1, graph_feature.shape[1])).expand(graph_feature.shape[0] - edge_num, graph_feature.shape[1]).type(torch.uint8)
        padding_mask = torch.cat((ones, padding), dim=0)
        padding_mask = torch.mm(padding_mask, padding_mask.T)
        padding_subsequent_mask = torch.mm(padding_mask, subsequent_mask)
        graph_feature = graph_feature.unsqueeze(0).to(self.device)
        subsequent_mask = subsequent_mask.unsqueeze(0).to(self.device)
        previous_output = previous_output.transpose(0, 1).to(self.device)
        pseudo_label = torch.ones_like(label)
        return self.encoder_decoder(graph_feature, previous_output, subsequent_mask, subsequent_mask).transpose(0, 1).squeeze(0).squeeze(1)

    def remove_memory(self):
        for model in self.attn_list:
            model.remove_memory()

