import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GATNode import GATNode
from config.defaults import cfg as cfg


class GCNNode(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(GCNNode, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(self.in_dim, self.out_dim)
        self.dropout = nn.Dropout(p=cfg.MODEL.DROPOUT)
        self.adj = torch.zeros(size=(cfg.MODEL.MAX_EDGES, cfg.MODEL.MAX_EDGES)).to(self.device)  # update with new input
        self.sigmoid = nn.Sigmoid()

    def conv(self, graph_features):
        # fully connected layer
        ascend_dim_graph_features = self.fc(graph_features)
        # now add the features of neighboring edges: for edge a, add it with features of neighbor edges
        # add neighbors, subtract itself, add to itself by ratio
        conved_graph_features = torch.add(input=ascend_dim_graph_features, alpha=cfg.MODEL.ADD_NEIGHBOR_FEATURE_RATIO, other=torch.sub(torch.mm(self.adj, ascend_dim_graph_features), ascend_dim_graph_features))
        conv_output = self.sigmoid(conved_graph_features)
        return conv_output

    def forward(self, adj, graph_features):
        # for edge adj
        # graph_features[a] is the feature of edge a
        # adj[a][b] is the mutual node of edge a and edge b, representing the id the character node
        # now add the incoming adj(adj in a situation) to the overall adj
        self.adj = torch.add(self.adj, adj)
        n = cfg.MODEL.MAX_EDGES
        ones = torch.ones(n, n).to(self.device)
        zeros = torch.zeros(n, n).to(self.device)
        self.adj = torch.where(self.adj > 0, ones, zeros)
        self.adj = torch.where(self.adj > 0, ones, zeros)

        return self.conv(graph_features)
