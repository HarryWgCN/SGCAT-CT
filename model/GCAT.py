import torch
import torch.nn as nn
from config.defaults import cfg as cfg
from model.GCNNode import GCNNode
from model.GATNode import GATNode


class GCAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, device):
        super(GCAT, self).__init__()
        self.device = device
        self.graph_features = torch.zeros(cfg.MODEL.MAX_EDGES, cfg.FEATURE.GRAPH_FEATURE_DIM)
        self.gat_node = GATNode(in_dim, hidden_dim, cfg.FEATURE.GRAPH_FEATURE_DIM)
        self.gcn_node = GCNNode(cfg.FEATURE.GRAPH_FEATURE_DIM, cfg.FEATURE.GRAPH_FEATURE_DIM, device)
        self.new_node_feature_weight = nn.Parameter(torch.empty(size=(1,)))
        nn.init.constant_(self.new_node_feature_weight.data, cfg.MODEL.NEW_NODE_FEATURE_WEIGHT)
        self.final_fc = nn.Sequential(
            nn.Linear(cfg.FEATURE.GRAPH_FEATURE_DIM, cfg.MODEL.CLASSIFICATION_HIDDEN_DIM),
            nn.ReLU(inplace=False),
            nn.Dropout(p=cfg.MODEL.DROPOUT),
            nn.Linear(cfg.MODEL.CLASSIFICATION_HIDDEN_DIM, cfg.MODEL.CLASSIFICATION_HIDDEN_DIM),
            nn.ReLU(inplace=False),
            nn.Dropout(p=cfg.MODEL.DROPOUT),
            nn.Linear(cfg.MODEL.CLASSIFICATION_HIDDEN_DIM, out_dim)
        )

    def forward(self, bert_feature, background_feature, graph_feature_face, graph_feature_edge_idx, adj):
        graph_features = self.graph_features.detach().to(self.device)
        self.graph_features.to(self.device)
        #  new_features, expand the input graph_feature_face to full size as graph_feature
        new_features = torch.zeros(size=(cfg.MODEL.MAX_EDGES, cfg.FEATURE.GRAPH_FEATURE_DIM)).to(self.device)
        multimodal_fusion_graph_features = self.gat_node(graph_feature_face, bert_feature, background_feature)
        new_features[graph_feature_edge_idx] = multimodal_fusion_graph_features

        # remove the corresponding features in the full graph_feature
        zeros = torch.zeros(size=(1, cfg.FEATURE.GRAPH_FEATURE_DIM)).to(self.device)
        ones = torch.ones(size=(1, cfg.FEATURE.GRAPH_FEATURE_DIM)).to(self.device)
        origin_features_unrelated = graph_features.to(self.device)
        origin_features_unrelated[graph_feature_edge_idx] = zeros
        origin_features_related_mask = torch.zeros(size=(cfg.MODEL.MAX_EDGES, cfg.FEATURE.GRAPH_FEATURE_DIM)).to(self.device)
        origin_features_related_mask[graph_feature_edge_idx] = ones
        origin_features_related = torch.mul(graph_features, origin_features_related_mask)

        # feature = origin * (1 - new_weight) + new_weight * new
        conved_new_graph_features = torch.add(origin_features_unrelated, torch.add(torch.mul((1 - self.new_node_feature_weight), origin_features_related), torch.mul(self.new_node_feature_weight, new_features)))

        # conv
        conved_new_graph_features = self.gcn_node(adj, conved_new_graph_features)
        self.graph_features = conved_new_graph_features.clone().detach()
        classification_results = self.final_fc(conved_new_graph_features[graph_feature_edge_idx])

        # # extend label to full size as graph_feature
        # new_label = torch.ones(size=(cfg.MODEL.MAX_EDGES, 1), dtype=torch.long)
        # new_label = -1 * new_label
        # new_label[graph_feature_edge_idx] = label

        return classification_results

    def return_graph_feature(self, graph_feature):
        self.graph_features = graph_feature.detach()

    def clear_graph_feature(self):
        self.graph_features.detach()
        del self.graph_features
        torch.cuda.empty_cache()

    def init_graph_features(self):
        self.graph_features = torch.zeros(cfg.MODEL.MAX_EDGES, cfg.FEATURE.GRAPH_FEATURE_DIM).to(self.device)