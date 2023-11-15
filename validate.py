import torch
import json

from torch import nn

from config.defaults import cfg
from itertools import permutations
import os

# print(torch.version.cuda)
# print(torch.__version__)
from data.MovieGraphsDataset import MovieGraphsDataset
from data.LVUDataset import LVUDataset
from model.CulmulativeTransformer import CumulativeTransformer
from model.GCAT import GCAT
from utils.construct_prediction import construct_prediction
from utils.evaluation import Evaluation

device = ''
model = GCAT(cfg.FEATURE.FACE_FEATURE_DIM, cfg.MODEL.HIDDEN_LAYER_DIM, cfg.MODEL.GCAT_OUT_DIM, device)
model.load_state_dict(torch.load(''))
model_ct = CumulativeTransformer(cfg.MODEL.GCAT_OUT_DIM, 8, cfg.MODEL.NUM_CLASSES)
model_ct.load_state_dict(torch.load(''))
# train_dataset = MovieGraphsDataset(train=True)
train_dataset = LVUDataset(train=True)
# test_dataset = MovieGraphsDataset(train=False)
test_dataset = LVUDataset(train=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=cfg.TRAIN.BATCH_SIZE,
                                               shuffle=True,
                                               # num_workers=cfg.TRAIN.NUM_WORKERS,
                                               num_workers=1,
                                               drop_last=False)
data_len = test_dataloader.dataset.__len__()
model.train()
model_ct.train()
result_stack = torch.empty((0, cfg.MODEL.GCAT_OUT_DIM)).to(device)
label_stack = torch.empty((0, 1), dtype=torch.long).to(device)
correct = 0
amount = 0
evaluation = Evaluation()
for i in range(0, data_len):
    bert_feature, background_feature, graph_features_face, graph_feature_edge_idx, adj, labels, end = test_dataloader.dataset.__getitem__(
        i)
    results = model(bert_feature, background_feature, graph_features_face, graph_feature_edge_idx, adj).clone()
    # select available output to move on
    available_idxs = torch.nonzero(labels)[:, 0].detach()
    results = results[available_idxs]
    labels = labels[available_idxs]
    result_stack = torch.cat((result_stack, results))
    label_stack = torch.cat((label_stack, labels))
    if cfg.MODEL_NAME == 'GCAT-CT' and end:
        result_stack = model_ct(result_stack, label_stack)
        model_ct.remove_memory()
        predictions = construct_prediction(result_stack)
        evaluation.update(predictions, label_stack)
        result_stack = torch.empty(size=(0, cfg.MODEL.GCAT_OUT_DIM))
        label_stack = torch.empty(size=(0, 1), dtype=torch.long)

print('illustrating testing evaluations ----------------------------------------')
evaluation.show_evaluation()
