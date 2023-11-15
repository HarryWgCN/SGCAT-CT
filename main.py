import torch

from config.defaults import cfg
from data.MovieGraphsDataset import MovieGraphsDataset
from data.LVUDataset import LVUDataset
from data.ViSRDataset import ViSRDataset
from data.HLVUDataset import HLVUDataset
from model.GCAT import GCAT
from model.CulmulativeTransformer import CumulativeTransformer
from engine.train import train


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
# device = torch.device('cuda')
device = torch.device('cuda:3')

if cfg.DATASET == 'MovieGraphs-Short':
    train_dataset = MovieGraphsDataset(train=True)
    test_dataset = MovieGraphsDataset(train=False)

if cfg.DATASET == 'LVU':
    train_dataset = LVUDataset(train=True)
    test_dataset = LVUDataset(train=False)

if cfg.DATASET == 'ViSR':
    train_dataset = ViSRDataset(train=True)
    test_dataset = ViSRDataset(train=False)

if cfg.DATASET == 'HLVU':
    train_dataset = HLVUDataset(train=True)
    test_dataset = HLVUDataset(train=False)

if cfg.MODEL_NAME == 'GCAT':
    model = GCAT(cfg.FEATURE.FACE_FEATURE_DIM, cfg.MODEL.HIDDEN_LAYER_DIM, cfg.MODEL.GCAT_OUT_DIM, device).to(device)
    model_ct = None

if cfg.MODEL_NAME == 'GCAT-CT':
    model = GCAT(cfg.FEATURE.FACE_FEATURE_DIM, cfg.MODEL.HIDDEN_LAYER_DIM, cfg.MODEL.GCAT_OUT_DIM, device).to(device)
    model_ct = CumulativeTransformer(cfg.MODEL.GCAT_OUT_DIM, 8, cfg.MODEL.NUM_CLASSES, device).to(device)

# for name, parameter in model.named_parameters():
#      parameter.requires_grad = False
# torch.autograd.set_detect_anomaly(True)
# for param in model.state_dict():
#     print(param, model.state_dict()[param])
# model.load_state_dict(torch.load(''))
train(device, train_dataset, test_dataset, model, model_ct)
