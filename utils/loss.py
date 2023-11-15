import torch
import torch.nn as nn


def compute_loss(results, labels):
    loss_func = nn.CrossEntropyLoss()
    return loss_func(results, labels)
