import torch.nn as nn


def construct_prediction(results):
    softmax_operator = nn.Softmax(dim=1)
    results = softmax_operator(results)
    max_idx = results.max(1, keepdim=False)[1]  # (N, 1), represent the idx of the maximum possibility
    return max_idx
