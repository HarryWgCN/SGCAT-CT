import numpy
import torch
from pycm import *

from config.defaults import cfg


class Evaluation:
    def __init__(self):
        self.cm = ConfusionMatrix(actual_vector=[0, 1], predict_vector=[0, 1])

    def update(self, prediction, label):
        prediction = prediction.cpu().detach().numpy()
        label = label.cpu().detach().numpy().squeeze(axis=1)
        zero = numpy.array([0])
        if cfg.DATASET == 'LVU' or cfg.DATASET == 'ViSR' or cfg.DATASET == 'HLVU':
            prediction = numpy.concatenate((prediction, zero), 0)
            label = numpy.concatenate((label, zero), 0)
        cm = ConfusionMatrix(actual_vector=label, predict_vector=prediction)
        self.cm = self.cm.combine(cm)

    def show_evaluation(self):
        print(self.cm.stat(summary=True))
