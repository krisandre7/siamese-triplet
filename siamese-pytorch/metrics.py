import numpy as np
import torch

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class TripletAccuracy(Metric):
    def __init__(self, margin = 2):
        self.correct = 0
        self.total = 0
        self.margin = margin
        
    def __call__(self, outputs, target, loss_outputs):
        anchor = outputs[0]
        positive = outputs[1]
        negative = outputs[2]
        distance_positive = torch.pairwise_distance(anchor, positive).cpu().data
        distance_negative = torch.pairwise_distance(anchor, negative).cpu().data
        # print((torch.abs(anchor - positive) > 0).count_nonzero() / anchor.shape[0])
        self.correct += (distance_positive < distance_negative).count_nonzero()
        self.total += len(distance_positive)
        return self.value()

    def value(self):
        return 100 * float(self.correct) / self.total
    
    def reset(self):
        self.correct = 0
        self.total = 0

    
    def name(self):
        return 'Accuracy'

class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss.item())
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'
