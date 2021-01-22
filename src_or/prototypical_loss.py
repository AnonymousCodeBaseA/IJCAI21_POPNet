# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix as conm

class PrototypicalLoss(Module):

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def prototypical_loss(input, target, n_support,mode = 'train'):


    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    # print(target_cpu.size())
    # 
    def takeFirst(elem):
        return elem[0]

    def supp_idxs(c):
        # FIXME when torch will support where as np

'''

Prototype Optimization Code will be publicly available until acceptance.

'''
#

    classes = torch.unique(target_cpu).sort()[0]

    n_classes = len(classes)

    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support


    support_idxs = list(map(supp_idxs, classes))

    

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    prototypes_cuda = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])

    loss_avg = None
    loss_all = []
    if mode == 'train':
        criterion = torch.nn.MarginRankingLoss()

'''

Triplet Optimization Code will be publicly available until acceptance.

'''









    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)



    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)#10 5 10


    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)

    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(2)

    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    accuracy,recall,precision = indicator_cal(y_hat,target_inds)



    if mode == 'train':
        loss_val += loss_triplet


    return loss_val,  acc_val ,recall,precision, y_hat,target_inds



def indicator_cal(pred,label):
    y_pred = pred.flatten().tolist()
    y_true = label.flatten().tolist()

    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
    # print(confusion_matrix)
    confusion_matrix = np.array(confusion_matrix)#.sum(0).reshape((2,2))

    accuracy = []
    precision = []
    recall = []


    for i in range(confusion_matrix.shape[0]):
        per_cm = confusion_matrix[i].reshape((2,2))


        TN,FN,FP,TP = per_cm[0,0],per_cm[1,0],per_cm[0,1],per_cm[1,1]


        accuracy.append((TP+TN)*1.0/((TP+TN+FP+FN)*1.0))

        if TP!=0 or FP!=0:
            precision.append(TP*1.0/((TP+FP)*1.0))
        if TP!=0 or FN!=0:
            recall.append(TP*1.0/((TP+FN)*1.0))

    return float(np.array(accuracy).mean()),float(np.array(precision).mean()),float(np.array(recall).mean())