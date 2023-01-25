#!/usr/bin/env python
# coding: utf-8
import csv

import argparse, time, math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
#from dgl.data import register_data_args
#from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from copy import deepcopy
import warnings
import os
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import json

warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.model_selection import KFold


def gcn_msg(edge):
    msg = edge.src['h'] * edge.src['norm']
    return {'m': msg}

def gcn_reduce(node):
    accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
    return {'h': accum}

class NodeApplyModule(nn.Module):
    '''
    input:out_feats, activation, bias
    '''
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        h = nodes.data['h']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.g = g
        #print(g)
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        #print(self.weight)
        #print(self.weight.shape)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.node_update = NodeApplyModule(out_feats, activation, bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        self.g.ndata['h'] = torch.mm(h, self.weight)
        self.g.update_all(gcn_msg, gcn_reduce, self.node_update)
        h = self.g.ndata.pop('h')
        return h


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(g, in_feats, n_hidden, activation, dropout))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(g, n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNLayer(g, n_hidden, n_classes, None, dropout))

    def forward(self, features):
        h = features
        layer1 = self.layers[0]
        layer2 = self.layers[1]
        layer3 = self.layers[2]
        #input out
        h = layer1(h)
        ten = layer2(h)
        h = layer3(ten)

        return h,ten


def metrics(y_true, y_pred, y_prob,ten):
    y_true, y_pred, y_prob = y_true.numpy(), y_pred.numpy(), y_prob.numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    pos_acc = tp / sum(y_true)
    # neg_acc = tn / (len(y_pred) - sum(y_pred))  # [y_true=0 & y_pred=0] / y_pred=0
    neg_acc = tn / (len(y_pred) - sum(y_true))  # [y_true=0 & y_pred=0] / y_pred=0
    accuracy = (tp + tn) / (tn + fp + fn + tp)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)

    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    specificity = tn / (tn + fp)

    # print('======================Counter(y_pred)：\n', Counter(y_pred))

    return (y_true, y_pred, y_prob), [aupr, roc_auc, f1, accuracy, recall, specificity, precision], ten


def evaluate(model, features, labels, mask, flag):
    model.eval()
    with torch.no_grad():
        logits, ten = model(features)
        #print(logits)
        #print(mask)
        logits = logits[mask]
        #print(logits)
        #print(logits)
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        #print(indices)
        probas = F.softmax(logits)
        #print(probas)

        return metrics(labels.cpu().detach(), indices.cpu().detach(), probas.cpu().detach()[:, 1], ten)


def main(g, features, labels, train_idx, n_hidden, n_layers, dropout, lr, poch):
    device = torch.device('cpu')
    num_nodes = g.number_of_nodes()
    train_mask = np.zeros(num_nodes, dtype='int64')
    train_mask[train_idx] = 1   #[0 1 1 ... 0 1 1]
    #print(train_mask.shape)
    test_mask = 1 - train_mask
    # print(Counter(train_mask), Counter(test_mask))
    # #Counter({1: 1547, 0: 387}) Counter({0: 1547, 1: 387})

    train_mask = torch.BoolTensor(train_mask)
    test_mask = torch.BoolTensor(test_mask)

    g.ndata['feat'] = features
    #print(g.ndata['feat'])
    g.ndata['label'] = labels
    #print(g.ndata['label'])
    g.ndata['train_mask'] = train_mask
    #print(g.ndata['train_mask'])
    g.ndata['test_mask'] = test_mask
    #print(g.ndata['test_mask'])

    g = g.to(device)

    in_feats = features.shape[1]
    #print(in_feats)
    n_classes = 2
    n_edges = g.number_of_edges()

    features, labels = features.to(device), labels.to(device)

    degs = g.in_degrees().float()   #tensor([1., 3., 5.,  ..., 2., 6., 5.])
    #print(degs.shape[0])
    norm = torch.pow(degs, -0.5)


    #(1,1934)
    norm[torch.isinf(norm)] = 0

    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    # n_hidden = 512  # 64 256 32
    # n_layers = 3  # (2) 4 5
    # dropout = 0.1  # 0.1-0.7
    model = GCN(g,
                in_feats,
                n_hidden,
                n_classes,
                n_layers,
                F.relu,
                dropout)

    # forward
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr,
                                 weight_decay=5e-4)
    best_pre = np.inf
    _,best_tensor = model(features)
    for epoch in range(poch):
        model.train()
        # forward
        logits, ten = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        #print(logits[train_mask])
        #print(labels[train_mask])
        # loss.requires_grad=True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #ys_train, metrics_train = evaluate(model, features, labels, train_mask, 1)
        ys_test, metrics_test, ten = evaluate(model, features, labels, train_mask, 0)

        if metrics_test[-1] >= best_pre:
            best_tensor = ten
            best_pre = metrics_test[-1]

    tensor = best_tensor
    print(best_tensor)

    return tensor


def run(name_csv, n_neigh, lr, poch):
    if name_csv == 'data_CKSAAP.csv':
        node_feature_label = pd.read_csv('./data/CKSAAP_x_all.csv', index_col=0)
    elif name_csv == 'data_TPC.csv':
        node_feature_label = pd.read_csv('./data/TPC_x_all.csv', index_col=0)
    elif name_csv == 'data_CTDC.csv':
        node_feature_label = pd.read_csv('./data/CTDC_x_all.csv', index_col=0)
    elif name_csv == 'data_CTDT.csv':
        node_feature_label = pd.read_csv('./data/CTDT_x_all.csv', index_col=0)

    train_test_id_idx = np.load("./graph/Tp__imbalanced__testlabel0_knn_edge_train_test_index_all.npz",allow_pickle=True)

    #print(train_test_id_idx)    ['train_index_all', 'test_index_all', 'train_id_all', 'test_id_all']
    train_index_all = train_test_id_idx['train_index_all']
    test_index_all = train_test_id_idx['test_index_all']

    num_nodes = node_feature_label.shape[0]
    #print(num_nodes)

    features = torch.FloatTensor(np.array(node_feature_label.iloc[:, 3:]))
    labels = torch.LongTensor(np.array(node_feature_label['2']))


    n_hidden = 256
    n_layers = 2
    dropout = 0.7
    for train_idx, test_idx in zip(train_index_all, test_index_all):
        pwd = "./graph/"
        knn_graph_file = name_csv + "task_Tp__imbalanced__testlabel0_knn" + str(n_neigh) + "neighbors_edge__fold.npz"

        knn_neighbors_graph = sp.load_npz(pwd + knn_graph_file)
        #print(knn_neighbors_graph)
        # knn_neighbors_graph.nonzero():
        # (array([0, 0, 0, ..., 10859, 10859, 10859]), array([0, 237, 33, ..., 10725, 6026, 10475]))
        edge_src = knn_neighbors_graph.nonzero()[0]
        edge_dst = knn_neighbors_graph.nonzero()[1]

        g = dgl.DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(edge_src, edge_dst)
        #print(g)
        g = dgl.add_self_loop(g)
        #print(g)

        tensor = main(g, features, labels, train_idx, n_hidden, n_layers, dropout, lr, poch)

    return tensor
