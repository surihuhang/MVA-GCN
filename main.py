import grid
import pandas as pd
import csv
import torch
import random
#from MMGCN import MMGCN
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from collections import Counter

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run MMGCN.")

    parser.add_argument("--epoch",
                        type=int,
                        default=1000,
                        help="Number of training epochs. Default is 651.")
    parser.add_argument("--poch",
                        type=int,
                        default=100,
                        help="Number of training epochs. Default is 651.")
    parser.add_argument("--gcn-layers",
                        type=int,
                        default=2,
                        help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--out-channels",
                        type=int,
                        default=2,
                        help="out-channels of cnn. Default is 128.")

    parser.add_argument("--protein-number",
                        type=int,
                        default=(1938+100+212),
                        help="miRNA number. Default is 853.")

    parser.add_argument("--fm",
                        type=int,
                        default=256,
                        help="miRNA feature dimensions. Default is 256.")

    parser.add_argument("--disease-number",
                        type=int,
                        default=1,
                        help="disease number. Default is 591.")

    parser.add_argument("--fd",
                        type=int,
                        default=256,
                        help="disease number. Default is 256.")

    parser.add_argument("--view",
                        type=int,
                        default=4,
                        help="views number. Default is 2(2 datasets for miRNA and disease sim)")

    parser.add_argument("--n-neigh",
                        type=int,
                        default=3,
                        )
    parser.add_argument("--lr",
                        type=float,
                        default=0.001)

    print(parser.parse_args())

    return parser.parse_args()



def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def data_pro(args):
    dataset = dict()
    "CKSSAP similar matrices "
    #dd_f_matrix = read_csv(args.dataset_path + '/d_d_f.csv')
    CKSSAP_matrix = grid.run('data_CKSAAP.csv', args.n_neigh, args.lr, args.poch)
    dd_f_edge_index = get_edge_index(CKSSAP_matrix)
    dataset['CKSSAP'] = {'data_matrix': CKSSAP_matrix, 'edges': dd_f_edge_index}

    "TPC similar matrices "
    CTD_matrix = grid.run('data_TPC.csv', args.n_neigh, args.lr, args.poch)
    dd_s_edge_index = get_edge_index(CTD_matrix)
    dataset['TPC'] = {'data_matrix': CTD_matrix, 'edges': dd_s_edge_index}

    "CTDC similar matrices "
    CTDC_matrix = grid.run('data_CTDC.csv', args.n_neigh, args.lr, args.poch)
    mm_f_edge_index = get_edge_index(CTDC_matrix)
    dataset['CTDC'] = {'data_matrix': CTDC_matrix, 'edges': mm_f_edge_index}

    "CTDT similar matrices "
    CTDT_matrix = grid.run('data_CTDT.csv', args.n_neigh, args.lr, args.poch)
    mm_s_edge_index = get_edge_index(CTDT_matrix)
    dataset['CTDT'] = {'data_matrix': CTDT_matrix, 'edges': mm_s_edge_index}
    #print(dataset)


    return dataset


def metrics(y_true, y_pred, y_prob):

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

    print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
    print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
    print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
    print(
        'acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}'.format(
            accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc))
    # return (y_true, y_pred, y_prob), (accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc)
    return (y_true, y_pred, y_prob), [aupr, roc_auc, f1, accuracy, recall, specificity, precision], y_prob[-317:]

def evaluate(model,features, labels, mask, flag):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        probas = F.softmax(logits)
        """
        if flag == 0:
            #print(probas)
            print(probas[-1409:].detach().numpy())
        """
    return metrics(labels.cpu().detach(), indices.cpu().detach(), probas.cpu().detach()[:, 1])
    #return probas

def train(model,train_data, optimizer, opt):
    node_feature_label = pd.read_csv(r'./data/CKSAAP_x_all.csv', index_col=0)
    train_test_id_idx = np.load("./graph/Tp__imbalanced__testlabel0_knn_edge_train_test_index_all.npz",
        allow_pickle=True)
    train_index_all = train_test_id_idx['train_index_all']
    test_index_all = train_test_id_idx['test_index_all']


    for train_idx, test_idx in zip(train_index_all, test_index_all):
        train_mask = np.zeros((1938+100+212), dtype='int64')

        train_mask[train_idx] = 1  # [0 1 1 ... 0 1 1]
        test_mask = 1 - train_mask
        name = node_feature_label['1']
        name = name[test_idx]
        train_mask = torch.BoolTensor(train_mask)
        test_mask = torch.BoolTensor(test_mask)


    loss_fc = torch.nn.CrossEntropyLoss()
    # use optimizer
    # lr = 1e-4
    model.train()
    for epoch in range(0, opt.epoch):
        labels = torch.LongTensor(np.array(node_feature_label['2']))
        model.zero_grad()
        score = model(train_data)
        # print(score)
        loss = loss_fc(score[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        print('=====Epoch {} | Loss {:.4f}'.format(epoch, loss.item()))
        ys_train, metrics_train, result_train = evaluate(model, train_data, labels, train_mask, 1)
        ys_test, metrics_test, result_test = evaluate(model, train_data, labels, test_mask, 0)



    return ys_train, metrics_train, ys_test, metrics_test,result_test,name


import torch
from torch import nn
from torch_geometric.nn import GCNConv

torch.backends.cudnn.enabled = False


class MMGCN(nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(MMGCN, self).__init__()
        self.args = args

        # AvgPool2d(kernel_size=(256, 853), stride=(1, 1), padding=0)
        '''
        kernel_size( int or tuple)-
        stride( int or tuple) - 
        padding( int or tuple)-
        '''
        self.globalAvgPool_x = nn.AvgPool2d((self.args.fm, self.args.protein_number), (1, 1))
        # AvgPool2d(kernel_size=(256, 591), stride=(1, 1), padding=0)
        #self.globalAvgPool_y = nn.AvgPool2d((self.args.fd, self.args.disease_number), (1, 1))
        self.fc1_x = nn.Linear(in_features=self.args.view,
                               out_features=5 * self.args.view * self.args.gcn_layers)
        self.fc2_x = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers,
                               out_features=self.args.view)

        self.sigmoidx = nn.Sigmoid()
        self.sigmoidy = nn.Sigmoid()

        self.cnn_x = nn.Conv1d(in_channels=self.args.view,  # 4
                               out_channels=self.args.out_channels,  # 128
                               kernel_size=(self.args.fm, 1),
                               stride=1,
                               bias=True)

    def forward(self, data):
        torch.manual_seed(1)
        x_m = torch.randn(self.args.protein_number, self.args.fm)
        # print(x_m.shape) #1934,256

        x_CKSSAP_f2 = torch.relu(data['CKSSAP']['data_matrix']).detach()
        x_CTD_s2 = torch.relu(data['TPC']['data_matrix']).detach()
        x_CTDC_f2 = torch.relu(data['CTDC']['data_matrix']).detach()
        x_CTDT_s2 = torch.relu(data['CTDT']['data_matrix']).detach()


        XM = torch.cat((x_CKSSAP_f2, x_CTD_s2, x_CTDC_f2, x_CTDT_s2), 1).t()
        #print(XM.shape) #（1024， 853）
        XM = XM.view(1, self.args.view, self.args.fm, -1)
        # print(XM.shape)
        # print(XM)
        # (1,4,256,853)

        # print("XM.shape",XM.shape)  #([1, 4, 256, 853])
        x_channel_attenttion = self.globalAvgPool_x(XM)
        #([1,4,1,1])
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        #x_channel_attenttion = XM.view(XM.size(0), -1)
        # print("XM_channel-2",x_channel_attenttion.shape)   #[1,4]

        x_channel_attenttion = self.fc1_x(x_channel_attenttion)
        # print("XM_channel-3",x_channel_attenttion.shape) #[1,20]

        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc2_x(x_channel_attenttion)
        # print("XM_channel-4",x_channel_attenttion.shape)  #[1,4]

        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1,
                                                         1)
        # print("XM_channel-5",x_channel_attenttion.shape)  #[1,4,1,1]

        XM_channel_attention = x_channel_attenttion * XM
        # print("XM_channel-6",XM_channel_attention.shape)  #[1,4,1,1]

        XM_channel_attention = torch.relu(XM_channel_attention)
        # print("XM_channel-7",XM_channel_attention) #[1,4,256,853]

        # XM_channel1=XM_channel_attention.view(256,823,1,1)
        #print(XM_channel_attention.shape)

        x = self.cnn_x(XM_channel_attention)  # 853
        # print(x)
        # print(x.shape)
        x = x.view(self.args.out_channels, self.args.protein_number).t()
        #print(x)
        #print(x.shape)

        return x

def To_csv(file):
    points = file
    df = pd.read_csv('./data/protein_label.csv')
    name = pd.DataFrame(np.array(df.iloc[1933:, 1]))
    prob = points['0']
    result = pd.concat([name, prob], axis=1)
    column = ['name', 'prob']
    result = pd.DataFrame(np.array(result), columns=column)
    result.sort_values(by=['prob', 'name'], ascending=[False, True], inplace=True, ignore_index=True)
    #print(result)
    result.to_csv('result.csv')

def main():
    for poch in [200]:
        for n_epochs in [1000]:
            for i in [3]:
                for lr in [0.01]:
                        args = parameter_parser()
                        args.poch = poch
                        args.epochs = n_epochs
                        args.n_neigh = i
                        args.lr = lr
                        dataset = data_pro(args)
                        model = MMGCN(args)
                        #model.cuda()
                        lr=0.01
                        optimizer = torch.optim.Adam(model.parameters(),lr, weight_decay=5e-4)
                        ys_train, metrics_train, ys_test, metrics_test,result_test,name = train(model, dataset, optimizer, args)
                        d = pd.DataFrame(result_test)
                        To_csv(d)


if __name__ == "__main__":
    main()
