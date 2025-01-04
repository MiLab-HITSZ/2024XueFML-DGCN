import math
import torch
import torch.nn as nn
from torch.nn import Parameter

from ..fml_dgcn.custom_matrix import CustomMatrix, genAdj


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, adjList=None, needToOptimize=True):
        super(DynamicGraphConvolution, self).__init__()
        self.static_adj = CustomMatrix(adjList, needToOptimize=needToOptimize)

        self.static_weight = Parameter(torch.Tensor(in_features, in_features))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1) 
        self.dynamic_weight = Parameter(torch.Tensor(in_features, out_features))

        self.reset_weight_parameters()

    def reset_weight_parameters(self):

        static_stdv = 1. / math.sqrt(self.static_weight.size(1))
        self.static_weight.data.uniform_(-static_stdv, static_stdv)

        dynamic_stdv = 1. / math.sqrt(self.dynamic_weight.size(1))
        self.dynamic_weight.data.uniform_(-dynamic_stdv, dynamic_stdv)

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def gen_adjs(self, A):
        batch_size = A.size(0)
        adjs = torch.zeros_like(A)
        for i in range(batch_size):
            D = torch.pow(A[i].sum(1).float(), -0.5)
            D = torch.diag(D)
            adj = torch.matmul(torch.matmul(A[i], D).t(), D)
            adjs[i] = adj
        return adjs

    def forward_gcn(self, input, weight, adj):
        output = torch.matmul(adj, input)
        output = self.relu(output)
        output = torch.matmul(output, weight)
        return output

    def forward_static_gcn(self, x):
        adj = self.static_adj()
        returned_adj = adj
        adj = self.gen_adj(adj)
        x = self.forward_gcn(x, self.static_weight, adj)
        return x, returned_adj

    def forward_construct_dynamic_graph(self, x, connect_vec):

        connect_vec = connect_vec.unsqueeze(-1).expand(connect_vec.size(0), connect_vec.size(1), x.size(2))

        x = torch.cat((connect_vec, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)

        dynamic_adj = torch.sigmoid(dynamic_adj)
        dynamic_adj = genAdj(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):

        transformed_adjs = self.gen_adjs(dynamic_adj)
        x = self.forward_gcn(x, self.dynamic_weight, transformed_adjs)
        x = self.relu(x)
        return x

    def forward(self, x, connect_vec, out1):

        out_static, static_adj = self.forward_static_gcn(x)
        x = x + out_static
        x = x.transpose(1, 2)
        dynamic_adj = self.forward_construct_dynamic_graph(x, connect_vec)
        x = x.transpose(1, 2)
        x = self.forward_dynamic_gcn(x, dynamic_adj)

        return x


class FIXED_CONNECT_STANDARD_GCN(nn.Module):
    def __init__(self, model, num_classes, in_features=300, out_features=2048, adjList=None, needToOptimize=True):
        super(FIXED_CONNECT_STANDARD_GCN, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.gcn = DynamicGraphConvolution(in_features, out_features, num_classes, adjList,
                                           needToOptimize=needToOptimize)

        feat_dim = 2048
        self.connect = torch.nn.Linear(in_features=feat_dim, out_features=in_features, bias=True)
        self.fc = torch.nn.Linear(in_features=feat_dim, out_features=num_classes, bias=True)

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_dgcn(self, x, connect_vec, out1):
        x = self.gcn(x, connect_vec, out1)
        return x

    def forward(self, x, inp):
        x = self.forward_feature(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out1 = self.fc(x)

        connect_vec = self.connect(x)
        z = self.forward_dgcn(inp, connect_vec, out1)

        out2 = torch.matmul(z, x.unsqueeze(-1)).squeeze(-1) / z.size(2)

        return out1, out2

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]

