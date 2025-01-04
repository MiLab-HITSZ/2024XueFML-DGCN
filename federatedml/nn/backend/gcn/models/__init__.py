import torchvision.models as torch_models

from federatedml.nn.backend.gcn.models.ablations.connect_prob_standard_gcn import CONNECT_PROB_STANDARD_GCN
from federatedml.nn.backend.gcn.models.ablations.fixed_connect_prob_gcn import FIXED_CONNECT_PROB_GCN
from federatedml.nn.backend.gcn.models.ablations.fixed_connect_standard_gcn import \
    FIXED_CONNECT_STANDARD_GCN
from federatedml.nn.backend.gcn.models.ablations.fixed_prob_standard_gcn import FIXED_PROB_STANDARD_GCN
from federatedml.nn.backend.gcn.models.fml_dgcn.fixed_connect_prob_standard_gcn import \
    FIXED_CONNECT_PROB_STANDARD_GCN
from federatedml.nn.backend.gcn.models.old_studies.add_gcn import ORIGIN_ADD_GCN
from federatedml.nn.backend.gcn.models.old_studies.c_gcn import ResnetCGCN
from federatedml.nn.backend.gcn.models.old_studies.p_gcn import ResnetPGCN


def resnet_c_gcn(pretrained, adjList=None, device='cpu', num_classes=80, in_channel=300, dataset='coco', t=0.4):
    model = torch_models.resnet101(pretrained=pretrained)
    model = ResnetCGCN(model=model, num_classes=num_classes, in_channel=in_channel, t=t, adjList=adjList)
    return model.to(device)


def resnet_p_gcn(pretrained, adjList=None, device='cpu', num_classes=80, in_channel=2048, out_channel=1):
    model = torch_models.resnet101(pretrained=pretrained)
    model = ResnetPGCN(model=model, num_classes=num_classes, in_channel=in_channel, out_channels=out_channel,
                       adjList=adjList)
    return model.to(device)


def origin_add_gcn(pretrained, device='cpu', num_classes=80):
    model = torch_models.resnet101(pretrained)
    model = ORIGIN_ADD_GCN(model, num_classes=num_classes)
    return model.to(device)


def fixed_prob_standard_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                            out_channels=2048, prob=True, gap=False, isVOC=True):
    model = torch_models.resnet101(pretrained)
    model = FIXED_PROB_STANDARD_GCN(model, num_classes, in_channels, out_channels, adjList,
                                    needOptimize=isVOC)
    return model.to(device)


def fixed_connect_standard_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                               out_channels=2048, prob=True, gap=False, needOptimize=True):
    model = torch_models.resnet101(pretrained)
    model = FIXED_CONNECT_STANDARD_GCN(model, num_classes, in_channels, out_channels, adjList,
                                       needToOptimize=needOptimize)
    return model.to(device)


def fixed_connect_prob_standard_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                                    out_channels=2048, prob=True, gap=False, needOptimize=True,
                                    isVOC=True):
    model = torch_models.resnet101(pretrained)
    model = FIXED_CONNECT_PROB_STANDARD_GCN(model, num_classes, in_channels, out_channels, adjList,
                                            isVOC)
    return model.to(device)


def connect_prob_standard_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                              out_channels=2048, prob=True, gap=False, needOptimize=True,
                              isVOC=True):
    model = torch_models.resnet101(pretrained)
    model = CONNECT_PROB_STANDARD_GCN(model, num_classes, in_channels, out_channels, adjList,
                                      isVOC)
    return model.to(device)


def fixed_connect_prob_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                           out_channels=2048, isVOC=True):
    model = torch_models.resnet101(pretrained)
    model = FIXED_CONNECT_PROB_GCN(model, num_classes, in_channels, out_channels, adjList, isVOC)
    return model.to(device)
