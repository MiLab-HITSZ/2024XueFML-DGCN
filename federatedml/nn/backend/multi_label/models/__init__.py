import torch.nn

resnet_models = dict()
resnet_models[50] = torch_models.resnet50
resnet_models[101] = torch_models.resnet101

def create_resnet101_model(pretrained, device, num_classes=80, layer_num=101):
    model = resnet_models[layer_num](pretrained=pretrained, num_classes=1000)
    model.fc = torch.nn.Sequential(torch.nn.Linear(2048, num_classes))
    torch.nn.init.kaiming_normal_(model.fc[0].weight.data)
    return model.to(device)
