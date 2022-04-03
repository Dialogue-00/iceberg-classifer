from torchvision import models
import torch
import torch.nn as nn
from model.vit import vit_base_patch16_224_in21k as create_model
# from vit import vit_base_patch16_224_in21k as create_model   # for debug
import argparse

def get_model(device, params):
    if params.model == 'resnet18':
        model = models.resnet18(pretrained=params.pretrain).to(device)
        cnn_features = model.fc.in_features
        num_classes = 2
        model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
        nn.init.xavier_uniform_(model.fc.weight)
    if params.model == 'resnet34':
        model = models.resnet34(pretrained=params.pretrain).to(device)
        cnn_features = model.fc.in_features
        num_classes = 2
        model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
        nn.init.xavier_uniform_(model.fc.weight)
    if params.model == 'resnet50':
        model = models.resnet50(pretrained=params.pretrain).to(device)
        cnn_features = model.fc.in_features
        num_classes = 2
        model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
        nn.init.xavier_uniform_(model.fc.weight)
    if params.model == 'vgg16':
        model = models.vgg16(pretrained=params.pretrain).to(device)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 2).to(device)
        nn.init.xavier_uniform_(model.classifier[6].weight)
    if params.model == 'vgg19':
        model = models.vgg19(pretrained=params.pretrain).to(device)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 2).to(device)
        nn.init.xavier_uniform_(model.classifier[6].weight)
    if params.model == 'alexnet':
        model = models.alexnet(pretrained=params.pretrain).to(device)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 2).to(device)
        nn.init.xavier_uniform_(model.classifier[6].weight)
    if params.model == 'vit':
        model = create_model(num_classes=2, has_logits=False).to(device)

        if str(params.pretrain) == 'True':
            weights_path = './model/vit_base_patch16_224.pth'
            weights_dict = torch.load(weights_path, map_location=device)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))
            print("pretrain")
            # for name, para in model.named_parameters():
            #     # 除head, pre_logits外，其他权重全部冻结
            #     if "head" not in name and "pre_logits" not in name:
            #         para.requires_grad_(False)
            #     else:
            #         print("training {}".format(name))
        elif params.pretrain is False:
            print("no pretrain")
        
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-m', '--model', type=str, default='vit')
    parser.add_argument('-p', '--pretrain', type=bool, default=False)
    params = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device, params)
    # for name in model.state_dict():
    #     print(name)
    # print(model.state_dict()['conv1.weight'])
    # print(params.batch_size)
    # print(len(train_dataloader.dataset))