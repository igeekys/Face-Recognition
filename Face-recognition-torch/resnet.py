from collections import namedtuple

import torch
import torch.nn as nn
import torchvision.models as models

from model_op import count_parameters


class ResNet(nn.Module):
    def __init__(self, config, output_dim, is_se=False):
        super().__init__()
                
        block, n_blocks, channels = config
        self.in_channels = channels[0]
        self.is_se = is_se
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):
    
        layers = []
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        layers.append(block(self.in_channels, channels, stride, downsample, self.is_se))
        
        for _ in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels, self.is_se))

        self.in_channels = block.expansion * channels
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x, h

class SE(nn.Module):
    def __init__(self, in_channels, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channels, in_channels//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_channels//ratio, in_channels, 1, 1, 0)
        self.relu  = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = self.relu(out)
        out = self.excitation(out)
        return self.sigmoid(out)

class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False, is_se=False):
        super().__init__()
                
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        self.is_se = is_se
        if self.is_se:
            self.se = SE(out_channels, 16)

        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        self.downsample = downsample
        
    def forward(self, x):
        i = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.is_se:
            coefficient = self.se(x)
            x *= coefficient

        if self.downsample is not None:
            i = self.downsample(i)     
        x += i
        x = self.relu(x)
        return x

class Bottleneck(nn.Module):
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False, is_se=False):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                               stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        self.is_se = is_se

        if self.is_se:
            self.se = SE(out_channels * self.expansion, 16)

        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        if self.is_se:
            coefficient = self.se(x)
            x *= coefficient

        if self.downsample is not None:
            i = self.downsample(i)
            
        x += i
        x = self.relu(x)
    
        return x

def build_model(backbone, OUTPUT_DIM, pretrained=False, model_path=None,is_se=False):
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
    resnet18_config = ResNetConfig(block = BasicBlock,
                                n_blocks = [2,2,2,2],
                                channels = [64, 128, 256, 512])

    resnet34_config = ResNetConfig(block = BasicBlock,
                                n_blocks = [3,4,6,3],
                                channels = [64, 128, 256, 512])

    resnet50_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 4, 6, 3],
                                channels = [64, 128, 256, 512])

    resnet101_config = ResNetConfig(block = Bottleneck,
                                    n_blocks = [3, 4, 23, 3],
                                    channels = [64, 128, 256, 512])

    resnet152_config = ResNetConfig(block = Bottleneck,
                                    n_blocks = [3, 8, 36, 3],
                                    channels = [64, 128, 256, 512])
    if backbone == 'resnet50':
        pretrained_model = models.resnet50(pretrained)
        model = ResNet(resnet50_config, OUTPUT_DIM, is_se)
    elif backbone =='resnet18':
        pass
    else:
        pass

    if model_path:
            pretrained_model.load_state_dict(torch.load(model_path))
    # print(pretrained_model)

    IN_FEATURES = pretrained_model.fc.in_features 

    fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    pretrained_model.fc = fc

    model.load_state_dict(pretrained_model.state_dict(), False)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    return model
