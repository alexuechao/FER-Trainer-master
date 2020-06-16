import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import load_state_dict_from_url

from .resnet import (
    ResNet,
    BasicBlock
)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class ResNetCenterLoss(ResNet):
    def __init__(self, block, num_blocks):
        super(ResNetCenterLoss, self).__init__(
            block=BasicBlock,
            num_blocks = num_blocks,
            num_classes=7
        )
        # state_dict = load_state_dict_from_url(model_urls['resnet18'])
        # self.load_state_dict(state_dict)
 
        # for center loss
        self.center_loss_fc = nn.Linear(512, 2)
        self.linear = nn.Linear(512, 7)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 6)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        features = self.relu(self.center_loss_fc(out))
        out = self.linear(out)
        return features, out

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)

    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
        
    #     features = self.relu(self.center_loss_fc(x))
    #     #outputs = self.fc(x)
    #     outputs = self.linear(x)
    #     return features, outputs
        #return outputs

def resnet18_centerloss(pretrained=False, progress=True, **kwargs):
    return ResNetCenterLoss(block=BasicBlock, num_blocks=[2, 2, 2, 2])