import torch
import torch.nn as nn
import numpy as np

# from nets.resnet import resnet50
# from nets.vgg import VGG16
from nets.swinunet_v2  import SwinTransformerSys



class SwinUnet(nn.Module):
    def __init__(self, num_classes=6, pretrained=False, backbone='swin'):
        super(SwinUnet, self).__init__()
        if backbone == 'swin':
            self.num_classes = num_classes
            self.swin = SwinTransformerSys(num_classes=self.num_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin.forward(x)
        return logits
        

    def freeze_backbone(self):
            for param in self.swin.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
            for param in self.swin.parameters():
                param.requires_grad = True



