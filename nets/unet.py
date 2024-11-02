import torch
import torch.nn as nn
import numpy as np

from nets.resnet import resnet50
from nets.vgg import VGG16
from nets.tranunet import vit_base_patch16_224

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
# class CBAM(nn.Module):
#     def __init__(self, in_planes, ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.ca = ChannelAttention(in_planes, ratio)
#         self.sa = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         out = x * self.ca(x)
#         result = out * self.sa(out)
#         return result



class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        outputs = self.up(outputs)
        return outputs







class Unet(nn.Module):
    def __init__(self, num_classes=6, pretrained=False, backbone='trans'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]

        elif backbone == "trans":
            self.trans = vit_base_patch16_224()
            in_filters = [192, 512, 1024, 1792]

        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        if backbone == 'swin':
            out_filters = [16, 64, 128, 256]
        else:
            out_filters = [64, 128, 256, 512]

        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # self.out_cbam4 = CBAM(out_filters[3])
        # self.out_cbam3 = CBAM(out_filters[2])
        # self.out_cbam2 = CBAM(out_filters[1])
        # self.out_cbam1 = CBAM(out_filters[0])
        #
        # self.in_cbam4 = CBAM(in_planes=768)
        # self.in_cbam3 = CBAM(in_planes=384)
        # self.in_cbam2 = CBAM(in_planes=192)
        # self.in_cbam1 = CBAM(in_planes=96)

        self.proj = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1, stride=1)


        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        elif backbone == 'trans':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[1], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),

            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)


        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)

        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        elif self.backbone == "trans":
            feat1, feat2, feat3, feat4 = self.trans.forward(inputs)
            # feat4 = self.in_cbam4(feat4)
            # feat3 = self.in_cbam3(feat3)
            # feat2 = self.in_cbam2(feat2)
            # feat1 = self.in_cbam1(feat1)

        if self.backbone == "trans":

            up4 = self.up_concat4(feat3, feat4)
            up3 = self.up_concat3(feat2, up4)
            up2 = self.up_concat2(feat1, up3)


        else:
            up4 = self.up_concat4(feat4, feat5)
            up3 = self.up_concat3(feat3, up4)
            up2 = self.up_concat2(feat2, up3)
            up1 = self.up_concat1(feat1, up2)
        if self.up_conv != None:
            if self.backbone == 'trans':
                up1 = self.up_conv(up2)
            else:
                up1 = self.up_conv(up1)
        final = self.final(up1)


        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False
        elif self.backbone == "swin":
            for param in self.swin.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
        elif self.backbone == "swin":
            for param in self.swin.parameters():
                param.requires_grad = True

