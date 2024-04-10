import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])#64， 3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])#128 4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])#256 23
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)      # 256 * 256, 64
        x = self.maxpool(x)

        x = self.layer1(x)      # 128 * 128, 256
        fourth_level_feat = x 
        x = self.layer2(x)      # 64 * 64, 512
        eighth_level_feat = x 
        x = self.layer3(x)      # 32 * 32, 1024
        sixteenth_level_feat = x
        x = self.layer4(x)      # 32 * 32, 2048

        return x, fourth_level_feat, eighth_level_feat, sixteenth_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet50(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 6, 3], os, pretrained=pretrained)
    return model


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SSFA(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce_sixteen = nn.Sequential(nn.Conv2d(1024, 256, 1, stride=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))
        self.reduce_eight = nn.Sequential(nn.Conv2d(512, 256, 1, stride=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))                              

    def forward(self, x_enc, fourth_level_feature, eighth_level_feature, sixteenth_level_feature):

        # feature concatentation
        x_enc_1x = torch.cat((x_enc, self.reduce_sixteen(sixteenth_level_feature)), dim=1)                  # 32 * 32, 512
        x_enc_2x = F.interpolate(x_enc, size=eighth_level_feature.size()[2:], mode='bilinear', align_corners=False)
        x_enc_2x = torch.cat((x_enc_2x, self.reduce_eight(eighth_level_feature)), dim=1)                    # 64 * 64, 512
        x_enc_4x = F.interpolate(x_enc, size=fourth_level_feature.size()[2:], mode='bilinear', align_corners=False)
        x_enc_4x = torch.cat((x_enc_4x, fourth_level_feature), dim=1)                                       # 128 * 128, 512

        # aggregation
        sixteen_to_four = F.interpolate(x_enc_1x, size=fourth_level_feature.size()[2:], mode='bilinear', align_corners=False)
        eight_to_four = F.interpolate(x_enc_2x, size=fourth_level_feature.size()[2:], mode='bilinear', align_corners=False)
        stride_four = x_enc_4x + eight_to_four + sixteen_to_four

        sixteen_to_eight = F.interpolate(x_enc_1x, size=eighth_level_feature.size()[2:], mode='bilinear', align_corners=False)
        stride_eight = x_enc_2x + sixteen_to_eight

        eight_to_four = F.interpolate(stride_eight, size=fourth_level_feature.size()[2:], mode='bilinear', align_corners=False)
        x_enc = stride_four + eight_to_four + sixteen_to_four                                               # 128 * 128, 512

        return x_enc


class CA(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.conv_gray = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
                                       )

    def forward(self, x1, x2, input):
        x_gray = torch.abs(x1 - x2)                                     # channel = 512

        # 二类别的输出
        x_gray = self.conv_gray(x_gray)
        x_gray = F.interpolate(x_gray, size=input.size()[2:], mode='bilinear', align_corners=False)

        return x_gray


class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=2, os=16, pretrained=True, _print=False):
        if _print:
            print("Constructing SSESN_bcd model...")
            print("Number of classes       : {}".format(n_classes))
            print("Output stride           : {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
            print("Input shape             : {}".format("batchsize, 3, 512, 512"))
            print("Output shape            : {}".format("batchsize,21, 512, 512"))

        super(DeepLabv3_plus, self).__init__()

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(1280, 256, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True)
                                    )

        self.ssfa1 = SSFA()
        self.ssfa2 = SSFA()
                                        
        # SCD
        self.ca = CA(n_classes=n_classes)

        self.init_weight()

        # Atrous Conv
        self.backbone = ResNet50(nInputChannels, os, pretrained=pretrained)
        

    def forward_post(self, input):
        x_enc, fourth_level_feature, eighth_level_feature, sixteenth_level_feature = self.backbone(input)#final_x:[1, 2048, 32, 32]  low_level_features:[1,256, 128, 128]
        x_enc_1 = self.aspp1(x_enc)   #[1, 256, 32, 32]
        x_enc_2 = self.aspp2(x_enc)   #[1, 256, 32, 32]
        x_enc_3 = self.aspp3(x_enc)   #[1, 256, 32, 32]
        x_enc_4 = self.aspp4(x_enc)   #[1, 256, 32, 32]
        x_enc_5 = self.global_avg_pool(x_enc) #[1, 256, 1, 1]
        x_enc_5 = F.interpolate(x_enc_5, size=x_enc_4.size()[2:], mode='bilinear', align_corners=False)

        x_enc = torch.cat((x_enc_1, x_enc_2, x_enc_3, x_enc_4, x_enc_5), dim=1)
        x_enc = self.project(x_enc)

        return x_enc, fourth_level_feature, eighth_level_feature, sixteenth_level_feature

    def forward(self, input):#input 1, 6, 512, 512
        input1, input2 = torch.split(input, 3, 1)

        # enc_1
        x_enc1, fourth_level_features_1, eighth_level_features_1, sixteenth_level_features_1 = self.forward_post(input1)
        
        # spatial-semantic aggregration
        x1 = self.ssfa1(x_enc1, fourth_level_features_1, eighth_level_features_1, sixteenth_level_features_1)


        # enc_2
        x_enc2, fourth_level_features_2, eighth_level_features_2, sixteenth_level_features_2 = self.forward_post(input2)#final_x:[1, 2048, 32, 32]  low_level_features:[1,256, 128, 128]

        # spatial-semantic aggregration
        x2 = self.ssfa2(x_enc2, fourth_level_features_2, eighth_level_features_2, sixteenth_level_features_2)


        x_gray = self.ca(x1, x2, input)
        
        return x_gray

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


