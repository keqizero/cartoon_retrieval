import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels as ptm
    
class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features

class ResNet50(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, embed_dim=512, list_style=False, no_norm=False, pretrained=True):
        super(ResNet50, self).__init__()

        if pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)
        
        return x

        #mod_x = self.model.last_linear(x)
        
        #return torch.nn.functional.normalize(mod_x, dim=-1)

class IDCM_NN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, embed_dim=512):
        super(IDCM_NN, self).__init__()
        self.cartoon_net = ResNet50(embed_dim)
        self.portrait_net = ResNet50(embed_dim)

    def forward(self, cartoons=None, portraits=None):
        if cartoons == None:
            portraits_feature = self.portrait_net(portraits)
            return portraits_feature
        if portraits == None:
            cartoons_feature = self.cartoon_net(cartoons)
            return cartoons_feature
        
        cartoons_feature = self.cartoon_net(cartoons)
        portraits_feature = self.portrait_net(portraits)

        return cartoons_feature, portraits_feature
