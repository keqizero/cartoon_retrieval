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
    def __init__(self, list_style=False, no_norm=False, pretrained=True):
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

        #self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)
        
        #x = self.model.last_linear(x)
        return x
        #return torch.nn.functional.normalize(x, dim=-1) * 50

        #mod_x = self.model.last_linear(x)
        
        #return torch.nn.functional.normalize(mod_x, dim=-1)

class F_R(nn.Module):
    def __init__(self, cls_num=512):
        super(F_R, self).__init__()
        self.backbone = ResNet50()
        self.linear_layer = nn.Linear(2048, cls_num)

    def forward(self, x):
        x = self.backbone(x)
        pred = self.linear_layer(x)
        return x, pred
        
class C2R(nn.Module):
    """Network to learn text representations"""
    def __init__(self, cls_num=512):
        super(C2R, self).__init__()
        self.cartoon_net = ResNet50()
        #pretrained_model = torch.load('weights/c_best1.pt')
        #model_dict = self.cartoon_net.state_dict()
        #state_dict = {k:v for k,v in pretrained_model.items() if k in model_dict.keys()}
        #model_dict.update(state_dict)
        #self.cartoon_net.load_state_dict(model_dict)
        
        self.portrait_net = ResNet50()
        #pretrained_model = torch.load('weights/p_best1.pt')
        #model_dict = self.portrait_net.state_dict()
        #state_dict = {k:v for k,v in pretrained_model.items() if k in model_dict.keys()}
        #model_dict.update(state_dict)
        #self.portrait_net.load_state_dict(model_dict)
        
        self.c_linear = nn.Linear(2048, cls_num)
        self.p_linear = nn.Linear(2048, cls_num)

    def forward(self, cartoons=None, portraits=None):
        if cartoons == None:
            portraits_feature = self.portrait_net(portraits)
            portraits_predict = self.p_linear(portraits_feature)
            return portraits_feature, portraits_predict
        if portraits == None:
            cartoons_feature = self.cartoon_net(cartoons)
            cartoons_predict = self.c_linear(cartoons_feature)
            return cartoons_feature, cartoons_predict
        
        cartoons_feature = self.cartoon_net(cartoons)
        cartoons_predict = self.c_linear(cartoons_feature)
        
        portraits_feature = self.portrait_net(portraits)
        portraits_predict = self.p_linear(portraits_feature)

        return cartoons_feature, portraits_feature, cartoons_predict, portraits_predict  


class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class VideoNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=4096, output_dim=1024):
        super(VideoNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out
    
class IDCM_NN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, sketch_input_dim=512, sketch_output_dim=512,
                 video_input_dim=2048, video_output_dim=1024, minus_one_dim=512):
        super(IDCM_NN, self).__init__()
        self.sketch_net = ImgNN(sketch_input_dim, sketch_output_dim)
        self.video_net = VideoNN(video_input_dim, video_output_dim)
        self.linearLayer1 = nn.Linear(sketch_output_dim, minus_one_dim)
        self.linearLayer2 = nn.Linear(video_output_dim, minus_one_dim)
        
    def forward(self, cartoons=None, portraits=None):
        if cartoons == None:
            view2_feature = self.video_net(portraits)
            view2_feature = self.linearLayer2(view2_feature)
            return view2_feature, 0
        if portraits == None:
            view1_feature = self.sketch_net(cartoons)
            view1_feature = self.linearLayer1(view1_feature)
            return view1_feature, 0
        
        view1_feature = self.sketch_net(cartoons)
        view2_feature = self.video_net(portraits)
        view1_feature = self.linearLayer1(view1_feature)
        view2_feature = self.linearLayer2(view2_feature)

        return view1_feature, view2_feature, 0, 0


class C2R_Se(nn.Module):
    """Network to learn text representations"""
    def __init__(self, embed_dim=512):
        super(C2R_Se, self).__init__()
        self.cartoon_net = ResNet50(embed_dim)
        self.portrait_net = ResNet50(embed_dim)
        
        
        self.cartoon_attn1 = nn.Linear(4, 4)
        self.cartoon_attn2 = nn.Linear(4, 4)
        
        self.portrait_attn1 = nn.Linear(4, 4)
        self.portrait_attn2 = nn.Linear(4, 4)

    def forward(self, cartoons=None, portraits=None):
        if cartoons == None:
            portraits_feature = self.extract_portrait(portraits)
            return portraits_feature
        if portraits == None:
            cartoons_feature = self.extract_cartoon(cartoons)
            return cartoons_feature
        
        cartoons_feature = self.extract_cartoon(cartoons)
        portraits_feature = self.extract_portrait(portraits)

        return cartoons_feature, portraits_feature
    
    def extract_cartoon(self, cartoons):
        cartoons_height = cartoons.shape[2]
        cartoons_area = int(cartoons_height * 0.5)
        
        cartoons_feature1 = self.cartoon_net(cartoons).unsqueeze(1)
        cartoons_feature2 = self.cartoon_net(cartoons[:, :, 0:cartoons_area, :]).unsqueeze(1)
        cartoons_feature3 = self.cartoon_net(cartoons[:, :, int(cartoons_area * 0.5):int(cartoons_area * 1.5), :]).unsqueeze(1)
        cartoons_feature4 = self.cartoon_net(cartoons[:, :, cartoons_area:, :]).unsqueeze(1)
        
        cartoons_feature = torch.cat([cartoons_feature1, cartoons_feature2, cartoons_feature3, cartoons_feature4], 1)
        
        cartoons_attn = torch.sigmoid(self.cartoon_attn2(F.relu(self.cartoon_attn1(cartoons_feature.mean(-1)))))
        cartoons_attn = cartoons_attn.unsqueeze(-1).expand_as(cartoons_feature)
        cartoons_feature = (cartoons_feature * cartoons_attn).sum(-2)
        
        return cartoons_feature
    
    def extract_portrait(self, portraits):
        portraits_height = portraits.shape[2]
        portraits_area = int(portraits_height * 0.5)
        
        portraits_feature1 = self.portrait_net(portraits).unsqueeze(1)
        portraits_feature2 = self.portrait_net(portraits[:, :, 0:portraits_area, :]).unsqueeze(1)
        portraits_feature3 = self.portrait_net(portraits[:, :, int(portraits_area * 0.5):int(portraits_area * 1.5), :]).unsqueeze(1)
        portraits_feature4 = self.portrait_net(portraits[:, :, portraits_area:, :]).unsqueeze(1)
        
        portraits_feature = torch.cat([portraits_feature1, portraits_feature2, portraits_feature3, portraits_feature4], 1)
        
        portraits_attn = torch.sigmoid(self.portrait_attn2(F.relu(self.portrait_attn1(portraits_feature.mean(-1)))))
        portraits_attn = portraits_attn.unsqueeze(-1).expand_as(portraits_feature)
        portraits_feature = (portraits_feature * portraits_attn).sum(-2)
        
        return portraits_feature

    
class ResNet34(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, list_style=False, no_norm=False, pretrained=True):
        super(ResNet34, self).__init__()

        if pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__['resnet34'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet34'](num_classes=1000, pretrained=None)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        #self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)
        
        #x = self.model.last_linear(x)
        return x    

class C2R_(nn.Module):
    """Network to learn text representations"""
    def __init__(self, cls_num=512):
        super(C2R_, self).__init__()
        self.cartoon_net = ResNet34()
        pretrained_model = torch.load('weights/resnet34_adam.pth', map_location='cuda:0')
        model_dict = self.cartoon_net.state_dict()
        state_dict = {k:v for k,v in pretrained_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.cartoon_net.load_state_dict(model_dict)
        
        self.portrait_net = ResNet34()
        
        self.c_linear = nn.Linear(512, cls_num)
        self.p_linear = nn.Linear(512, cls_num)

    def forward(self, cartoons=None, portraits=None):
        if cartoons == None:
            portraits_feature = self.portrait_net(portraits)
            portraits_predict = self.p_linear(portraits_feature)
            return portraits_feature, portraits_predict
        if portraits == None:
            cartoons_feature = self.cartoon_net(cartoons)
            cartoons_predict = self.c_linear(cartoons_feature)
            return cartoons_feature, cartoons_predict
        
        cartoons_feature = self.cartoon_net(cartoons)
        cartoons_predict = self.c_linear(cartoons_feature)
        
        portraits_feature = self.portrait_net(portraits)
        portraits_predict = self.p_linear(portraits_feature)

        return cartoons_feature, portraits_feature, cartoons_predict, portraits_predict  