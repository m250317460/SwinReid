# 建立这个文件是为了区分开用于veri下游任务和swin模型
# 在这里添加classifier等后续操作,这样可以是swin加载预训练模型


from swin_transformer.model import swin_base_patch4_window7_224 as swin
import torch.nn as nn
from torch.nn import init
import torch

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            num_bottleneck = linear
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        fm = x
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f,fm]
        else:
            x = self.classifier(x)
            return x
class swin_for_veri(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512,pretrain=False):
        # 默认circle为0
        super(swin_for_veri, self).__init__()
        # 放弃直接使用timm创建swin,改用可编辑的swintransformer
        # model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        # class_num 576
        model = swin(num_classes=class_num)

        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model.head = nn.Sequential() # save memory
        self.model = model

        self.circle = circle
        # self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle)
        # 定义classifier
        # 添加了线性+norm+drop层
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)
        # self.classifier = ClassBlock(1024, 576, 0.5, linear=512, return_f=True)
        # 默认使用预训练模型
        if pretrain:
            # 这里尝试加入预训练
            weights_dict = torch.load("./pretrain/swin_base_patch4_window7_224.pth")["model"]
            # weights_dict = torch.load("./swin_transformer/swin_base_patch4_window7_224.h5")["model"]
            # print(weights_dict)
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                # print(k)
                if "head" in k:
                    del weights_dict[k]
                if "attn_mask" in k:
                    del weights_dict[k]

            model.load_state_dict(weights_dict, strict=True)


    def forward(self, x):
        # x = self.model.forward_features(x)
        x = self.model.forward_features(x)
        # 这里查看x的样子,[16,1024]
        # print("X的shape为:{}".format(x.shape))
        # print(x.shape)
        x = self.classifier(x)
        return x

    def load_param(self, model_path):

        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(model_path))


