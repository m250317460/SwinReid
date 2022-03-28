# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
from swin_transformer.Swin_veri import swin_for_veri
from random_erasing import RandomErasing
import yaml
from shutil import copyfile
from circle_loss import CircleLoss, convert_label_to_similarity


version = torch.__version__
# fp16
try:
    from apex.fp16_utils import *
    from apex import amp
    from apex.optimizers import FusedSGD
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

from pytorch_metric_learning import losses, miners  # pip install pytorch-metric-learning

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default='/home/dataset/VeRi/pytorch', type=str, help='training dir path')


parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_swin', action='store_true', help='use swin transformer 224x224')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--total_epoch', default=60, type=int, help='total training epoch')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')

parser.add_argument('--circle', action='store_true', help='use Circle loss')

parser.add_argument('--ins_gamma', default=32, type=int, help='gamma for instance loss')


parser.add_argument('--pretrain', default='True', type=bool, help='默认使用预训练模型')
opt = parser.parse_args()


data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#

# if opt.use_swin:
h, w = 224, 224


transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    # resize使用插值
    transforms.Resize((h, w), interpolation=3),
    transforms.Pad(10),
    # 随即裁剪
    transforms.RandomCrop((h, w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(h, w), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]



if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

print(transform_train_list)

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
# 在这里不适用train——all
# if opt.train_all:
#     train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=2, pin_memory=True)  # 8 workers may work faster
               for x in ['train', 'val']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print("class_names:{}".format(len(class_names)))
use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time() - since)
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []




def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # best_model_wts = model.state_dict()
    # best_acc = 0.0

    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch

    # if opt.arcface:
    #     criterion_arcface = losses.ArcFaceLoss(num_classes=opt.nclasses, embedding_size=512)
    # if opt.cosface:
    #     criterion_cosface = losses.CosFaceLoss(num_classes=opt.nclasses, embedding_size=512)
    if opt.circle:
        # 只使用了circle损失
        criterion_circle = CircleLoss(m=0.25, gamma=32)  # gamma = 64 may lead to a better result.
    # if opt.triplet:
    #     miner = miners.MultiSimilarityMiner()
    #     criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    # if opt.lifted:
    #     criterion_lifted = losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0)
    # if opt.contrast:
    #     criterion_contrast = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    # if opt.instance:
    #     criterion_instance = InstanceLoss(gamma=opt.ins_gamma)
    # if opt.sphere:
    #     criterion_sphere = losses.SphereFaceLoss(num_classes=opt.nclasses, embedding_size=512, margin=4)
    # 开始训练
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    # 直接使用model得到输出
                    # 输入图片
                    outputs = model(inputs)
                    # print(type(outputs))
                    # temp = torch.tensor(outputs,device='cpu')
                    # temp = np.array(outputs.data.cpu())
                    # 查看一下outputs的样式
                    # [16,576]
                    # print("查看outputs[0]样式:{}".format(outputs[0].shape))
                    # [16,512]
                    # print("查看outputs[1]样式:{}".format(outputs[1].shape))

                    # print("查看outputs[2]样式:{}".format(outputs[2].shape))
                    # print(outputs.data)

                sm = nn.Softmax(dim=1)
                # log_sm = nn.LogSoftmax(dim=1)
                # return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.instance or opt.lifted or opt.sphere
                return_feature = opt.circle
                if return_feature:
                    # print("rf")
                    logits, ff ,fm = outputs
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                    loss = criterion(logits, labels)
                    _, preds = torch.max(logits.data, 1)

                    if opt.circle:
                        loss += criterion_circle(*convert_label_to_similarity(ff, labels)) / now_batch_size
                    # if opt.triplet:
                    #     hard_pairs = miner(ff, labels)
                    #     loss += criterion_triplet(ff, labels, hard_pairs)  # /now_batch_size
                    # if opt.lifted:
                    #     loss += criterion_lifted(ff, labels)  # /now_batch_size
                    # if opt.contrast:
                    #     loss += criterion_contrast(ff, labels)  # /now_batch_size
                    # if opt.instance:
                    #     loss += criterion_instance(ff, labels) / now_batch_size
                    # if opt.sphere:
                    #     loss += criterion_sphere(ff, labels) / now_batch_size
                else:  # norm
                    print("norm")
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                del inputs

                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss = loss * warm_up

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                del loss
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)
            if phase == 'train':
                scheduler.step()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print(time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime()))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))


######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

# return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.instance or opt.lifted or opt.sphere
return_feature = opt.circle

# 使用可编辑的swin模型
model = swin_for_veri(class_num=len(class_names),circle=True,pretrain=opt.pretrain)



opt.nclasses = len(class_names)
# print(model)

# model to gpu
model = model.cuda()



# 使用SGD优化器
optim_name = optim.SGD  # apex.optimizers.FusedSGD
# if opt.FSGD:  # apex is needed
#     optim_name = FusedSGD

# if not opt.PCB:
#     ignored_params = list(map(id, model.classifier.parameters()))
#     base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
#     classifier_params = model.classifier.parameters()
#     optimizer_ft = optim_name([
#         {'params': base_params, 'lr': 0.1 * opt.lr},
#         {'params': classifier_params, 'lr': opt.lr}
#     ], weight_decay=5e-4, momentum=0.9, nesterov=True)
# else:
#     ignored_params = list(map(id, model.model.fc.parameters()))
#     ignored_params += (list(map(id, model.classifier0.parameters()))
#                        + list(map(id, model.classifier1.parameters()))
#                        + list(map(id, model.classifier2.parameters()))
#                        + list(map(id, model.classifier3.parameters()))
#                        + list(map(id, model.classifier4.parameters()))
#                        + list(map(id, model.classifier5.parameters()))
#                        # +list(map(id, model.classifier6.parameters() ))
#                        # +list(map(id, model.classifier7.parameters() ))
#                        )
#     base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
#     classifier_params = filter(lambda p: id(p) in ignored_params, model.parameters())
#     optimizer_ft = optim_name([
#         {'params': base_params, 'lr': 0.1 * opt.lr},
#         {'params': classifier_params, 'lr': opt.lr}
#     ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
# 设置优化的参数
ignored_params = list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
classifier_params = model.classifier.parameters()
optimizer_ft = optim_name([
    {'params': base_params, 'lr': 0.1 * opt.lr},
    {'params': classifier_params, 'lr': opt.lr}
], weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=opt.total_epoch * 2 // 3, gamma=0.1)
# if opt.cosine:
#     exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, opt.total_epoch, eta_min=0.01 * opt.lr)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#

dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
# record every run
copyfile('./train.py', dir_name + '/train.py')
copyfile('./swin_transformer/Swin_veri.py', dir_name + '/model.py')

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

criterion = nn.CrossEntropyLoss()
# if fp16:
#     # model = network_to_half(model)
#     # optimizer_ft = FP16_Optimizer(optimizer_ft, static_loss_scale = 128.0)
#     model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=opt.total_epoch)

