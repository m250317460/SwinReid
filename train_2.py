# 该文件主要差别是使用了yacs进行参数的管理
# 关于yacs见https://github.com/rbgirshick/yacs

from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from torch.optim import lr_scheduler
import torch.optim as optim

from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# python train.py --config_file configs/VeRi/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    # 参数的初始化
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # 冻结参数
    cfg.freeze()
    # seed的默认值，1234
    set_seed(cfg.SOLVER.SEED)

    # 是否使用多GPU，默认false
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
    # torch.cuda.set_device(args.local_rank)

    # OUTPUT_DIR: './logs/veri_vit_transreid_stride'
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # name,save_dir,iftrain
    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    # normal是使用val的trans变化的train数据
    # 这里的trans有点不同，暂时先记着，后面再做对比
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    print("view_num:"+str(view_num))
    # numclass 776,camera_num 20 view_num 8
    # 这边为了契合车辆把view改为0，并且不使用JPM，查看效果
    # 这里在原来代码上支持了swin
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)


    model = torch.nn.DataParallel(model).cuda()

    # 这里的loss函数使用了center和triplet两个，在person里面只使用了交叉熵和circle
    # loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    # 使用center和triplet效果比较差，这边尝试用回交叉熵和circle，这边定义不用管，在训练的时候不使用就可以了

    # 优化器使用SGD
    # optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    # scheduler = create_scheduler(cfg, optimizer)
    # 放弃使用cos更新算法,改用train中的setp固定值更新算法

    # do_train(
    #     cfg,
    #     model,
    #     center_criterion,
    #     train_loader,
    #     val_loader,
    #     optimizer,
    #     optimizer_center,
    #     scheduler,
    #     loss_func,
    #     num_query,
    #     args.local_rank
    # )
    optim_name = optim.SGD  # apex.optimizers.FusedSGD
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    classifier_params = model.classifier.parameters()
    optimizer_ft = optim_name([
        {'params': base_params, 'lr': 0.1 * cfg.SOLVER.BASE_LR},
        {'params': classifier_params, 'lr': cfg.SOLVER.BASE_LR}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=cfg.SOLVER.MAX_EPOCHS * 2 // 3, gamma=0.1)
    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer_ft,
        exp_lr_scheduler,
        num_query,
        args.local_rank
    )