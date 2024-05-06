"""
Author: Benny
Date: Nov 2019
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch.nn as nn
import datetime
import logging
import sys
import importlib
import shutil
import torch

from pathlib import Path
# from pointnet2.tools import *

# from extensions.chamfer_dist import ChamferFunction,ChamferDistanceL2
# from extensions.emd.emd import earth_mover_distance


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
ROOT = os.path.dirname(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(ROOT)
from openpoints.dataset.mydataset.tools_dataset import ToolsDataset
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg_siamese', help='model name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch Size during training')
    parser.add_argument('--epoch', default=100, type=int, help='epoch to run')
    parser.add_argument('--dataset', default='data2/Reisch', type=str, help='pu1k or pugan')
    parser.add_argument('--input_num', default=2048, type=str, help='optimizer, adam or sgd')
    parser.add_argument('--file_path', default='data2', type=str, help='the path of train dataset')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    # parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('siamese_tools')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    # checkpoints_dir = exp_dir.joinpath('checkpoints/')
    # checkpoints_dir.mkdir(exist_ok=True)
    checkpoints_dir=Path('./pointnet2/checkpoints')
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    '''LOG'''
    args = parse_args()
    args.file_path=os.path.join(ROOT,args.file_path)
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model.split('/')[-1]))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    TRAIN_DATASET = ToolsDataset(args)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=1, shuffle=True, num_workers=8, drop_last=True)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))


    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    # shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    siamese_model = MODEL.get_model().cuda()
    # criterion = MODEL.get_loss().cuda()
    siamese_model.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load('./pointnet2/checkpoints/best_model.pth')
        start_epoch = 0
        siamese_model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        siamese_model = siamese_model.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            siamese_model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(siamese_model.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    # criterion = ChamferDistanceL2()
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(start_epoch, args.epoch):

        ep_idx = 0
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        siamese_model = siamese_model.apply(lambda x: bn_momentum_adjust(x, momentum))
        siamese_model = siamese_model.train()

        '''learning one epoch'''
        for i, (input1,input2,label,msg) in enumerate(trainDataLoader):
            optimizer.zero_grad()

            input1 = input1.float().cuda()
            input1 = input1.transpose(2, 1)
            input2 = input2.float().cuda()
            input2 = input2.transpose(2, 1)
            label = label.view(1,1).float().cuda()
            pc_sim = siamese_model(input1,input2,training=True)
            loss = criterion(pc_sim, label)
            loss.backward()
            optimizer.step()

            logger.info('epoch {}/{}, batch {}/{}, kind of sample pairs:{}, loss: {:.10f}'.format(epoch,args.epoch,i,len(trainDataLoader),msg[0],loss.item()))


        if (epoch == args.epoch):
            logger.info('Save model...')
            savepath = os.path.join(checkpoints_dir,'best_model.pth')#str(checkpoints_dir) + 'model_{}.pth'.format(epoch)
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': siamese_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')




if __name__ == '__main__':
    args = parse_args()
    main(args)
