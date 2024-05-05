import argparse
import datetime
import logging
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from extensions.chamfer_dist import ChamferDistanceL2
from datasets.tools_dataset import ToolsDataset
import models as models
from logger import Logger
from utils import save_model, save_args

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--ckpt_dir', type=str, default=os.path.join(BASE_DIR,'checkpoint'), help='path to save checkpoint (default: ckpt)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='Point_PN_AE', help='model name')
    parser.add_argument('--file_path', default=os.path.join(ROOT_DIR,'data'), help='training datasets')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--input_num', type=int, default=2048, help='point number')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='decay rate')
    parser.add_argument('--seed', type=int, default=6212, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='workers')
    parser.add_argument('--optim', type=str, default="sgd", help='optimizer')
    parser.add_argument('--eps', type=float, default=0.4, help='smooth loss')

    return parser.parse_args()


def main():
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
    device = 'cuda'

    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(10)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg
    args.ckpt_dir = args.ckpt_dir + args.model + message + '-' + str(args.seed)
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.ckpt_dir, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    # Model
    printf(f"args: {args}")
    printf('==> Building model..')
    net = models.__dict__[args.model]()
    criterion = ChamferDistanceL2()
    net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    num_params = 0
    for p in net.parameters():
        if p.requires_grad:
            num_params += p.numel()
    printf("===============================================")
    printf("model parameters: " + str(num_params))
    printf("===============================================")

   
    start_epoch = 0
    
    save_args(args)
    logger = Logger(os.path.join(args.ckpt_dir, 'log.txt'), title="ModelNet" + args.model)
 

    printf('==> Preparing data..')
    train_loader = DataLoader(ToolsDataset(args), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
   
   
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, eps=1e-4)

    elif args.optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, eps=1e-4)

    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-5, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))

        train(net, train_loader, optimizer, criterion, args.eps, device,epoch)
    
        scheduler.step()

        if epoch%10==0:
            save_model(net, epoch, path=args.ckpt_dir,
            )
       
    logger.close()

   


def train(net, trainloader, optimizer, criterion, eps, device,epoch):
    net.train()
    ep_idx = 0
    time_cost = datetime.datetime.now()
    for batch_idx, data in enumerate(trainloader):
        data = data.to(device).float()
        data = data.permute(0, 2, 1)
        optimizer.zero_grad()
        rec_pc,global_feature = net(data)
        points = data.transpose(2,1)
        rec_pc = rec_pc.transpose(2,1)
        loss = criterion(rec_pc, points)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        print(loss)
        if (epoch%5 == 0):
            np.savetxt('res/pred_'+str(ep_idx)+'.txt',rec_pc.cpu().detach().numpy()[0])
            np.savetxt('res/gt_'+str(ep_idx)+'.txt',points.cpu().detach().numpy()[0])
            np.savetxt('res/glob_fea_'+str(ep_idx)+'.txt',global_feature.cpu().detach().numpy()[0])
            ep_idx+=1
    return 


if __name__ == '__main__':
    main()
