import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
from openpoints.dataset.mydataset.tools_dataset import ToolsDataset

CURRENT_PATH = os.path.abspath(__file__)
BASE_1 = os.path.dirname(CURRENT_PATH)
BASE_2 = os.path.dirname(BASE_1)
ROOT = os.path.dirname(BASE_2)

def main(gpu, cfg, profile=False):
    
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)
    TRAIN_DATASET = ToolsDataset(cfg)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=1, shuffle=True, num_workers=8, drop_last=True)

    criterion = nn.BCEWithLogitsLoss()
    model.zero_grad()
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):

        loss = train_one_epoch(model, trainDataLoader,
                            optimizer, scheduler, epoch,cfg.epochs, cfg,criterion)


        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'loss {loss:.10f}')

        if cfg.sched_on_epoch:
            scheduler.step(epoch)

        if (epoch == cfg.epochs):
            print('Save model...')
            savepath = os.path.join(BASE_2,'sim_checkpoint/best_model.pth')
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)


def train_one_epoch(model, train_loader, optimizer, scheduler, epoch,endepochs, cfg,criterion):
   
    

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, (input1,input2,label,msg) in pbar:

        num_iter += 1
        input1 = input1.float().cuda()
        # input1 = input1.transpose(2, 1)
        input2 = input2.float().cuda()
        # input2 = input2.transpose(2, 1)
        label = label.view(-1,1).float().cuda()
        pc_sim = model(input1,input2)
        loss = criterion(pc_sim, label)
        loss.backward()
        print('epoch {}/{}, batch {}/{}, kind of sample pairs:{}, loss: {:.10f}'.format(epoch,endepochs,num_iter,len(train_loader),msg[0],loss.item()))
        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

    return loss.item()

