import os
import glob
import sys
import yaml
import json
import shutil

from tqdm import tqdm
import logging
import warnings
import pprint

import open3d as o3d
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
import torch.nn.functional as F

torch.backends.cudnn.benchmark = False
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from openpoints.models import build_model_from_cfg
from openpoints.models.layers import torch_grouping_operation, knn_point
from openpoints.loss import build_criterion_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_class_weights, get_features_by_keys
from openpoints.transforms import build_transforms_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.models.layers import furthest_point_sample

def is_number(element):
    return isinstance(element, (int, float, complex))

bag_labels = [4.0,5.0]

dataset_base_dir = "./data/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal"

DATA_PATH_DIR = f"{dataset_base_dir}/02773838"
CONFIG_PATH = "./welding_seam/cfgs/shapenetpart/pointnext-s.yaml"


def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def part_seg_refinement(pred, pos, cls, cls2parts, n=10):
    pred_np = pred.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):  # sample_idx
        parts = cls2parts[cls[shape_idx]]
        counter_part = Counter(pred_np[shape_idx])
        if len(counter_part) > 1:
            for i in counter_part:
                if counter_part[i] < n or i not in parts:
                    less_idx = np.where(pred_np[shape_idx] == i)[0]
                    less_pos = pos[shape_idx][less_idx]
                    knn_idx = knn_point(n + 1, torch.unsqueeze(less_pos, axis=0),
                                        torch.unsqueeze(pos[shape_idx], axis=0))[1]
                    neighbor = torch_grouping_operation(pred[shape_idx:shape_idx + 1].unsqueeze(1), knn_idx)[0][0]
                    counts = batched_bincount(neighbor, 1, cls2parts[-1][-1] + 1)
                    counts[:, i] = 0
                    pred[shape_idx][less_idx] = counts.max(dim=1)[1]
    return pred


def get_ins_mious(pred, target, cls, cls2parts,
                  multihead=False,
                  ):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    ins_mious = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        parts = cls2parts[cls[shape_idx]]
        if multihead:
            parts = np.arange(len(parts))

        for part in parts:
            pred_part = pred[shape_idx] == part
            target_part = target[shape_idx] == part
            I = torch.logical_and(pred_part, target_part).sum()
            U = torch.logical_or(pred_part, target_part).sum()
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    return ins_mious


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        # Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    # build dataset
    cfg.batch_size = 1
    val_loader = build_dataloader_from_cfg(cfg.batch_size,
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}")
    cfg.cls2parts = val_loader.dataset.cls2parts
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    if cfg.model.get('decoder_args', False):
        cfg.model.decoder_args.cls2partembed = val_loader.dataset.cls2partembed
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).cuda()
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')


    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # transforms
    if 'vote' in cfg.datatransforms:
        voting_transform = build_transforms_from_cfg('vote', cfg.datatransforms)
    else:
        voting_transform = None

    model_module = model.module if hasattr(model, 'module') else model
    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            test_ins_miou, test_cls_miou, test_cls_mious = validate_fn(model, val_loader, cfg,
                                                                            num_votes=cfg.num_votes,
                                                                            data_transform=voting_transform
                                                                            )

            logging.info(f'\nresume val instance mIoU is {test_ins_miou}, val class mIoU is {test_cls_miou} \n ')
        else:
            if cfg.mode in ['val', 'test']:
                load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                test_ins_miou, test_cls_miou, test_cls_mious = validate_fn(model, val_loader, cfg,
                                                                            num_votes=cfg.num_votes,
                                                                            data_transform=voting_transform
                                                                            )
                return test_ins_miou
            elif cfg.mode == 'finetune':
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, pretrained_path=cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                logging.info(f'Load encoder only, finetuning from {cfg.pretrained_path}')
                load_checkpoint(model_module.encoder, pretrained_path=cfg.pretrained_path)
    else:
        logging.info('Training from scratch')

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    if cfg.get('cls_weighed_loss', False):
        if hasattr(train_loader.dataset, 'num_per_class'):
            cfg.criterion_args.weight = None
            cfg.criterion_args.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=True)
        else:
            logging.info('`num_per_class` attribute is not founded in dataset')
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    # ===> start training
    best_ins_miou, cls_miou_when_best, cls_mious_when_best = 0., 0., []
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        # some dataset sets the dataset length as a fixed steps.
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        cfg.epoch = epoch
        train_loss = \
            train_one_epoch(model, train_loader, criterion,
                            optimizer, scheduler, epoch, cfg)
        is_best = False
        if epoch % cfg.val_freq == 0:
            val_ins_miou, val_cls_miou, val_cls_mious = validate_fn(model, val_loader, cfg)
            if val_ins_miou > best_ins_miou:
                best_ins_miou = val_ins_miou
                cls_miou_when_best = val_cls_miou
                cls_mious_when_best = val_cls_mious
                best_epoch = epoch
                is_best = True
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_ins_miou {best_ins_miou:.2f} val_cls_miou {cls_miou_when_best:.2f}, '
                        f'\ncls_mious: {cls_mious_when_best}')

        lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar('val_ins_miou', val_ins_miou, epoch)
            writer.add_scalar('val_class_miou', val_cls_miou, epoch)
            writer.add_scalar('best_val_instance_miou',
                              best_ins_miou, epoch)
            writer.add_scalar('val_class_miou_when_best', cls_miou_when_best, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('lr', lr, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)

        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'ins_miou': best_ins_miou,
                                             'cls_miou': cls_miou_when_best},
                            is_best=is_best
                            )
    # if writer is not None:
    #     Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.logname}_ckpt_latest.pth'))
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Best Epoch {best_epoch},'
                     f'Instance mIoU {best_ins_miou:.2f}, '
                     f'Class mIoU {cls_miou_when_best:.2f}, '
                     f'\n Class mIoUs {cls_mious_when_best}')

    if cfg.get('num_votes', 0) > 0:
        load_checkpoint(model, pretrained_path=os.path.join(
            cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        set_random_seed(cfg.seed)
        test_ins_miou, test_cls_miou, test_cls_mious  = validate_fn(model, val_loader, cfg, num_votes=cfg.get('num_votes', 0),
                                 data_transform=voting_transform)
        with np.printoptions(precision=2, suppress=True):
            logging.info(f'---Voting---\nBest Epoch {best_epoch},'
                        f'Voting Instance mIoU {test_ins_miou:.2f}, '
                        f'Voting Class mIoU {test_cls_miou:.2f}, '
                        f'\n Voting Class mIoUs {test_cls_mious}')

        if writer is not None:
            writer.add_scalar('test_ins_miou_voting', test_ins_miou, epoch)
            writer.add_scalar('test_class_miou_voting', test_cls_miou, epoch)
    torch.cuda.synchronize()
    if writer is not None:
        writer.close()
    
    # 修改为了避免出现 assert pg is not None 错误
    if torch.distributed.is_initialized():
        dist.destroy_process_group()
    # wandb.finish(exit_code=True)


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        num_iter += 1
        batch_size, num_point, _ = data['pos'].size()
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        data['x'] = get_features_by_keys(data, cfg.feature_keys)

        logits = model(data)
        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            loss = criterion(logits, target)
        else:
            loss = criterion(logits, target, data['cls'])

        loss.backward()

        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        loss_meter.update(loss.item(), n=batch_size)
        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.avg:.3f} "
                                 )
    train_loss = loss_meter.avg
    return train_loss


@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=0, data_transform=None):

    model.eval()  # set model to eval mode
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    cls_mious = torch.zeros(cfg.shape_classes, dtype=torch.float32).cuda(non_blocking=True)
    cls_nums = torch.zeros(cfg.shape_classes, dtype=torch.int32).cuda(non_blocking=True)
    ins_miou_list = []

    # label_size: b, means each sample has one corresponding class
    for idx, data in pbar:

        # [8:2048]
        # key pos,x,y,cls
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        cls = data['cls']
        # cls = 
        
        print(data['heights'])
        print(f"shape fo data['heights'] = {data['heights'].shape}")
        print(f"shape fo data['heights'] = {data['heights'].shape}")
        
        data['x'] = get_features_by_keys(data, cfg.feature_keys)


        batch_size, num_point, _ = data['pos'].size()
        logits = 0
        for v in range(num_votes+1):
            set_random_seed(v)
            if v > 0:
                data['pos'] = data_transform(data['pos'])
            print(data.keys())
            print(data['x'].shape)
            logits += model(data)
            print(logits.shape)
        logits /= (num_votes + 1)
        preds = logits.max(dim=1)[1]
        if cfg.get('refine', False):
            part_seg_refinement(preds, data['pos'], data['cls'], cfg.cls2parts, cfg.get('refine_n', 10))

        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            batch_ins_mious = get_ins_mious(preds, target, data['cls'], cfg.cls2parts)
            ins_miou_list += batch_ins_mious
        else:
            iou_array = []
            for ib in range(batch_size):
                sl = data['cls'][ib][0]
                iou = get_ins_mious(preds[ib:ib + 1], target[ib:ib + 1], sl.unsqueeze(0), cfg.cls2parts,
                                    multihead=True)
                iou_array.append(iou)
            ins_miou_list += iou_array

        # per category iou at each batch_size:
        for shape_idx in range(batch_size):  # sample_idx
            cur_gt_label = int(cls[shape_idx].cpu().numpy())
            # add the iou belongs to this cat
            cls_mious[cur_gt_label] += batch_ins_mious[shape_idx]
            cls_nums[cur_gt_label] += 1

    ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()
    if cfg.distributed:
        dist.all_reduce(cls_mious), dist.all_reduce(cls_nums), dist.all_reduce(ins_mious_sum), dist.all_reduce(count)

    for cat_idx in range(cfg.shape_classes):
        # indicating this cat is included during previous iou appending
        if cls_nums[cat_idx] > 0:
            cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

    ins_miou = ins_mious_sum/count
    cls_miou = torch.mean(cls_mious)
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Test Epoch [{cfg.epoch}/{cfg.epochs}],'
                        f'Instance mIoU {ins_miou:.2f}, '
                        f'Class mIoU {cls_miou:.2f}, '
                        f'\n Class mIoUs {cls_mious}')
    return ins_miou, cls_miou, cls_mious


def obj2txt(folder_data):
    bag_labels = [4.0,5.0]
    for obj_file_path in glob.glob(f"{folder_data}/*.obj"):
        print(obj_file_path)
        mesh = o3d.io.read_triangle_mesh(obj_file_path)
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        pcd = mesh.sample_points_poisson_disk(number_of_points=10_000)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)

        labels = np.full((points.shape[0],1),5.0)
        points_and_noprmal_concat = np.hstack((points, normals))
        res = np.column_stack((points_and_noprmal_concat, labels))
        num_samples = 2048
        sampled_idx = np.random.choice(res.shape[0], num_samples, replace=False)
        sampled_result = res[sampled_idx,:]
        with open(f"{dataset_base_dir}/02773838/{os.path.basename(obj_file_path).split('.')[0]}.txt", 'w') as file:
            for row in sampled_result:
                file.write(' '.join(map(str, row)) + '\n')

def update_train_test_json():
    workpiece_list = glob.glob(f"{dataset_base_dir}/02773838/*.txt")
    workpiece_list = [ f"shape_data/02773838/{os.path.basename(x).replace('.txt','')}" for x in workpiece_list]

    with open(f"{dataset_base_dir}/train_test_split/shuffled_test_file_list.json") as f:
        shuffled_test_file_list_json= json.load(f)

    shuffled_test_file_list = []

    for row in shuffled_test_file_list_json:
        if row.split("/")[1] != "02773838":
            shuffled_test_file_list.append(row)

    shuffled_test_file_list.extend(workpiece_list)
    # print(shuffled_test_file_list)


    with open(f"{dataset_base_dir}/train_test_split/shuffled_test_file_list.json","w") as fout:
        json.dump(shuffled_test_file_list,fout)

    with open(f"{dataset_base_dir}/train_test_split/shuffled_train_file_list.json") as f:
        shuffled_train_file_list_json= json.load(f)

    print(len(shuffled_train_file_list_json))

    shuffled_train_file_list = []

    for row in shuffled_train_file_list_json:
        if row.split("/")[1] != "02773838":
            shuffled_train_file_list.append(row)

    shuffled_train_file_list.extend(workpiece_list)
    print(shuffled_train_file_list)

    with open(f"{dataset_base_dir}/train_test_split/shuffled_train_file_list.json","w") as fout:
        json.dump(shuffled_train_file_list,fout)


    with open(f"{dataset_base_dir}/train_test_split/shuffled_val_file_list.json") as f:
        shuffled_val_file_list_json= json.load(f)

    print(len(shuffled_val_file_list_json))

    shuffled_val_file_list = []

    for row in shuffled_val_file_list_json:
        if row.split("/")[1] != "02773838":
            shuffled_val_file_list.append(row)

    shuffled_val_file_list.extend(workpiece_list)
    print(shuffled_val_file_list)


    with open(f"{dataset_base_dir}/train_test_split/shuffled_val_file_list.json","w") as fout:
        json.dump(shuffled_val_file_list,fout)
    
    # 删除 pkl 文件
    if os.path.exists(f"{dataset_base_dir}/processed/test_2048_fps.pkl"):
        os.remove(f"{dataset_base_dir}/processed/test_2048_fps.pkl")

def welding_seam_train_prepare(folder_data):
    # print(folder_data)
    for pcd_file in glob.glob(f"{folder_data}/*.pcd"):
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd.estimate_normals()

        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)

        labels = []
        with open(pcd_file, 'r') as file:
            for line in file:
                line_arr = line.split(" ")
                try:
                    line_float_arr = [float(x) for x in line_arr]
                    # print(int(line_float_arr[-2]))
                    label_idx = int(line_float_arr[-2])
                    labels.append(bag_labels[label_idx])
                except:
                    pass

        points_and_noprmal_concat = np.hstack((points, normals))
        res = np.column_stack((points_and_noprmal_concat, np.array(labels)))
        # print(res.shape)
        num_samples = 2048
        sampled_idx = np.random.choice(res.shape[0], num_samples, replace=False)
        sampled_result = res[sampled_idx,:]

        with open(f"./{DATA_PATH_DIR}/{os.path.basename(pcd_file).split('.')[0]}.txt", 'w') as file:
            for row in sampled_result:
                file.write(' '.join(map(str, row)) + '\n')
    
def welding_seam_train():
    cfg = EasyConfig()
    # 加载配置文件
    cfg.load(CONFIG_PATH, recursive=True)
    # 设置随机数
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)


    cfg.epochs = 10

    
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1
    cfg.task_name = CONFIG_PATH.split('.')[-2].split('/')[-2]
    cfg.cfg_basename = CONFIG_PATH.split('.')[-2].split('/')[-1]

    print(f"task_name = {cfg.task_name}")
    print(f"basename = {cfg.cfg_basename}")

    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    
    # pprint.pprint(tags)


    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']

    if cfg.mode in ['resume', 'test', 'val']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        # cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        # cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (CONFIG_PATH, cfg.run_dir))
    
    cfg.cfg_path = cfg_path
    print(f"run_name = {cfg.run_name}")
    # wandb config
    # cfg.wandb.name = cfg.run_name

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)

def replace_file(source_path, target_path):
    """
    Replace the file at target_path with the file at source_path.
    
    Args:
    source_path (str): The path to the source file.
    target_path (str): The path to the target file where the source file should be moved.
    """
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        shutil.copy2(source_path, target_path)
        print(f"File from {source_path} has been successfully replaced to {target_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    folder_data = "pcd_data"
    obj_folder_data = "obj_data"

    # obj2txt(obj_folder_data)
    welding_seam_train_prepare(folder_data)

    update_train_test_json()
    welding_seam_train()