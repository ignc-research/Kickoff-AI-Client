import __init__
import os, argparse, yaml, numpy as np
from torch import multiprocessing as mp
from pointnext.classification.train import main as train
from pointnext.classification.mytrain import main as mytrain
from pointnext.classification.pretrain import main as pretrain
from openpoints.utils import EasyConfig, dist_utils, find_free_port, generate_exp_directory, resume_exp_directory, Wandb

CURRENT_PATH = os.path.abspath(__file__)
BASE_1 = os.path.dirname(CURRENT_PATH)
ROOT = os.path.dirname(BASE_1)
ROOT2= os.path.dirname(ROOT)
if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, default=os.path.join(ROOT,'cfgs/modelnet40ply2048/pointnext-s.yaml'), help='config file')
    parser.add_argument('--input_num', type=int, default=2048, help='config file') # tfj add
    parser.add_argument('--file_path', type=str, default='data2', help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.exp_name = args.cfg.split('.')[-2].split('/')[-1]
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.exp_name,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:  # resume from the existing ckpt and reuse the folder.
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path
    cfg.wandb.name = cfg.run_name

    if cfg.mode == 'pretrain':
        main = pretrain
    else:
        main = train
    main = mytrain  # my add
    cfg.distributed = False 
    cfg.mp = False 
    cfg.input_num = args.input_num
    cfg.file_path = os.path.join(ROOT2,args.file_path)
    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg, args.profile))
    else:
        main(0, cfg, profile=args.profile)
