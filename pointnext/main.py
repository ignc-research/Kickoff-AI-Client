import os, argparse, yaml, numpy as np
import shutil

import torch,torch.nn
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import open3d as o3d
import time
from openpoints.utils import EasyConfig, dist_utils, generate_exp_directory, resume_exp_directory
from openpoints.models import build_model_from_cfg
from openpoints.dataset.mydataset.tools_dataset import pc_normalize,farthest_point_sampling

CURRENT_PATH = os.path.abspath(__file__)
BASE_1 = os.path.dirname(CURRENT_PATH)
ROOT = os.path.dirname(BASE_1)

bs = 1
checkpoint_dir = os.path.join(BASE_1, 'sim_checkpoint/best_model.pth')

parser = argparse.ArgumentParser('S3DIS scene segmentation training')
parser.add_argument('--cfg', type=str, default=os.path.join(BASE_1,'cfgs/modelnet40ply2048/pointnext-s.yaml'), help='config file')
parser.add_argument('--input_num', type=int, default=2048, help='config file')  # tfj add
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
opt_list = []  # for checking experiment configs from logging file
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
cfg.distributed = False
cfg.mp = False
cfg.input_num = args.input_num



def pointnext(file_path,SNahts,tree,xml_path,slice_name_list):
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
   
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    
    
    sig = torch.nn.Sigmoid()
    # optimizer & scheduler
    pc_list = slice_name_list
    checkpoint = torch.load(checkpoint_dir)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])


    name_id = {}
    for Snaht in SNahts:
        Name = Snaht.attrib.get('Name')
        ID = Snaht.attrib.get('ID')
        name_id[Name] = ID
    # print(name_id)

    all_datas, all_names = process_pc(file_path, pc_list)
    retrieved_map_id = {}
    retrieved_map_name = {}
    tic = time.time()

    with torch.no_grad():
        for query_pc in pc_list:
            if query_pc.split('.')[0] not in name_id:
                continue
            similar_list=[]
            similar_list_name=[]
            query_id = all_names.index(query_pc)
            query_data = all_datas[query_id]
            query_data = torch.from_numpy(query_data)
            query_data = query_data.float().cuda()
            # query_data = query_data.transpose(2, 1)
            all_sim = []
            for pc_2 in pc_list:
                compare_id  =  all_names.index(pc_2)
                compare_data = all_datas[compare_id]
                compare_data = torch.from_numpy(compare_data)[None,...]
                compare_data = torch.tensor(compare_data).squeeze()
                compare_data = compare_data.view(1,-1,3)
                compare_data = compare_data.float().cuda()
                query_indata = torch.repeat_interleave(query_data,len(compare_data),dim=0)
                pc_sim = sig(model(query_indata,compare_data))
                # print('one calculation time:',toc_1-tic_1)
                all_sim.extend(pc_sim.cpu().numpy().reshape(-1))
            st = np.argsort(all_sim)[::-1]
            for s in st:
                if all_sim[s]<0.95:
                    continue
                if all_names[s] == query_pc:
                    continue
                similar_list.append(name_id[all_names[s].split('.')[0]])
                similar_list_name.append(all_names[s].split('.')[0])
                string = 'slices: '+all_names[s]+', similarity: {}'.format(all_sim[s])
                # print(string)
            print('query slices:{}'.format(query_pc.split('.')[0])+', similarity: {}'.format(similar_list))
            retrieved_map_id[name_id[query_pc.split('.')[0]]] = similar_list
            retrieved_map_name[query_pc.split('.')[0]]=similar_list_name
            # print(query_pc+' finished!')

        for SNaht in SNahts:
            attr_dict={}
            for key, value in SNaht.attrib.items():
                if key == 'ID':
                    if value in retrieved_map_id:
                        # print(retrieved_map_id[value])
                        attr_dict[key] = value
                        attr_dict['Naht_ID'] = ','.join(retrieved_map_id[value])
                    else:
                        continue
                elif key == 'Naht_ID':
                    continue
                else:
                    attr_dict[key] = value
            SNaht.attrib.clear()
            for key, value in attr_dict.items():
                SNaht.set(key, value)
        # tree.write(xml_path)
        if os.path.exists('./log'):
            shutil.rmtree('./log')

    return retrieved_map_id,retrieved_map_name,tree


def process_pc(query_pcdir,pcs):
    
    datas = []
    names = []
    for pc in pcs:
        if pc.endswith('pcd'):
            tmp_name = os.path.join(query_pcdir,pc)
            pcd=o3d.io.read_point_cloud(tmp_name)
            input=np.asarray(pcd.points)
            
            lens = len(input)
            if lens==0:
                continue
            if lens < 2048:
                ratio = int(2048 /lens + 1)
                tmp_input = np.tile(input, (ratio, 1))
                input = tmp_input[:2048 ]
            
            if lens > 2048 :
                input = farthest_point_sampling(input,2048)
                
            input = pc_normalize(input)
            datas.append(input[None,...])
            names.append(pc) 

    return datas,names


# if __name__ == "__main__":
    # main(0,cfg,file_path,SNahts,tree,xml_path)
