import argparse
import datetime
import logging
import shutil

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from extensions.chamfer_dist import ChamferDistanceL2
import pointpn.models as models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default=os.path.join(BASE_DIR,'checkpoint'), help='path to save checkpoint (default: ckpt)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--model', default='Point_PN_AE', help='model name')
    parser.add_argument('--model_path', default=os.path.join(BASE_DIR,'checkpoint','Point_PN_AE-20230827171620-6212','checkpoint_290.pth'), help='model name')
    parser.add_argument('--input_num', type=int, default=2048, help='point number')

    parser.add_argument('--seed', type=int, default=6212, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='workers')

    return parser.parse_args()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sampling(points, num_samples):
    num_points = points.shape[0]
    sampled_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.full(num_points, np.inf)


    start_index = np.random.randint(num_points)
    sampled_indices[0] = start_index

    for i in range(1, num_samples):
        last_sampled_index = sampled_indices[i - 1]
        last_sampled_point = points[last_sampled_index]


        dist_to_last_sampled = np.linalg.norm(points - last_sampled_point, axis=1)


        distances = np.minimum(distances, dist_to_last_sampled)


        next_sampled_index = np.argmax(distances)
        sampled_indices[i] = next_sampled_index

    return points[sampled_indices]



def process_data(args,file_path,slice_name_list):
    import open3d as o3d
    datas = []
    names = []
    print('loading data ~~~~')

    load_data_path = os.path.join(file_path, 'pc_{}pts.npz'.format(args.input_num))
    if os.path.exists(load_data_path):  # 加载已有数据
        tdatas = np.load(load_data_path, allow_pickle=True)['data']
        tnames = np.load(load_data_path, allow_pickle=True)['name']
        for iii in range(len(tdatas)):
            datas.append(tdatas[iii])
            names.append(str(tnames[iii]))
    for silce in slice_name_list:
        tmp_name = os.path.join(file_path, silce)


    # for root, dirs, files in os.walk():
    #     for file in files:
    #         if file.endswith('pcd'):
    #             tmp_name = os.path.join(root, file)

        if tmp_name in names:
            continue

        pcd = o3d.io.read_point_cloud(tmp_name)
        input = np.asarray(pcd.points)

        lens = len(input)
        if lens==0:
            continue

        if lens < args.input_num:
            ratio = int(args.input_num / lens + 1)
            tmp_input = np.tile(input, (ratio, 1))
            input = tmp_input[:args.input_num]

        if lens > args.input_num:
            np.random.shuffle(input)
            input = farthest_point_sampling(input, args.input_num)

        datas.append(input)
        names.append(tmp_name)

    np.savez(load_data_path, data=datas, name=names)
    print('data lens: ', len(datas))
    return load_data_path

def save_feature(file_path,slice_name_list):
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
    args.ckpt_dir = args.ckpt_dir +'/'+ args.model + message + '-' + str(args.seed)
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


    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    features = []
    mynames = []
    with torch.no_grad():
        load_data_path=process_data(args,file_path,slice_name_list)
        path = os.path.join(file_path,'pc_{}pts.npz'.format(args.input_num))
        datas = np.load(path,allow_pickle=True)['data']
        names = np.load(path,allow_pickle=True)['name']
        for iii, data in enumerate(datas):

            input = pc_normalize(data)
            points = torch.from_numpy(input)[None,...]

            points = points.float().cuda()
            points = points.transpose(2, 1)

            rec_pc,global_feature = net(points)

            features.append(global_feature.cpu().detach().numpy())
            mynames.append(str(names[iii]))
        features_path=os.path.join(BASE_DIR,'cnn_feature','pnn_tpc_cnn_feature')
        np.savez(features_path,cnn_feature=features,name=mynames)
        print(load_data_path)
        if os.path.exists(load_data_path):
            os.remove(load_data_path)
        print(' saved cnn feature!')

if __name__ == '__main__':
    file_path=os.path.join(ROOT_DIR,'data','Aehn3Test_welding_zone')
    save_feature(file_path)
