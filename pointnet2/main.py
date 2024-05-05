"""
Author: Benny
Date: Nov 2019
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch.nn as nn
import sys
import torch
import importlib
import open3d as o3d
from pointnet2.tools import *
import time
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
def get_distance(SNaht):
    list1 = []
    for punkt in SNaht.iter('Punkt'):
        list2 = []
        list2.append(float(punkt.attrib['X']))
        list2.append(float(punkt.attrib['Y']))
        list2.append(float(punkt.attrib['Z']))
        list1.append(list2)
    weld_info=np.asarray(list1)
    seam_vector=weld_info[-1,:]-weld_info[0,:]
    x_diff = np.max(weld_info[:, 0]) - np.min(weld_info[:, 0])
    if x_diff < 2:
        x_diff = 0
    y_diff = np.max(weld_info[:, 1]) - np.min(weld_info[:, 1])
    if y_diff < 2:
        y_diff = 0
    z_diff = np.max(weld_info[:, 2]) - np.min(weld_info[:, 2])
    if z_diff < 2:
        z_diff = 0
    distance = int(pow(pow(x_diff, 2) + pow(y_diff, 2) + pow(z_diff, 2), 0.5)) + 25
    return distance,seam_vector.astype(int)

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg_siamese', help='model name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch Size during training')
    parser.add_argument('--epoch', default=100, type=int, help='epoch to run')
    parser.add_argument('--dataset', default='data', type=str, help='pu1k or pugan')
    parser.add_argument('--input_num', default=2048, type=str, help='optimizer, adam or sgd')
    parser.add_argument('--file_path', default=os.path.join(ROOT_DIR,'data','Aehn3Test_welding_zone'), help='model name')
    parser.add_argument('--model_path', default=os.path.join(BASE_DIR,'checkpoints','model_100.pth'), help='model name')

    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def process_pc(file_path,pcs):
    datas = []
    names = []
    for pc in pcs:
        if pc.endswith('pcd'):
            tmp_name = os.path.join(file_path,pc)
            pcd=o3d.io.read_point_cloud(tmp_name)
            input=np.asarray(pcd.points)

            lens = len(input)

            if lens < 2048:
                ratio = int(2048 /lens + 1)
                tmp_input = np.tile(input, (ratio, 1))
                input = tmp_input[:2048 ]

            if lens > 2048 :
                np.random.shuffle(input)
                input = farthest_point_sampling(input,2048)

            datas.append(input)
            names.append(pc)

    return datas,names



def pointnet2(file_path,SNahts,tree,xml_path,slice_name_list):
    args = parse_args()

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)

    siamese_model = MODEL.get_model().cuda()
    # criterion = MODEL.get_loss().cuda()
    siamese_model.apply(inplace_relu)

    sig = nn.Sigmoid()

    checkpoint = torch.load(args.model_path)
    start_epoch = checkpoint['epoch']
    siamese_model.load_state_dict(checkpoint['model_state_dict'])

    name_id = {}
    for Snaht in SNahts:
        Name = Snaht.attrib.get('Name')
        ID = Snaht.attrib.get('ID')
        name_id[Name] = ID
    pc_list = slice_name_list
    all_datas, all_names = process_pc(file_path, pc_list)
    retrieved_map = {}
    retrieved_map_name = {}
    # query_pcs = ['PgmDef_260_0.pcd']
    tic = time.time()

    with torch.no_grad():
        for pc_1 in pc_list:
            similar_list=[]
            similar_list_name = []
            query_id = all_names.index(pc_1)
            query_data = all_datas[query_id]
            query_data = pc_normalize(query_data)
            query_data = torch.from_numpy(query_data)[None, ...]
            query_data = query_data.float().cuda()
            query_data = query_data.transpose(2, 1)
            all_sim = []
            cat_data = query_data
            for pc_2 in pc_list:
                compare_id  =  all_names.index(pc_2)
                compare_data = all_datas[compare_id]
                compare_data = pc_normalize(compare_data)
                compare_data = torch.from_numpy(compare_data)[None,...]
                compare_data = compare_data.float().cuda()
                compare_data = compare_data.transpose(2, 1)
                cat_data=torch.cat([cat_data,compare_data],0)
            pc_sim = siamese_model(query_data, cat_data,training=False)
            for i in range(len(pc_sim)):
                score = sig(pc_sim[i])
                all_sim.append(score.item())
            st = np.argsort(all_sim)[::-1]
            for s in st:
                if all_sim[s]<0.95:
                    continue
                if all_names[s] == pc_1:
                    continue
                similar_list.append(name_id[all_names[s].split('.')[0]])
                similar_list_name.append(all_names[s].split('.')[0])
                string = '点云: ' + all_names[s] + ', 相似度: {}'.format(all_sim[s])
            print('query slices:{}'.format(pc_1.split('.')[0]) + ', similarity: {}'.format(similar_list))
            retrieved_map[name_id[pc_1.split('.')[0]]]=similar_list
            retrieved_map_name[pc_1.split('.')[0]] = similar_list_name
        for SNaht in SNahts:
            attr_dict={}
            for key, value in SNaht.attrib.items():
                if key == 'ID':
                    if value in retrieved_map:
                        attr_dict[key] = value
                        attr_dict['Naht_ID'] = ','.join(retrieved_map[value])
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
    return retrieved_map,retrieved_map_name,tree
if __name__ == '__main__':
    pointnet2()
