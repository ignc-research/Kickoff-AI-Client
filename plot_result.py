import random
import xml.etree.ElementTree as ET
import json
# import open3d.core as o3c
import matplotlib.pyplot as plt
from pointnn.save_pn_feature import save_feature
from pointnn.cossim import pointnn
from pointnet2.main import pointnet2
from pointnext.main import pointnext
from ICP_RMSE import ICP
import os.path
from util import npy2pcd
from tools import get_ground_truth,get_weld_info,WeldScene
from evaluation import mean_metric
import open3d as o3d
import numpy as np
import time
import torch
import shutil
from create_pc import split,convert

CURRENT_PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(CURRENT_PATH)
# ROOT = os.path.dirname(BASE)

def matching(data_folder,xml_file,auto_del=False):
    start_time=time.time()
    xml_path=os.path.join(ROOT,data_folder,xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    Baugruppe=root.attrib['Baugruppe']
    data_path = data_folder
    wz_path = os.path.join(data_path,Baugruppe)
    if not os.path.exists(os.path.join(data_path,Baugruppe+'.pcd')):
        split(data_path,Baugruppe)
        convert(data_path,40,Baugruppe)

    inter_time=time.time()
    # print('creating pointcloud time',inter_time-start_time)
    os.makedirs(wz_path,exist_ok=True)
    ws = WeldScene(os.path.join(data_path,Baugruppe+'.pcd'))
    weld_infos=get_weld_info(xml_path)
    gt_id_map, gt_name_map = get_ground_truth(weld_infos)
    weld_infos=np.vstack(weld_infos)
    SNahts_list = root.findall("SNaht")
    for SNaht in SNahts_list:
        slice_name = SNaht.attrib['Name']
        # if os.path.exists(os.path.join(wz_path,slice_name+'.pcd'))==False:
        weld_info=weld_infos[weld_infos[:,0]==slice_name][:,3:].astype(float)
        if len(weld_info)==0:
            continue
        cxy, cpc, new_weld_info = ws.crop(weld_info=weld_info, num_points=2048)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(cxy)
        o3d.io.write_point_cloud(os.path.join(wz_path, slice_name + '.pcd'), pointcloud=pc, write_ascii=True)
    time_dict={}
    metric_dict={}
    for i in range(39,200,40):
        time_list=[]
        metric_list=[]
        SNahts=random.sample(SNahts_list,i)
        slice_name_list = []
        for SNaht in SNahts:
            slice_name = SNaht.attrib['Name']
            slice_name_list.append(slice_name + '.pcd')
        models=['pointnext']
        for model in models:
            retrieved_map = {}
            gt_copy=gt_id_map.copy()
            method_time=time.time()
            if model == 'icp':
                print('run icp')
                retrieved_map=ICP(SNahts,wz_path,tree,xml_path)

            elif model == 'pointnn':
                print('run pointnn')
                save_feature(wz_path,slice_name_list)
                retrieved_map=pointnn(SNahts,tree,xml_path)

            elif model == 'pointnet2':
                print('run pointnet2')
                retrieved_map=pointnet2(wz_path,SNahts,tree,xml_path,slice_name_list)

            elif model == 'pointnext':
                print('run pointnext')
                retrieved_map=pointnext(wz_path,SNahts,tree,xml_path,slice_name_list)

            for key,value in gt_id_map.items():
                if key not in retrieved_map:
                    del gt_copy[key]
            # print('gt_map',gt_copy)
            # print('retrieved_map',retrieved_map)
            metric=mean_metric(gt_copy,retrieved_map)
            print('metric',metric)
            end_time = time.time()
            total_time = end_time - method_time
            time_list.append(int(total_time))
            metric_list.append(int(metric))
            metric_dict[str(i)]=metric_list
            time_dict[str(i)]=time_list
            print('sample number ',i,' method time=', end_time - method_time)
    if auto_del:
        shutil.rmtree(wz_path)

    return metric_dict,time_dict

if __name__ == "__main__":
    metric_dict,time_dict=matching(os.path.join(ROOT, 'data'),'Reisch_origin.xml')
    print(metric_dict)
    print(time_dict)
    with open('metric.json', 'w') as m:
        json.dump(metric_dict, m, indent=4)
    with open('time.json', 'w') as t:
        json.dump(time_dict, t, indent=4)

    icp_time_list = []
    icp_metric_list = []
    pointnn_time_list = []
    pointnn_metric_list = []
    pointnet2_time_list = []
    pointnet2_metric_list = []
    pointnext_time_list = []
    pointnext_metric_list = []
    x=[]
    for key,val in time_dict.items():
        x.append(int(key))
        icp_time_list.append(val[0])
        pointnn_time_list.append(val[1])
        pointnet2_time_list.append(val[2])
        pointnext_time_list.append(val[3])

    for key,val in metric_dict.items():
        icp_metric_list.append(val[0])
        pointnn_metric_list.append(val[1])
        pointnet2_metric_list.append(val[2])
        pointnext_metric_list.append(val[3])
    # x=[39,79,119,159,199]
    # icp_time_list=[12,54,122,239,375]
    # pointnn_time_list=[1,1,1,1,1]
    # pointnet2_time_list=[167,629,1543,2723,4246]
    # pointnext_time_list=[15,63,135,244,382]
    #
    # icp_metric_list=[0.487,0.443,0.429,0.527,0.581]
    # pointnn_metric_list=[0.308,0.424,0.381,0.445,0.488]
    # pointnet2_metric_list=[0.547,0.594,0.582,0.642,0.727]
    # pointnext_metric_list=[0.564,0.474,0.524,0.519,0.514]
    #
    #
    #
    x_ticks_label=[40,80,120,160,200]
    fig=plt.figure(figsize=(10,10))

    plt.subplot(2,1,1)
    plt.xlabel('amount of slices')
    plt.ylabel('mAP')
    plt.plot(x, icp_metric_list,marker='.',markersize=3)
    plt.plot(x, pointnn_metric_list,marker='.',markersize=3)
    plt.plot(x, pointnet2_metric_list,marker='.',markersize=3)
    plt.plot(x, pointnext_metric_list,marker='.',markersize=3)
    plt.title('slices-metric')
    plt.xticks(x_ticks_label)
    plt.legend(['icp','pointnn','pointnet2','pointnext'])

    plt.subplot(2,1,2)
    plt.xlabel('amount of slices')
    plt.ylabel('time')
    plt.plot(x, icp_time_list,marker='.',markersize=3)
    plt.plot(x, pointnn_time_list,marker='.',markersize=3)
    plt.plot(x, pointnet2_time_list,marker='.',markersize=3)
    plt.plot(x, pointnext_time_list,marker='.',markersize=3)
    plt.title('slices-time')
    plt.xticks(x_ticks_label)
    plt.legend(['icp','pointnn','pointnet2','pointnext'])

    plt.savefig('./figure.jpg',bbox_inches='tight')
    plt.show()

