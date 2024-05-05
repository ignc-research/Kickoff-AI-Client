import os
import numpy as np
import open3d as o3d
import time
import json
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

def ICP(SNahts,wz_path,tree,xml_path):
    retrieved_map = {}
    retrieved_map_name = {}
    for SNaht_src in SNahts:
        dict={}
        similar_str=''
        similar_list = []
        similar_list_name = []
        src_ID = SNaht_src.attrib.get('ID')
        src_name=SNaht_src.attrib.get('Name')
        src_path=wz_path + '/' + src_name + '.pcd'
        if os.path.exists(src_path)!=True:
            continue
        seam_length_src,seam_vec_src=get_distance(SNaht_src)
        pcd1=o3d.io.read_point_cloud(src_path)
        point1=np.array(pcd1.points).astype('float32')
        src=point1
        for SNaht_tgt in SNahts:
            tgt_ID = SNaht_tgt.attrib.get('ID')
            tgt_name=SNaht_tgt.attrib.get('Name')
            if src_name==tgt_name:
                continue
            tgt_path=wz_path + '/' + tgt_name + '.pcd'
            if os.path.exists(tgt_path)!=True:
                continue
            seam_length_tgt,seam_vec_tgt=get_distance(SNaht_tgt)
            pcd2 = o3d.io.read_point_cloud(tgt_path)
            point2 = np.array(pcd2.points).astype('float32')
            target = point2

            seam_vec_diff=seam_vec_src-seam_vec_tgt
            # if abs(seam_vec_diff[0])>3 or abs(seam_vec_diff[1])>3 or abs(seam_vec_diff[2])>3:
            #     continue
            # if abs(seam_length_src-seam_length_tgt)>5:
            #     continue

            centroid1 = np.mean(src, axis=0)
            src = src - centroid1
            m1 = np.max(np.sqrt(np.sum(src ** 2, axis=1)))
            src = src / m1

            centroid2 = np.mean(target, axis=0)
            target = target - centroid2
            m2 = np.max(np.sqrt(np.sum(target ** 2, axis=1)))
            target = target / m2

            src_cloud = o3d.geometry.PointCloud()
            src_cloud.points = o3d.utility.Vector3dVector(src)
            tgt_cloud = o3d.geometry.PointCloud()
            tgt_cloud.points = o3d.utility.Vector3dVector(target)
            tic=time.time()
            icp_s_t = o3d.pipelines.registration.registration_icp(source=src_cloud, target=tgt_cloud,
                                                                  max_correspondence_distance=0.2,
                                                                  estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
            # mean_distance_s_t = np.mean(src_cloud.compute_point_cloud_distance(tgt_cloud))
            fitness_s_t = icp_s_t.fitness
            rmse_s_t = icp_s_t.inlier_rmse
            correspondence_s_t = len(np.asarray(icp_s_t.correspondence_set))
            if rmse_s_t > 0.03 or correspondence_s_t < 1900:
                continue

            icp_t_s = o3d.pipelines.registration.registration_icp(source=tgt_cloud, target=src_cloud,
                                                                  max_correspondence_distance=0.2,
                                                                  estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

            # mean_distance_t_s = np.mean(tgt_cloud.compute_point_cloud_distance(src_cloud))
            toc=time.time()
            # print(toc-tic)
            fitness_t_s = icp_t_s.fitness
            rmse_t_s = icp_t_s.inlier_rmse
            correspondence_t_s = len(np.asarray(icp_t_s.correspondence_set))
            src_cloud.paint_uniform_color([1, 0, 0])
            tgt_cloud.paint_uniform_color([0, 1, 0])

            if rmse_t_s > 0.03 or correspondence_t_s < 1900:
                continue
            # o3d.visualization.draw_geometries([src_cloud, tgt_cloud], width=800)
            similar_list.append(tgt_ID)
            similar_list_name.append(tgt_name)
            if similar_str=='':
                similar_str+=tgt_ID
            else:
                similar_str += (','+tgt_ID)
        retrieved_map[src_ID] = similar_list
        retrieved_map_name[src_name]=similar_list_name
        for key,value in SNaht_src.attrib.items():
            if key=='Name':
                dict[key]=value
                dict['Naht_Name']=similar_str
            elif key=='Naht_Name':
                continue
            else:
                dict[key]=value
        SNaht_src.attrib.clear()
        for key,value in dict.items():
            SNaht_src.set(key,value)
    # tree.write(xml_path)

    return retrieved_map,retrieved_map_name,tree