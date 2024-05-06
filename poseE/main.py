import torch
import torch.nn.functional as F
import open3d as o3d
from torch.utils.data import Dataset
import os
import numpy as np
import xml.etree.ElementTree as ET
import json
from copy import copy
import os
from tools import get_weld_info
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_PATH)
device = torch.device('cuda:0')

def euler_angles_to_rotation_matrix(angles):
    """ Convert Euler angles to a rotation matrix. Angles are assumed to be in ZYX order. """
    B = angles.shape[0]
    ones = torch.ones(B)
    zeros = torch.zeros(B)

    # Extracting individual angles
    yaw, pitch, roll = angles[:, 0], angles[:, 1], angles[:, 2]

    # Precomputing sine and cosine of angles
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cr = torch.cos(roll)
    sr = torch.sin(roll)

    # Constructing rotation matrix
    rotation_matrix = torch.zeros((B, 3, 3), device=angles.device)
    rotation_matrix[:, 0, 0] = cy * cp
    rotation_matrix[:, 0, 1] = cy * sp * sr - sy * cr
    rotation_matrix[:, 0, 2] = cy * sp * cr + sy * sr
    rotation_matrix[:, 1, 0] = sy * cp
    rotation_matrix[:, 1, 1] = sy * sp * sr + cy * cr
    rotation_matrix[:, 1, 2] = sy * sp * cr - cy * sr
    rotation_matrix[:, 2, 0] = -sp
    rotation_matrix[:, 2, 1] = cp * sr
    rotation_matrix[:, 2, 2] = cp * cr

    return rotation_matrix

def rotation_matrix_to_euler_angles(matrix):
    """ Assumes the order of the axes is ZYX """
    sy = torch.sqrt(matrix[0, 0] * matrix[0, 0] +  matrix[1, 0] * matrix[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = torch.atan2(matrix[2, 1], matrix[2, 2])
        y = torch.atan2(-matrix[2, 0], sy)
        z = torch.atan2(matrix[1, 0], matrix[0, 0])
    else:
        x = torch.atan2(-matrix[1, 2], matrix[1, 1])
        y = torch.atan2(-matrix[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z]).to(device)


def get_distance_and_translate(weld_info):
    x_center = (np.max(weld_info[:, 1]) + np.min(weld_info[:, 1])) / 2
    y_center = (np.max(weld_info[:, 2]) + np.min(weld_info[:, 2])) / 2
    z_center = (np.max(weld_info[:, 3]) + np.min(weld_info[:, 3])) / 2
    translate = np.array([x_center, y_center, z_center])
    x_diff = np.max(weld_info[:, 1]) - np.min(weld_info[:, 1])
    if x_diff < 2:
        x_diff = 0
    y_diff = np.max(weld_info[:, 2]) - np.min(weld_info[:, 2])
    if y_diff < 2:
        y_diff = 0
    z_diff = np.max(weld_info[:, 3]) - np.min(weld_info[:, 3])
    if z_diff < 2:
        z_diff = 0
    distance = int(pow(pow(x_diff, 2) + pow(y_diff, 2) + pow(z_diff, 2), 0.5)) + 50

    return distance, translate



def find_xml_files(directory):
    xml_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))
    return xml_files


def get_json(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    data_separated = []
    for snaht in root.findall('.//SNaht'):
        point_cloud_file_name = snaht.attrib['Name']
        for frame in snaht.findall('.//Frame'):
            pos = frame.find('Pos')
            weld_position = [float(pos.attrib['X']), float(pos.attrib['Y']), float(pos.attrib['Z'])]
            x_vek = frame.find('XVek')
            y_vek = frame.find('YVek')
            z_vek = frame.find('ZVek')
            rotation_matrix = [
                [float(x_vek.attrib['X']), float(y_vek.attrib['X']), float(z_vek.attrib['X'])],
                [float(x_vek.attrib['Y']), float(y_vek.attrib['Y']), float(z_vek.attrib['Y'])],
                [float(x_vek.attrib['Z']), float(y_vek.attrib['Z']), float(z_vek.attrib['Z'])]
            ]
            data_separated.append({
                "point_cloud_file_name": os.path.join('\\'.join(xml_file_path.split('\\')[:-1]),
                                                      (point_cloud_file_name + '.pcd')),
                "weld_position": weld_position,
                "rotation_matrix": rotation_matrix
            })
    print(data_separated[:3])
    return data_separated

def read_obj(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    vertices = []
    faces = []

    for line in lines:
        if line.startswith('v '):
            vertex = [float(value) for value in line.split()[1:]]
            vertices.append(vertex)
        elif line.startswith('f '):
            face = [int(value.split('/')[0]) for value in line.split()[1:]]
            faces.append(face)

    vertices = np.array(vertices)
    faces = np.array(faces)

    return vertices


import torch
from torch.utils.data import Dataset
import json

class PointCloudDataset(Dataset):
    def __init__(self, json_data, welding_gun_pcd):
        self.data = json_data
        self.welding_gun_pcd = welding_gun_pcd

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        point_cloud_file_name = item['point_cloud_file_name']
        weld_position = torch.tensor(item['weld_position'], dtype=torch.float32)
        rotation_matrix = torch.tensor(item['rotation_matrix'], dtype=torch.float32)

        pcd = o3d.io.read_point_cloud(point_cloud_file_name)
        point_cloud = torch.tensor(pcd.points, dtype=torch.float32)

        return point_cloud.cuda(), weld_position.cuda(), rotation_matrix.cuda()



from poseE.network import PointCloudNet

def poseestimation(data_path,wz_path,xml_path,SNahts,tree,map,vis=False):
    weld_infos = get_weld_info(xml_path)
    weld_infos = np.vstack(weld_infos)
    # print(weld_infos[0,:])
    # device = torch.device('cuda:1')
    model = PointCloudNet().cuda()
    # model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(CURRENT_PATH, 'checkpoints/best_model.pth')))
    # model=model.to(device)
    true_matrices = []
    predicted_matrices = []
    predict_rot_dict={}
    slice_rot_list=[]
    slice_rot_dict={}
    with torch.no_grad():
        coor1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])
        for key,vals in map.items():
            if key in slice_rot_dict:
                continue
            pcd = o3d.io.read_point_cloud(os.path.join(wz_path, key + '.pcd'))
            pcd.paint_uniform_color([1,0,0])
            point_cloud = torch.tensor(pcd.points, dtype=torch.float32).cuda()
            torch_name=weld_infos[weld_infos[:,0]==key][0,2]
            torch_name='MRW510_10GH'
            welding_gun_pcd = read_obj(os.path.join(data_path, 'torch', torch_name + '.obj'))
            welding_gun_pcd = torch.tensor(welding_gun_pcd, dtype=torch.float32).cuda()
            weld_info=weld_infos[weld_infos[:,0]==key][:,3:].astype(float)
            _, translate = get_distance_and_translate(weld_info)
            weld_seam = o3d.geometry.PointCloud()
            weld_seam.points = o3d.utility.Vector3dVector(weld_infos[weld_infos[:,0]==key][:,3:][:,1:4].astype(float))
            weld_seam.translate(-translate)
            for i in range(len(weld_info)):
                weld_spot_points=weld_info[i,1:4]
                weld_spot = o3d.geometry.PointCloud()
                weld_spot.points = o3d.utility.Vector3dVector(weld_spot_points.reshape((1,3)))
                weld_spot.translate(-translate)
                translate_weld_spot_points = np.array(weld_spot.points)
                pose_position = torch.tensor(translate_weld_spot_points.reshape(3, ), dtype=torch.float32).cuda()
                rotation_matrix=weld_info[i,14:23].reshape((3,3))
                rotation_matrix = torch.tensor(rotation_matrix.astype(float), dtype=torch.float32).cuda()

                # print(rotation_matrix)
                predicted_euler_angle = model(point_cloud.unsqueeze(0), pose_position.unsqueeze(0),
                                                  welding_gun_pcd.unsqueeze(0))
                predicted_rotation_matrix = euler_angles_to_rotation_matrix(predicted_euler_angle)
                # print('euler',predicted_euler_angle)
                # print('predict',predicted_rotation_matrix)

                if vis:
                    elements = []
                    # print(rotation_matrix)
                    # print(np.array(predicted_rotation_matrix.cpu())[0].T)
                    torch_model = o3d.io.read_triangle_mesh(os.path.join(data_path, 'torch', torch_name + '.obj'))
                    copy_torch = copy(torch_model)
                    GT_torch=copy(torch_model)
                    elements.append(pcd)
                    elements.append(coor1)
                    tf = np.zeros((4, 4))
                    tf[3, 3] = 1.0
                    tf[0:3,0:3]=np.transpose(np.array(predicted_rotation_matrix.cpu()).reshape(3,3))
                    tf[0:3,3]=translate_weld_spot_points

                    gt_tf = np.zeros((4, 4))
                    gt_tf[3, 3] = 1.0
                    gt_tf[0:3,0:3]=np.transpose(np.array(rotation_matrix.cpu()))
                    gt_tf[0:3,3]=translate_weld_spot_points
                    GT_torch.compute_vertex_normals()
                    GT_torch.paint_uniform_color([0, 0, 1])
                    GT_torch.transform(gt_tf)


                    # print('tf',tf)
                    # theta = np.pi*0.5
                    # rotation_matrix_kkk = np.array([[np.cos(theta), 0, np.sin(theta)],
                    #                             [0, 1, 0],
                    #                             [-np.sin(theta), 0, np.cos(theta)]])
                    # copy_torch.rotate(rotation_matrix_kkk, center=(0, 0, 0))

                    copy_torch.compute_vertex_normals()
                    copy_torch.paint_uniform_color([0,1,0])
                    copy_torch.transform(tf)




                    elements.append(GT_torch)
                    elements.append(copy_torch)
                    elements.append(weld_seam)
                    # print('rotation_matrix',rotation_matrix)
                    # print('predicted_rotation_matrix',predicted_rotation_matrix.squeeze(0))
                    print('mse',F.mse_loss(rotation_matrix,predicted_rotation_matrix.squeeze(0)))
                    # o3d.visualization.draw_geometries(elements)


                slice_rot_list.append(predicted_rotation_matrix.squeeze(0))
                predict_rot_dict[key] = predicted_rotation_matrix.squeeze(0)
                true_matrices.append(rotation_matrix)
                predicted_matrices.append(predicted_rotation_matrix.squeeze(0))
            slice_rot_dict[key]=slice_rot_list
            if len(vals)==0:
                continue
            for val in vals:
                if val in slice_rot_dict:
                    continue
                else:
                    slice_rot_dict[val]=slice_rot_list
        true_matrices = torch.stack(true_matrices)
        predicted_matrices = torch.stack(predicted_matrices)
        mse = torch.mean((true_matrices - predicted_matrices) ** 2)
        print(f"Mean Squared Error: {mse}")
        print('slice_rot_dict',slice_rot_dict)
        for SNaht in SNahts:
            slice_name=SNaht.attrib.get('Name')
            rotation_matrix_list=slice_rot_dict[slice_name]
            # print('rotation_matrix_list',rotation_matrix_list,rotation_matrix_list)
            for Frames in SNaht.findall('Frames'):
                Pose_num=len(rotation_matrix_list)
                j=0
                if j>Pose_num:
                    break
                for Frame in Frames.findall('Frame'):
                    for XVek in Frame.findall('XVek'):
                        XVek.set('X', str(np.array(rotation_matrix_list[j][0, 0].cpu())))
                        XVek.set('Y', str(np.array(rotation_matrix_list[j][1, 0].cpu())))
                        XVek.set('Z', str(np.array(rotation_matrix_list[j][2, 0].cpu())))
                    for YVek in Frame.findall('YVek'):
                        YVek.set('X', str(np.array(rotation_matrix_list[j][0, 1].cpu())))
                        YVek.set('Y', str(np.array(rotation_matrix_list[j][1, 1].cpu())))
                        YVek.set('Z', str(np.array(rotation_matrix_list[j][2, 1].cpu())))
                    for ZVek in Frame.findall('YVek'):
                        ZVek.set('X', str(np.array(rotation_matrix_list[j][0, 2].cpu())))
                        ZVek.set('Y', str(np.array(rotation_matrix_list[j][1, 2].cpu())))
                        ZVek.set('Z', str(np.array(rotation_matrix_list[j][2, 2].cpu())))
                j+=1
    return tree

