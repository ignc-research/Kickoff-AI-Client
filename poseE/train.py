import torch
import open3d as o3d
import numpy as np
import os
from torch.utils.data import Dataset
from poseE.network import PointCloudNet
import xml.etree.ElementTree as ET
import torch.optim as optim
import torch.nn as nn
import json

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_PATH)
checkpoints_dir = os.path.join(CURRENT_PATH,'checkpoints')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dir = os.path.join(ROOT,'data/torch')
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


def find_xml_files(directory):
    xml_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))
    return xml_files


def get_json(xml_file_path):
    print('xml_file_path',xml_file_path)
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    data_separated = []
    for snaht in root.findall('.//SNaht'):
        point_cloud_file_name = snaht.attrib['Name']
        torch_name=snaht.attrib['WkzName']
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
                "point_cloud_file_name": os.path.join(xml_file_path.split('.')[0],
                                                      (point_cloud_file_name + '.pcd')),
                "weld_position": weld_position,
                "rotation_matrix": rotation_matrix,
                "torch_name": torch_name
            })
    print(data_separated[:4])
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


class PointCloudDataset(Dataset):
    def __init__(self, json_data):
        self.data = json_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        point_cloud_file_name = item['point_cloud_file_name']
        print('point_cloud_file_name',point_cloud_file_name)
        weld_position = torch.tensor(item['weld_position'], dtype=torch.float32)
        rotation_matrix = torch.tensor(item['rotation_matrix'], dtype=torch.float32)
        torch_name = item['torch_name']
        pcd = o3d.io.read_point_cloud(point_cloud_file_name)

        torch_pcd = read_obj(os.path.join(torch_dir, torch_name + '.obj'))
        torch_pcd = torch.tensor(torch_pcd, dtype=torch.float32).cuda()
        point_cloud = torch.tensor(pcd.points, dtype=torch.float32)

        return point_cloud.cuda(), weld_position.cuda(), rotation_matrix.cuda(),torch_pcd.cuda()


def load_dataset(directory_path):
    # directory_path = 'dataset'
    welding_gun_pcd = read_obj(os.path.join(ROOT,'data/torch','MRW510_10GH.obj'))
    welding_gun_pcd = torch.tensor(welding_gun_pcd, dtype=torch.float32).to(device)
    xml_files = find_xml_files(directory_path)

    result_json = []

    for path in xml_files:
        result_json += get_json(path)
    dataset = PointCloudDataset(result_json)
    return dataset,welding_gun_pcd

def training(dataset_path):
    model = PointCloudNet().to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset,welding_gun_pcd=load_dataset(dataset_path)
    epochs = 80
    welding_gun_pcd = welding_gun_pcd.cuda()
    for epoch in range(epochs):
        for i, data in enumerate(dataset):
            point_cloud, weld_position, true_rotation_matrix,torch_pcd = data
            # print(point_cloud,'\n',weld_position,'\n',true_rotation_matrix,'\n')
            point_cloud = point_cloud.cuda()
            weld_position = weld_position.cuda()
            true_rotation_matrix = true_rotation_matrix.cuda()
            optimizer.zero_grad()
            predicted_euler_angles = model(point_cloud.unsqueeze(0), weld_position.unsqueeze(0),
                                           torch_pcd.unsqueeze(0))
            true_euler_angles = rotation_matrix_to_euler_angles(true_rotation_matrix).cuda()
            loss = loss_function(predicted_euler_angles, true_euler_angles)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataset)}], Loss: {loss.item()}")

        if (epoch%5 == 0):
            savepath = os.path.join(checkpoints_dir,'model_{}.pth'.format(epoch))#str(checkpoints_dir) + 'model_{}.pth'.format(epoch)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)