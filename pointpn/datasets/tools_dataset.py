import os
import torch
import numpy as np
import open3d as o3d
import torch.utils.data as data

def farthest_point_sampling(points, num_samples):
    num_points = points.shape[0]
    sampled_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.full(num_points, np.inf)

    # 随机选择一个起始点
    start_index = np.random.randint(num_points)
    sampled_indices[0] = start_index

    for i in range(1, num_samples):
        last_sampled_index = sampled_indices[i - 1]
        last_sampled_point = points[last_sampled_index]

        # 计算每个点到最后一个采样点的距离
        dist_to_last_sampled = np.linalg.norm(points - last_sampled_point, axis=1)

        # 更新距离数组，保留最小距离
        distances = np.minimum(distances, dist_to_last_sampled)

        # 选择最远的点作为下一个采样点
        next_sampled_index = np.argmax(distances)
        sampled_indices[i] = next_sampled_index

    return points[sampled_indices]

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ToolsDataset(data.Dataset):
    def __init__(self, args):
        super(ToolsDataset, self).__init__()

        self.input_num = args.input_num
        self.args = args
        # input and gt: (b, n, 3) radius: (b, 1)
        self.datas = []
        self.names = []
        print('loading ~~~~')
        
        load_data_path = os.path.join(args.file_path,'pc_{}pts.npz'.format(self.input_num))
        if os.path.exists(load_data_path): # 加载已有数据
            self.tdatas = np.load(load_data_path,allow_pickle=True)['data']
            self.tnames = np.load(load_data_path,allow_pickle=True)['name']
            for iii in range(len(self.tdatas)):
                self.datas.append(self.tdatas[iii])
                self.names.append(str(self.tnames[iii]))
            
        for root, dirs, files in os.walk(args.file_path):
            for file in files:
                if file.endswith('pcd'):
                    tmp_name = os.path.join(root,file)
                    
                    if tmp_name in self.names:
                        continue
                    
                    pcd=o3d.io.read_point_cloud(tmp_name)#路径需要根据实际情况设置
                    input=np.asarray(pcd.points)#A已经变成n*3的矩阵
                    
                    lens = len(input)
        
                    if lens < self.input_num :
                        ratio = int(self.input_num /lens + 1)
                        tmp_input = np.tile(input, (ratio, 1))
                        input = tmp_input[:self.input_num ]
                    
                    if lens > self.input_num :
                        np.random.shuffle(input) # 每次取不一样的1024个点
                        input = farthest_point_sampling(input,self.input_num)
                        
                    self.datas.append(input)
                    self.names.append(tmp_name) 
                    
        np.savez(load_data_path,data=self.datas,name=self.names)
    
        print('data lens: ',len(self.datas))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # (n, 3)
        input = self.datas[index]
        
        # pcd=o3d.io.read_point_cloud(input)#路径需要根据实际情况设置
        # input=np.asarray(pcd.points)#A已经变成n*3的矩阵
        lens = len(input)
          
        # if lens < self.input_num :
        #     ratio = int(self.input_num /lens + 1)
        #     tmp_input = np.tile(input, (ratio, 1))
        #     input = tmp_input[:self.input_num ]
        
        # if lens > self.input_num :
        #     np.random.shuffle(input) # 每次取不一样的1024个点
        #     input = farthest_point_sampling(input,self.input_num)
        
        input = pc_normalize(input)
        input = torch.from_numpy(input)
        
        return input