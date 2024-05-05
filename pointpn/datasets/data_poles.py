import os
import numpy as np
from torch.utils import data


class PolesDataset(data.Dataset):
    def __init__(self, args):
        super(PolesDataset, self).__init__()

        self.input_num = args.input_num
        self.args = args
        # input and gt: (b, n, 3) radius: (b, 1)
        self.poles,self.poles_size = [],[]
        npzs = os.listdir(args.npz_file_path)
        for npz in npzs:
            if npz.endswith('npz') and '03-17' not in npz:
                print('load ',npz)
                data = np.load(os.path.join(args.npz_file_path,npz),allow_pickle=True)
                desc = data['descriptors.npy']
                for des in desc:
                    for po in des['poles']:
                        if len(po) < 50: # 小于100个点的
                            continue       
                        self.poles.append(po)
                        self.poles_size.append(len(po))
           
        # self.input_data, self.gt_data, self.radius_data = load_h5_data(args)

    def __len__(self):
        return len(self.poles)

    def __getitem__(self, index):
        # (n, 3)
        input = self.poles[index]
        lens = self.poles_size[index]
        if lens < self.input_num :
            ratio = int(self.input_num /lens + 1)
            tmp_input = np.tile(input, (ratio, 1))
            input = tmp_input[:self.input_num ]
        
        if lens > self.input_num :
            np.random.shuffle(input) # 每次取不一样的1024个点
            input = farthest_point_sampling(input,self.input_num)
        
        input = pc_normalize(input)
        input = torch.from_numpy(input)
        
        return input