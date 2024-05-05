"""
Author: Benny
Date: Nov 2019
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
import torch.nn as nn
import sys
import importlib

from main.tools import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


bs = 1 # 只支持1

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--data_dir', type=str, required=True, help='存储部分数据的地址')
    parser.add_argument('--query_pc', type=str, required=True, help='查询的点云名称')
    parser.add_argument('--part_id', type=int, required=True, help='当前id')
    
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg_siamese', help='model name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--dataset', default='data2', type=str, help='pu1k or pugan')
    parser.add_argument('--input_num', default=2048, type=str, help='optimizer, adam or sgd')
    parser.add_argument('--file_path', default='/data/tfj/workspace/python_projects/jianzhi/Pointnet_Pointnet2_pytorch/data2', help='model name')
    parser.add_argument('--model_path', default='log/siamese_tools/2023-09-12_15-26/checkpoints/best_model.pth', help='model name')
    
    parser.add_argument('--log_dir', type=str, default=None, help='log path')

    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True



def main(args):

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)

    siamese_model = MODEL.get_model().cuda()
    # criterion = MODEL.get_loss().cuda()
    siamese_model.apply(inplace_relu)

    sig = nn.Sigmoid()
    
    checkpoint = torch.load(args.model_path)
    start_epoch = checkpoint['epoch']
    siamese_model.load_state_dict(checkpoint['model_state_dict'])
    print('Use pretrain model')
  
    part_id = args.part_id
    os.makedirs('./res_sim',exist_ok=True)
    with torch.no_grad():
        part_data = np.load(args.data_dir,allow_pickle=True)
        all_datas,all_names = part_data['data'],part_data['name']
        query_pc = np.load(args.query_pc+'.npz')['data']
        query_data = torch.from_numpy(query_pc)
        query_data = query_data.float().cuda()
        query_data = query_data.transpose(2, 1)
        all_sim = []
        iii = 0
        input_data = all_datas[iii:iii+bs]
        while len(input_data) > 0:
            iii += bs
            compare_data = torch.tensor(input_data).squeeze()
            compare_data = compare_data.view(len(input_data),-1,3)
            compare_data = compare_data.float().cuda()
            compare_data = compare_data.transpose(2, 1)
            query_indata = torch.repeat_interleave(query_data,len(compare_data),dim=0)
            pc_sim = sig(siamese_model(query_indata,compare_data))
            all_sim.extend(pc_sim.cpu().numpy().reshape(-1))
            input_data = all_datas[iii:iii+bs]
        
        st = np.argsort(all_sim)[::-1]
        tmp_name = args.query_pc.split('/')[-1].split('.')[0]
        f = open('multiprocess_calc_res/'+tmp_name+'_part_'+str(part_id)+'.txt','w')
        f.write('query pc: '+tmp_name+'\n')
        print('query pc: '+tmp_name)
        for s in st:
            string = '点云: '+all_names[s]+', 相似度: {}'.format(all_sim[s])
            print(string)
            f.write(string+'\n')
        f.close()
        print(tmp_name+' finished!')
            
            
            


if __name__ == '__main__':
    args = parse_args()
    main(args)
