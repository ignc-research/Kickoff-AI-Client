import os
import subprocess
import open3d as o3d
import numpy as np
from main.tools import pc_normalize,farthest_point_sampling

def process_pc(query_pcdir,pcs):
    
    datas = []
    names = []
    for pc in pcs:
        if pc.endswith('pcd'):
            tmp_name = os.path.join(query_pcdir,pc)
            pcd=o3d.io.read_point_cloud(tmp_name)#路径需要根据实际情况设置
            input=np.asarray(pcd.points)#A已经变成n*3的矩阵
            
            lens = len(input)

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


num_thread = 5   # 一个程序大约占2G显存，比如你的显卡显存12G，则num_thread=6
query_pcdir = 'Aehn3Test_welding_zone'
query_pc = 'AutoDetect_25_26.pcd'  # 需要查询的点云名称
pcs = os.listdir(query_pcdir)
all_datas,all_names = process_pc(query_pcdir,pcs)

query_id = all_names.index(query_pc)
query_data = all_datas[query_id]
np.savez('./multiprocess_calc_res/'+query_pc,data=query_data)
os.makedirs('./multiprocess_calc_res',exist_ok=True)

num_part = int(np.ceil(len(all_datas)/num_thread))
for i in range(num_thread):
    save_name = './multiprocess_calc_res/'+'part'+str(i)+'data'
    cdata = all_datas[i*num_part:(i+1)*num_part]
    cname = all_names[i*num_part:(i+1)*num_part]
    np.savez(save_name,data=cdata,name=cname)
np.savez(save_name,data=cdata,name=cname)

processes = []
# 启动四个子进程来执行指令
for cmd in range(num_thread):
    ss = 'python multicalculate_pc_sim.py --data_dir '+'multiprocess_calc_res/part'+str(cmd)+'data.npz '+'--query_pc multiprocess_calc_res/'+query_pc \
            +' --part_id '+str(cmd)
    process = subprocess.Popen(ss, shell=True)
    processes.append(process)

# 等待所有子进程完成
for process in processes:
    process.wait()

# 合并数据
tmp_name = query_pc.split('/')[-1].split('.')[0]
out = []
for i in range(num_thread):
    txt = 'multiprocess_calc_res/'+tmp_name+'_part_'+str(i)+'.txt'
    res = open(txt).read().splitlines()
    out.extend(res)

datas =[]
for item in out:
    if 'query' in item:
        datas.append(-1)
        continue
    data = float(item.split('相似度:')[-1])
    datas.append(data)
order = np.argsort(datas)[::-1]
final_res = np.array(out)[order].tolist()
f = open('multiprocess_calc_res/res_'+tmp_name+'.txt','w')
f.write(final_res[-1]+'\n')
for j in final_res:
    f.write(j+'\n')
f.close()