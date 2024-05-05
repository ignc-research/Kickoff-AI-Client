from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import open3d as o3d

tmp = np.load('./pcd_features_k100.npz',allow_pickle=True)
features = tmp['feature']
names = tmp['name']

query_name = 'Aehn3Test/AutoDetect_11_15.pcd'
for ii,nn in enumerate(names):
    if query_name in nn:
        query_id = ii
        break
query_pcf = features[query_id]
query_pcn = names[query_id]

# sims = []
# for idx in range(len(features)):
#     if idx!=query_id:
#         cur_feature = features[idx]
#         similarity = cosine_similarity([cur_feature], query_pcf)
#         sims.append(similarity)
#     else:
#         sims.append(9999)

similarity = cosine_similarity(features,np.reshape(query_pcf,(1,-1))).reshape(-1)

sorted_idx = np.argsort(similarity)[::-1]

for idx in sorted_idx:
    cname = str(names[idx])
    print(cname)
    # cname = cname.replace('/data/tfj/workspace/python_projects/jianzhi/Pointnet_Pointnet2_pytorch','/Users/tfj/Downloads')
    # if cname.split('/')[-1] in valid:
    #     print(idx)
    if idx == query_id:
        print(cname+' query pcd! '+str(idx)+':'+str(similarity[idx]))
        pcd = o3d.io.read_point_cloud(cname)
        # o3d.visualization.draw_geometries_with_editing([pcd], window_name="Open3D", width=800, height=600)
    else:
        print(cname+' sim pcd! '+str(idx)+':'+str(similarity[idx]))
        pcd = o3d.io.read_point_cloud( cname)
        # o3d.visualization.draw_geometries_with_editing([pcd], window_name="Open3D", width=800, height=600)



print(1)