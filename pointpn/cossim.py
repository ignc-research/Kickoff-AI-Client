from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import open3d as o3d
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def pointpn(SNahts,tree,xml_path):
    name_id_dict={}
    for Snaht in SNahts:
        Name = Snaht.attrib.get('Name')
        ID = Snaht.attrib.get('ID')
        name_id_dict[Name] = ID

    feature_path = os.path.join(BASE_DIR, 'cnn_feature')
    tmp = np.load(os.path.join(feature_path, 'pnn_tpc_cnn_feature.npz'), allow_pickle=True)
    features = tmp['cnn_feature'].squeeze()
    names = [str(tname) for tname in tmp['name']]

    retrieved_map = {}
    for query_name in names:
        similar_list = []
        for ii, nn in enumerate(names):
            if query_name in nn:
                query_id = ii
                break
        query_pcf = features[query_id]
        similarity = cosine_similarity(features, np.reshape(query_pcf, (1, -1))).reshape(-1)
        sorted_idx = np.argsort(similarity)[::-1]
        for idx in sorted_idx:
            cname = str(names[idx])
            if idx == query_id:
                print(cname + ' query pcd! ' + str(idx) + ':' + str(similarity[idx]))
                # pcd = o3d.io.read_point_cloud(cname)
                # o3d.visualization.draw_geometries_with_editing([pcd], window_name="Open3D", width=800, height=600)
            else:
                if similarity[idx] < 0.95:
                    continue
                similar_list.append(name_id_dict[cname.split('/')[-1].split('.')[0]])
        if query_name.split('/')[-1].split('.')[0] not in name_id_dict:
            continue
        retrieved_map[name_id_dict[query_name.split('/')[-1].split('.')[0]]]=similar_list


    for SNaht in SNahts:
        src_name=SNaht.attrib.get('Name')
        dict={}
        for key, value in SNaht.attrib.items():
            if key == 'ID':
                if value in retrieved_map:
                    dict[key] = value
                    dict['Naht_ID'] = ','.join(retrieved_map[value])
                else:
                    continue
            elif key == 'Naht_ID':
                continue
            else:
                dict[key] = value
        SNaht.attrib.clear()
        for key, value in dict.items():
            SNaht.set(key, value)
        tree.write(xml_path)
    return retrieved_map

if __name__ == "__main__":
    pointpn()
