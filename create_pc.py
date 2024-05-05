import os
import numpy as np
import open3d as o3d
import shutil
from utils.foundation import points2pcd
from utils.compatibility import listdir




def split(data_path: str,name):
    i = 0
    idx_start = 0
    comp = os.path.join(data_path, name+'.obj')

    with open(comp, 'r') as f:
        count = 0
        # get the name of current component mesh0]
        # prefix of single disassembled part
        split_path=os.path.join(data_path,name+'_split')
        os.makedirs(split_path,exist_ok=True)

        outstr = os.path.join(split_path,name)
        for line in f.readlines():
            if line[0:1] == 'o':
                idx_start = count
                path_part = outstr +'_'+str(i) + '.obj'
                i += 1
                # create new obj file
                g = open(path_part, 'w')
                g.write('mtllib ' + name + '.mtl' + '\n')
                g.write(line)
            elif line[0:1] == 'v':
                if i > 0:
                    # add vertices
                    g = open(outstr + '_'+ str(i - 1) + '.obj', 'a')
                    g.write(line)
                count += 1
            elif line[0:1] == 'f':
                new_line = 'f '
                new_line += str(int(line.split()[1]) - idx_start) + ' '
                new_line += str(int(line.split()[2]) - idx_start) + ' '
                new_line += str(int(line.split()[3]) - idx_start) + '\n'
                if i > 0:
                    # define the connection of vertices using the order of the new file
                    g = open(outstr + '_'+ str(i - 1) + '.obj', 'a')
                    g.write(new_line)
            else:
                if i > 0:
                    g = open(outstr + '_'+ str(i - 1) + '.obj', 'a')
                    g.write(line)
    # shutil.rmtree(split_path)
                # os.system('cp %s %s' % (os.path.join(model_path,namestr+'.mtl'),
                #                         split_path))

def convert(data_path,density,name):
    path_split=os.path.join(data_path,name+'_split')
    files = listdir(path_split)
    allpoints = np.zeros(shape=(1, 3))
    # os.makedirs(wz_path,exist_ok=True)
    for file in files:
        if os.path.splitext(file)[1] == '.obj':
            # load mesh
            mesh = o3d.io.read_triangle_mesh(os.path.join(path_split, file))
            if np.asarray(mesh.triangles).shape[0] > 1:
                key = os.path.abspath(os.path.join(path_split, file))
                # get number of points according to surface area
                number_points = int(mesh.get_surface_area() / density)
                if number_points <= 0:
                    continue
                # poisson disk sampling
                if number_points > 10101:
                    pc = mesh.sample_points_uniformly(number_points)
                    # o3d.visualization.draw_geometries([pc])
                else:
                    pc = mesh.sample_points_poisson_disk(number_points)
                xyz = np.asarray(pc.points)
                allpoints = np.concatenate((allpoints, xyz), axis=0)
    pc_ = o3d.geometry.PointCloud()
    pc_.points = o3d.utility.Vector3dVector(allpoints[1:,:])
    o3d.io.write_point_cloud(os.path.join(data_path, name + '.pcd'),pointcloud=pc_,write_ascii=True)
    shutil.rmtree(path_split)