import os
import sys
import numpy as np
import h5py
import open3d as o3d
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
if sys.version[0] == '3':
    from numba import jit, njit, float64, int64, boolean

def obj2pcd(path_obj, path_pcd):
    if not os.path.exists(path_pcd):
        os.makedirs(path_pcd)

    for home, dirs, files in os.walk(path_obj):
        for filename in files:
            # Filelist.append(os.path.join(home, filename))
            if os.path.splitext(filename)[1] == '.obj':
                # os.system('cp %s %s'%(os.path.join(home, filename), path_pcd))
                mesh = o3d.io.read_triangle_mesh(os.path.join(home, filename)) # load .obj mesh
                number_points = int(mesh.get_surface_area()/100) # get number of points according to surface area
                # print (number_points)
                pc = mesh.sample_points_poisson_disk(number_points, init_factor=5) # poisson disk sampling
                # print (np.asarray(pc.points).shape)
                o3d.io.write_point_cloud(os.path.join(path_pcd, os.path.splitext(filename)[0]+'.pcd'), pc, write_ascii=True)

def xyz2pcd(path_xyz, path_pcd):
    pcd = o3d.io.read_point_cloud(path_xyz)
    o3d.io.write_point_cloud(os.path.join(path_pcd,os.path.splitext(os.path.split(path_xyz)[1])[0]+'.pcd'), pcd, write_ascii=True)

def points2pcd(path, points):
    """
    path: ***/***/1.pcd
    points: ndarray, xyz+lable
    """
    point_num=points.shape[0]
    # handle.write('VERSION .7\nFIELDS x y z label object\nSIZE 4 4 4 4 4\nTYPE F F F I I\nCOUNT 1 1 1 1 1')
    # string = '\nWIDTH '+str(point_num)
    # handle.write(string)
    # handle.write('\nHEIGHT 1')
    # string = '\nPOINTS '+str(point_num)
    # handle.write(string)
    # handle.write('\nVIEWPOINT 0 0 0 1 0 0 0')
    # handle.write('\nDATA ascii')
    content = ''
    content += 'VERSION .7\nFIELDS x y z label object\nSIZE 4 4 4 4 4\nTYPE F F F I I\nCOUNT 1 1 1 1 1'
    content += '\nWIDTH '+str(points.shape[0])
    content += '\nHEIGHT 1'
    content += '\nPOINTS '+str(point_num)
    content += '\nVIEWPOINT 0 0 0 1 0 0 0'
    content += '\nDATA ascii'
    obj = -1 * np.ones((point_num,1))
    points_f = np.c_[points, obj]
    for i in range(point_num):
        content += '\n'+str(points_f[i, 0])+' '+str(points_f[i, 1])+' '+ \
                str(points_f[i, 2])+' '+str(int(points_f[i, 3]))+' '+str(int(points_f[i,4]))
    
    handle = open(path, 'w')
    handle.write(content)
    handle.close()



def load_pcd_data(file_path):
    '''
    file_path: path to .pcd file, i.e., './data/abc.pcd'
    return:
        res: np.array with shape (n,4)
    '''

    f = open(file_path, 'r')
    data = f.readlines()
    f.close()
    res = np.zeros((len(data[10:]),4), dtype = np.float64)
    for j,line in enumerate(data[10:]):
        # line = line.strip('\n')
        # xyzlable = line.split(' ')
        xyzlable = line.strip('\n').split(' ')
        res[j] = np.array([float(i) if '.' in i else int(i) for i in xyzlable[:4]]) # 5x faster
        
    return res

def draw(xyz, label):
    """
    display point clouds with label
    xyz: Nx3
    label: N
    """
    fig = plt.figure()
    # ax = Axes3D(fig,auto_add_to_figure=False)
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    max_edge = np.max(np.max(xyz, axis=0)-np.min(xyz, axis=0))

    ax.set_xlim(-0.5*max_edge, 0.5*max_edge)
    ax.set_ylim(-0.5*max_edge, 0.5*max_edge)
    ax.set_zlim(-0.5*max_edge, 0.5*max_edge)
    plt.show()

def show_obj(path):
    files = os.listdir(path)
    geometries = []
    for file in files:
        if os.path.splitext(file)[1] == '.obj':
            mesh = o3d.io.read_triangle_mesh(os.path.join(path, file))
            mesh.compute_vertex_normals()
            geometries.append(mesh)
    o3d.visualization.draw_geometries(geometries, path)


def fps_nojit(points, n_samples):
    """
    points: [N, >=3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]
    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]
    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    selected = 0
    sample_inds[0] = points_left[selected]
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
        dist_to_last_added_point = (
                (points[last_added, 0:3] - points[points_left, 0:3])**2).sum(-1) # [P - i]

        dists[points_left] = np.minimum(dist_to_last_added_point,
                                        dists[points_left]) # [P - i]

        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    # dist_to_centroid = (((points - points.mean(axis=0, keepdims= True))**2).sum(axis=1))**0.5
    # sorted_inds = np.argsort(dist_to_centroid)[-n_samples:]
    
    return points[sample_inds]
    
if sys.version[0] == '3':
    @njit(float64[:, :](float64[:, :], int64[:], int64[:], float64[:], int64))
    def fps_loop(points, points_left, sample_inds, dists, n_samples):
        # Iteratively select points for a maximum of n_samples
        for i in range(1, n_samples):
            # Find the distance to the last added point in selected
            # and all the others
            last_added = sample_inds[i-1]
            # arg1 = points[last_added][np.array([0,1,2])].repeat(points_left.shape[0]).reshape(points_left.shape[0], -1)
            # arg2 = points[points_left][:,np.array([0,1,2])]
            arg1 = points[last_added][np.array([0,1,2])]
            arg2 = points[points_left][:,np.array([0,1,2])]
            dist_to_last_added_point = np.power(arg1 - arg2 ,2).sum(-1)

            dists[points_left] = np.minimum(dist_to_last_added_point,
                                            dists[points_left]) # [P - i]

            # distance to the sampled points
            selected = np.argmax(dists[points_left])
            sample_inds[i] = points_left[selected]

            # Update points_left
            points_left = np.delete(points_left, selected)

        # dist_to_centroid = (((points - points.mean(axis=0, keepdims= True))**2).sum(axis=1))**0.5
        # sorted_inds = np.argsort(dist_to_centroid)[-n_samples:]

        return points[sample_inds]

    @njit(float64[:, :](float64[:, :], int64))
    def fps(points, n_samples):
        """
        points: [N, >=3] array containing the whole point cloud
        n_samples: samples you want in the sampled point cloud typically << N
        """
        # if type(points) != np.ndarray:
        #     points = np.array(points)
        
        # Represent the points by their indices in points
        points_left = np.arange(len(points), dtype =np.int64) # [P]
        # Initialise an array for the sampled indices
        sample_inds = np.zeros(n_samples, dtype=np.int64) # [S]
        # Initialise distances to inf
        dists = np.ones_like(points_left) * np.inf # [P]

        selected = 0
        sample_inds[0] = points_left[selected]
        points_left = np.delete(points_left, selected)  # [P - 1]
        return fps_loop(points, points_left, sample_inds, dists, n_samples)
else:
    fps = fps_nojit
    
if __name__ == '__main__':
    pass
    # import time
    # points :np.ndarray = np.load('points.npy')
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(points[:, :3])
    # o3d.visualization.draw_geometries([pc])
    
    # # start = time.time()
    # # _points = fps_nojit(points, 2048)
    # # print('compilation time jit : ', time.time()- start)
    
    # # start = time.time()
    # # _points = fps_nojit(points, 2048)
    # # print('run time jit : ', time.time()- start)
    # # pc.points = o3d.utility.Vector3dVector(_points[:, :3])
    # # # o3d.visualization.draw_geometries([pc])

    # start = time.time()
    # _points = fps(points, 2048)
    # print('run time njit : ', time.time()- start)
    # pc.points = o3d.utility.Vector3dVector(_points[:, :3])
    # o3d.visualization.draw_geometries([pc])

    # start = time.time()
    # _points = fps_nojit(points, 2048)
    # print('run time no jit : ', time.time()- start)
    # pc.points = o3d.utility.Vector3dVector(_points[:, :3])
    # o3d.visualization.draw_geometries([pc])
