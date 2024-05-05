from utils.xml_parser import list2array,parse_frame_dump
from utils.foundation import load_pcd_data, points2pcd, fps
from utils.math_util import rotate_mat, rotation_matrix_from_vectors
import os.path
import open3d as o3d
import numpy as np
import copy
import time
from openpoints.dataset.mydataset.tools_dataset import pc_normalize,farthest_point_sampling

CURRENT_PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(CURRENT_PATH)
def image_save(retrieved_map_name,wz_path):
    result_image_dir = os.path.join(ROOT, 'result_image')
    os.makedirs(result_image_dir, exist_ok=True)
    for key, values in retrieved_map_name.items():
        if len(values):
            query_img_dir = os.path.join(result_image_dir, key)
            os.makedirs(query_img_dir, exist_ok=True)
            pcd1 = o3d.io.read_point_cloud(os.path.join(wz_path, key + '.pcd'))
            point1 = np.array(pcd1.points).astype('float32')
            point1_cloud = o3d.geometry.PointCloud()
            point1_cloud.points = o3d.utility.Vector3dVector(point1)
            for value in values:
                pcd2 = o3d.io.read_point_cloud(os.path.join(wz_path, value + '.pcd'))
                point2 = np.array(pcd2.points).astype('float32')
                point2_cloud = o3d.geometry.PointCloud()
                point2_cloud.points = o3d.utility.Vector3dVector(point2)

                point1_cloud.paint_uniform_color([1, 0, 0])
                point2_cloud.paint_uniform_color([0, 1, 0])
                all_point = point1_cloud + point2_cloud

                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(all_point)
                vis.update_geometry(all_point)
                vis.poll_events()
                vis.update_renderer()

                save_name = key + '_' + value
                save_file = os.path.join(query_img_dir, save_name + '.png')
                vis.capture_screen_image(save_file)
                vis.destroy_window()
                time.sleep(0.2)
    return

def process_pc(query_pcdir, pcs):
    datas = []
    names = []
    for pc in pcs:
        if pc.endswith('pcd'):
            tmp_name = os.path.join(query_pcdir, pc)
            pcd = o3d.io.read_point_cloud(tmp_name)  # 路径需要根据实际情况设置
            input = np.asarray(pcd.points)  # A已经变成n*3的矩阵

            lens = len(input)
            if lens == 0:
                continue
            if lens < 2048:
                ratio = int(2048 / lens + 1)
                tmp_input = np.tile(input, (ratio, 1))
                input = tmp_input[:2048]

            if lens > 2048:
                # np.random.shuffle(input) # 每次取不一样的1024个点
                input = farthest_point_sampling(input, 2048)

            input = pc_normalize(input)
    return input

def get_distance_from_SNaht(SNaht):
    list1 = []
    for punkt in SNaht.iter('Punkt'):
        list2 = []
        list2.append(float(punkt.attrib['X']))
        list2.append(float(punkt.attrib['Y']))
        list2.append(float(punkt.attrib['Z']))
        list1.append(list2)
    weld_info=np.asarray(list1)
    seam_vector=weld_info[-1,:]-weld_info[0,:]
    x_diff = np.max(weld_info[:, 0]) - np.min(weld_info[:, 0])
    if x_diff < 2:
        x_diff = 0
    y_diff = np.max(weld_info[:, 1]) - np.min(weld_info[:, 1])
    if y_diff < 2:
        y_diff = 0
    z_diff = np.max(weld_info[:, 2]) - np.min(weld_info[:, 2])
    if z_diff < 2:
        z_diff = 0
    distance = int(pow(pow(x_diff, 2) + pow(y_diff, 2) + pow(z_diff, 2), 0.5)) + 25
    return distance,seam_vector.astype(int)

def get_distance_from_seam(seam):
    x_diff = np.max(seam[:, 0]) - np.min(seam[:, 0])
    if x_diff < 2:
        x_diff = 0
    y_diff = np.max(seam[:, 1]) - np.min(seam[:, 1])
    if y_diff < 2:
        y_diff = 0
    z_diff = np.max(seam[:, 2]) - np.min(seam[:, 2])
    if z_diff < 2:
        z_diff = 0
    distance = int(pow(pow(x_diff, 2) + pow(y_diff, 2) + pow(z_diff, 2), 0.5)) + 25
    return distance,x_diff,y_diff,z_diff

def get_weld_info(xml_path):
    frames = list2array(parse_frame_dump(xml_path))
    weld_infos=[]
    for i in range(len(frames)):
        tmp = frames[frames[:, -2] == i]
        if len(tmp)==0:
            tmp = frames[frames[:, -2] == str(i)]
        if len(tmp) != 0:
            weld_infos.append(tmp)
    # weld_infos=np.vstack(weld_infos)
    weld_infos=np.asarray(weld_infos)
    return weld_infos

def get_ground_truth(weld_infos):
    gt_id={}
    gt_name={}
    for i in range (len(weld_infos)):
        gt_id_list=[]
        gt_name_list = []
        name_1=weld_infos[i][0,0]
        weld_info_1=weld_infos[i][:,14:-4].astype(float)
        seam_1=weld_infos[i][:,4:7].astype(float)
        ID_1=weld_infos[i][0,-1]
        spot_number_1=len(seam_1)
        distance_1,x_diff1,y_diff1,z_diff1=get_distance_from_seam(seam_1)
        for j in range(len(weld_infos)):
            name_2 = weld_infos[j][0, 0]
            weld_info_2=weld_infos[j][:,14:-4].astype(float)
            seam_2 = weld_infos[j][:, 4:7].astype(float)
            ID_2 = weld_infos[j][0, -1]
            spot_number_2 = len(seam_2)
            distance_2,x_diff2,y_diff2,z_diff2=get_distance_from_seam(seam_2)
            if ID_1==ID_2:
                continue
            if abs(distance_1-distance_2)>3:
                continue
            if (abs(x_diff1-x_diff2)>2 or abs(y_diff1-y_diff2)>2 or abs(z_diff1
                                                                         -z_diff2)>2):
                continue
            if spot_number_1 != spot_number_2:
                continue
            flag=True
            for n in range(weld_info_1.shape[0]):
                for m in range(weld_info_1.shape[1]):
                    if(abs(weld_info_1[n][m]-weld_info_2[n][m])>0.000005):
                        flag=False
            if flag:
                gt_id_list.append(str(ID_2))
                gt_name_list.append(str(name_2))
        gt_id[str(ID_1)]=gt_id_list
        gt_name[str(name_1)]=gt_name_list
    return gt_id,gt_name


def sample_and_label_alternative(path, path_pcd,label_dict, class_dict, density=40):
    '''Convert mesh to pointcloud
    two pc will be generated, one is .pcd format with labels, one is .xyz format withou labels
    Args:
        path (str): path to single component
        label_dict (dict): the class name with an index
        density (int): Sampling density, the smaller the value the greater the point cloud density
    '''
    # get the current component name
    namestr = os.path.split(path)[-1]
    files = os.listdir(path)
    # label_list = {}
    label_count = 0

    allpoints = np.zeros(shape=(1,4))
    for file in files:
        if os.path.splitext(file)[1] == '.obj':
            # load mesh
            mesh = o3d.io.read_triangle_mesh(os.path.join(path, file))
            if np.asarray(mesh.triangles).shape[0] > 1:
                key = os.path.abspath(os.path.join(path, file))
                label = label_dict[class_dict[key]]
                # get number of points according to surface area
                number_points = int(mesh.get_surface_area()/density)
                if number_points <= 0:
                    number_points = 1000
                    f = open('objects_with_0_points.txt', 'a')
                    f.write(file)
                    f.write('\n')
                    f.close()
                # poisson disk sampling
                if number_points > 10101:
                    pc = mesh.sample_points_uniformly(number_points)
                    # o3d.visualization.draw_geometries([pc])
                else:
                    pc = mesh.sample_points_poisson_disk(number_points)
                    # o3d.visualization.draw_geometries([pc])
                xyz = np.asarray(pc.points)
                l = label * np.ones(xyz.shape[0])
                xyzl = np.c_[xyz, l]
                allpoints = np.concatenate((allpoints, xyzl), axis=0)
    points2pcd(os.path.join(path_pcd, namestr+'.pcd'), allpoints[1:])


def points2pcd(path, points):
    """
    path: ***/***/1.pcd
    points: ndarray, xyz+norm
    """
    point_num = points.shape[0]
    # handle.write('VERSION .7\nFIELDS x y z norm object\nSIZE 4 4 4 4 4\nTYPE F F F I I\nCOUNT 1 1 1 1 1')
    # string = '\nWIDTH '+str(point_num)
    # handle.write(string)
    # handle.write('\nHEIGHT 1')
    # string = '\nPOINTS '+str(point_num)
    # handle.write(string)
    # handle.write('\nVIEWPOINT 0 0 0 1 0 0 0')
    # handle.write('\nDATA ascii')
    content = ''
    content += 'VERSION .7\nFIELDS x y z label object\nSIZE 4 4 4 4 4\nTYPE F F F I I\nCOUNT 1 1 1 1 1'
    content += '\nWIDTH ' + str(points.shape[0])
    content += '\nHEIGHT 1'
    content += '\nPOINTS ' + str(point_num)
    content += '\nVIEWPOINT 0 0 0 1 0 0 0'
    content += '\nDATA ascii'
    obj = -1 * np.ones((point_num, 1))
    points_f = np.c_[points, obj]
    for i in range(point_num):
        content += '\n' + str(points_f[i, 0]) + ' ' + str(points_f[i, 1]) + ' ' + \
                   str(points_f[i, 2]) + ' ' + str(int(points_f[i, 3])) + ' ' + \
                   str(int(points_f[i, 4]))+ ' ' + str(int(points_f[i, 5]))

    handle = open(path, 'w')
    handle.write(content)
    handle.close()
class WeldScene:
    '''
    Component point cloud processing, mainly for slicing

    Attributes:
        path_pc: path to labeled pc

    '''

    def __init__(self, pc_path):
        self.pc = o3d.geometry.PointCloud()
        pcd=o3d.io.read_point_cloud(pc_path)
        self.xyz = np.asarray(pcd.points)
        self.pc.points = o3d.utility.Vector3dVector(np.asarray(self.xyz))

    def rotation(self,axis,norm):
        rot_axis = np.cross(axis, norm) / (np.linalg.norm(axis) * np.linalg.norm(norm))
        theta = np.arccos((axis @ norm)) / (np.linalg.norm(axis) * np.linalg.norm(norm))
        rotation = rotate_mat(axis=rot_axis, radian=theta)
        return rotation

    def get_distance_and_translate(self,weld_info):
        x_center = (np.max(weld_info[:,1]) + np.min(weld_info[:,1])) / 2
        y_center = (np.max(weld_info[:,2]) + np.min(weld_info[:,2])) / 2
        z_center = (np.max(weld_info[:,3]) + np.min(weld_info[:,3])) / 2
        translate=np.array([x_center,y_center,z_center])
        x_diff=np.max(weld_info[:,1])-np.min(weld_info[:,1])
        if x_diff<2:
            x_diff=0
        y_diff = np.max(weld_info[:, 2])-np.min(weld_info[:, 2])
        if y_diff<2:
            y_diff=0
        z_diff = np.max(weld_info[:, 3])-np.min(weld_info[:, 3])
        if z_diff<2:
            z_diff=0
        distance = int(pow(pow(x_diff,2)+pow(y_diff,2)+pow(z_diff,2),0.5))+50

        return distance,translate

    def bbox_(self,norm1,norm2,distance,extent,mesh_arrow1,mesh_arrow2):
        norm_ori = np.array([0, 0, 1])
        vector_seam=abs(np.cross(norm1,norm2))
        crop_extent1=np.array([100,200,300])
        crop_extent2 = np.array([100,200,300])
        rotation_bbox1 = rotation_matrix_from_vectors(norm_ori, norm_ori)
        rotation_bbox2 = rotation_matrix_from_vectors(norm_ori, norm_ori)
        axis_mesh = np.array([0, 0, 1])
        if abs(norm1[2]) == 0 and (abs(norm1[0]) != 0 or abs(norm1[1]) != 0):
            rotation_mesh1 = self.rotation(axis_mesh, norm1)
            mesh_arrow1.rotate(rotation_mesh1, center=np.array([0, 0, 0]))
            print('aa')
        elif norm1[2] < 0 and abs(norm1[0]) == 0 and abs(norm1[1]) == 0:
            rotation_mesh1=np.array([1,0,0,0,1,0,0,0,-1]).reshape(3,3)
            mesh_arrow1.rotate(rotation_mesh1, center=np.array([0, 0, 0]))
            print('bb')

        if abs(norm2[2]) == 0 and (abs(norm2[0]) != 0 or abs(norm2[1]) != 0):
            rotation_mesh2 = self.rotation(axis_mesh, norm2)
            mesh_arrow2.rotate(rotation_mesh2, center=np.array([0, 0, 0]))
            print('cc')
        elif norm2[2] < 0 and abs(norm2[0]) == 0 and abs(norm2[1]) == 0:
            rotation_mesh2=np.array([1,0,0,0,1,0,0,0,-1]).reshape(3,3)
            mesh_arrow2.rotate(rotation_mesh2, center=np.array([0, 0, 0]))
            print('dd')
        #the situation that the norm vector is parallel to axis
            #norm [0,0,1], plane in XOY

        if abs(norm1[2]) != 0 and abs(norm1[1]) == 0 and abs(norm1[0]) == 0:
            if np.max(vector_seam)==vector_seam[0]:
                crop_extent1 = np.array([distance,extent,1])
            elif np.max(vector_seam)==vector_seam[1]:
                crop_extent1 = np.array([extent, distance, 1])
        if abs(norm2[2]) != 0 and abs(norm2[1]) == 0 and abs(norm2[0]) == 0:
            if np.max(vector_seam)==vector_seam[0]:
                crop_extent2 = np.array([distance,extent,1])
            elif np.max(vector_seam)==vector_seam[1]:
                crop_extent2 = np.array([extent, distance, 1])
        #norm [0,1,0], plane in XOZ
        if abs(norm1[1]) != 0 and abs(norm1[0]) == 0 and abs(norm1[2]) == 0:
            if np.max(vector_seam)==vector_seam[0]:
                crop_extent1 = np.array([distance,1,extent])
            elif np.max(vector_seam)==vector_seam[2]:
                crop_extent1 = np.array([extent, 1, distance])
        if abs(norm2[1]) != 0 and abs(norm2[0]) == 0 and abs(norm2[2]) == 0:
            if np.max(vector_seam)==vector_seam[0]:
                crop_extent2 = np.array([distance,1,extent])
            elif np.max(vector_seam)==vector_seam[2]:
                crop_extent2 = np.array([extent, 1, distance])

        # if y_diff > 0 and x_diff == 0 and z_diff == 0:
            #norm [1,0,0], plane in YOZ
        if abs(norm1[0]) != 0 and abs(norm1[1]) == 0 and abs(norm1[2]) == 0:
            if np.max(vector_seam)==vector_seam[1]:
                crop_extent1 = np.array([1,distance, extent])
            elif np.max(vector_seam)==vector_seam[2]:
                crop_extent1 = np.array([1, extent, distance])
        if abs(norm2[0]) != 0 and abs(norm2[1]) == 0 and abs(norm2[2]) == 0:
            if np.max(vector_seam)==vector_seam[1]:
                crop_extent2 = np.array([1,distance, extent])
            elif np.max(vector_seam)==vector_seam[2]:
                crop_extent2 = np.array([1, extent, distance])

        #the situation that the norm vector is not parallel to axis
        #norm on XOZ plane
        if abs(norm1[1]) == 0 and abs(norm1[0]) != 0 and abs(norm1[2]) != 0:
            axis_bbox1=np.array([1,0,0])
            crop_extent1=np.array([1,extent,distance])
            rotation_mesh1=self.rotation(axis_mesh,norm1)
            mesh_arrow1.rotate(rotation_mesh1,center=np.array([0, 0, 0]))
            rotation_bbox1=self.rotation(axis_bbox1,norm1)
            if abs(norm1[0])>=abs(norm1[2]):
                axis_bbox2 = np.array([1, 0, 0])
            elif abs(norm1[0])<abs(norm1[2]):
                axis_bbox2 = np.array([[0,0,1]])
            rotation_bbox2 = self.rotation(axis_bbox2, norm1)

        if abs(norm2[1]) == 0 and abs(norm2[0]) != 0 and abs(norm2[2]) != 0:
            axis_bbox2=np.array([1,0,0])
            crop_extent2=np.array([1,extent,distance])
            rotation_mesh2=self.rotation(axis_mesh,norm2)
            mesh_arrow2.rotate(rotation_mesh2,center=np.array([0, 0, 0]))
            rotation_bbox2=self.rotation(axis_bbox2,norm2)
            if abs(norm2[0])>=abs(norm2[2]):
                axis_bbox1 = np.array([1, 0, 0])
            elif abs(norm2[0])<abs(norm2[2]):
                axis_bbox1 = np.array([[0,0,1]])
            rotation_bbox1 = self.rotation(axis_bbox1, norm2)

        #norm on XOY plane
        if abs(norm1[2]) == 0 and abs(norm1[0]) != 0 and abs(norm1[1]) != 0:
            axis_bbox1 = np.array([1, 0, 0])
            crop_extent1 = np.array([1, distance, extent])
            rotation_mesh1 = self.rotation(axis_mesh, norm1)
            mesh_arrow1.rotate(rotation_mesh1, center=np.array([0, 0, 0]))
            rotation_bbox1 = self.rotation(axis_bbox1, norm1)
            if abs(norm1[0])>=abs(norm1[1]):
                axis_bbox2 = np.array([1,0,0])
            elif abs(norm1[0])<abs(norm1[1]):
                axis_bbox2 = np.array([0, 1, 0])
            rotation_bbox2 = self.rotation(axis_bbox2, norm1)

        if abs(norm2[2]) == 0 and abs(norm2[0]) != 0 and abs(norm2[1]) != 0:
            axis_bbox2=np.array([1,0,0])
            axis_bbox1=np.array([1,0,0])
            crop_extent2=np.array([1,distance,extent])
            rotation_mesh2=self.rotation(axis_mesh,norm2)
            mesh_arrow2.rotate(rotation_mesh2,center=np.array([0, 0, 0]))
            rotation_bbox2=self.rotation(axis_bbox2,norm2)
            if abs(norm2[0]) >= abs(norm2[1]):
                axis_bbox1 = np.array([1, 0, 0])
            elif abs(norm2[0]) < abs(norm2[1]):
                axis_bbox1 = np.array([0, 1, 0])
            rotation_bbox1 = self.rotation(axis_bbox1, norm2)

        #norm on YOZ plane
        if abs(norm1[0]) == 0 and abs(norm1[1]) != 0 and abs(norm1[2]) != 0:
            axis_bbox1 = np.array([0, 0, 1])
            crop_extent1 = np.array([extent, distance, 1])
            rotation_mesh1 = self.rotation(axis_mesh, norm1)
            mesh_arrow1.rotate(rotation_mesh1, center=np.array([0, 0, 0]))
            rotation_bbox1 = self.rotation(axis_bbox1, norm1)
            if abs(norm1[1])>=abs(norm1[2]):
                axis_bbox2 = np.array([0, 1, 0])
            elif abs(norm1[1])<abs(norm1[2]):
                axis_bbox2 = np.array([0, 0, 1])
            rotation_bbox2 = self.rotation(axis_bbox2, norm1)

        if abs(norm2[0]) == 0 and abs(norm2[1]) != 0 and abs(norm2[2]) != 0:
            axis_bbox2=np.array([0,0,1])
            crop_extent2=np.array([extent,distance,1])
            rotation_mesh2=self.rotation(axis_mesh,norm2)
            mesh_arrow2.rotate(rotation_mesh2,center=np.array([0, 0, 0]))
            rotation_bbox2=self.rotation(axis_bbox2,norm2)
            if abs(norm2[1])>=abs(norm2[2]):
                axis_bbox1 = np.array([0, 1, 0])
            elif abs(norm2[1])<abs(norm2[2]):
                axis_bbox1 = np.array([0, 0, 1])
            rotation_bbox1 = self.rotation(axis_bbox1, norm2)

        return rotation_bbox1,rotation_bbox2,crop_extent1,crop_extent2

    def crop(self, weld_info,num_points=2048, vis=False):
        '''Cut around welding spot

        Args:
            weld_info (np.ndarray): welding info, including torch type, weld position, surface normals, torch pose
            crop_size (int): side length of cutting bbox in mm
            num_points (int): the default point cloud contains a minimum of 2048 points, if not enough then copy and fill
            vis (Boolean): True for visualization of the slice while slicing
        Returns:
            xyzl_crop (np.ndarray): cropped pc with shape num_points*4, cols are x,y,z,label
            cropped_pc (o3d.geometry.PointCloud): cropped pc for visualization
            weld_info (np.ndarray): update the rotated component pose for torch (if there is)
        '''
        # print(weld_info)
        torch_path='./data/torch/MRW510_10GH.obj'
        torch_model=o3d.io.read_triangle_mesh(torch_path)
        # tf=np.zeros((4,4))
        # tf[3,3]=1.0
        # tf[0:3,0:3]=weld_info[1,14:23].reshape(3,3).T
        # tf[0:3,3]=weld_info[0,1:4].reshape(1,3)


        pc = copy.copy(self.pc)
        weld_seam_points=weld_info[:,1:4]
        weld_seam=o3d.geometry.PointCloud()
        weld_seam.points=o3d.utility.Vector3dVector(weld_seam_points)
        distance,translate=self.get_distance_and_translate(weld_info)
        pc.translate(-translate)
        weld_seam.translate(-translate)
        # print(np.array([weld_seam.points[0]]))
        # tf[0:3, 3]=np.array([weld_seam.points[0]]).reshape(3,)
        # print('tf',tf)
        # torch_model.compute_vertex_normals()
        # torch_model.transform(tf)
        extent = 250
        # crop_extent = np.array([max(x_diff,extent), max(y_diff,extent),max(z_diff,extent)])
        crop_extent=np.array([distance,extent+5,extent+5])
        # move the coordinate center to the welding spot

        # rotation at this welding spot  1.
        rot = weld_info[0,10:13] * np.pi / 180
        rotation = rotate_mat(axis=[1, 0, 0], radian=rot[0])

        # tf1 = np.zeros((4, 4))
        # tf1[3, 3] = 1.0
        # tf1[0:3, 0:3] = rotation
        # pc.transform(tf)
        # weld_seam.transform(tf)
        # new normals
        norm1 = np.around(weld_info[0, 4:7], decimals=6)
        norm2 = np.around(weld_info[0, 7:10], decimals=6)
        # print(norm1,norm2)
        norm1_r = np.matmul(rotation, norm1.T)
        norm2_r = np.matmul(rotation, norm2.T)
        # torch pose
        for i in range(weld_info.shape[0]):
            weld_info[i,4:7] = norm1_r
            weld_info[i,7:10] = norm2_r

        coor1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])
        mesh_arrow1 = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=90 * 1,
            cone_radius=3 * 1,
            cylinder_height=40 * 1,
            cylinder_radius=3 * 1
        )
        mesh_arrow1.paint_uniform_color([0, 0, 1])

        mesh_arrow2 = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=90 * 1,
            cone_radius=3 * 1,
            cylinder_height=40 * 1,
            cylinder_radius=3 * 1
        )
        mesh_arrow2.paint_uniform_color([0, 1, 0])
        norm_ori = np.array([0, 0, 1])
        # bounding box of cutting area
        # rotation_bbox1, rotation_bbox2, crop_extent1, crop_extent2=self.bbox_(norm1,norm2,distance,extent,mesh_arrow1
                                                                              # ,mesh_arrow2)

        rotation_bbox = rotation_matrix_from_vectors(norm_ori, norm_ori)
        seams_direction=np.cross(norm1,norm2)
        if (abs(seams_direction[0])==0 and (abs(seams_direction[1])!=0 or abs(seams_direction[2])!=0)) or (abs(seams_direction[0])!=0 and (abs(seams_direction[1])!=0 or abs(seams_direction[2])!=0)):
            rotation_bbox=self.rotation(np.array([1,0,0]),seams_direction)
        center_bbox=norm2/np.linalg.norm(norm2)*extent/2+norm1/np.linalg.norm(norm1)*extent/2
        bbox = o3d.geometry.OrientedBoundingBox(center=center_bbox, R=rotation_bbox,extent=crop_extent)

        pc.paint_uniform_color([1,0,0])
        weld_seam.paint_uniform_color([0,1,0])
        cropped_pc_large=pc.crop(bbox)
        idx_crop_large=bbox.get_point_indices_within_bounding_box(pc.points)
        xyz_crop = self.xyz[idx_crop_large]
        xyz_crop -= translate
        xyz_crop_new = np.matmul(rotation_matrix_from_vectors(norm_ori, norm_ori), xyz_crop.T).T
        # print('xyz_crop_new.shape[0]',xyz_crop_new.shape[0])
        # if vis:
        # if xyz_crop_new.shape[0]<500:
        # o3d.visualization.draw_geometries([cropped_pc_large,bbox,weld_seam,mesh_arrow1,mesh_arrow2])

        while (len(xyz_crop_new)!=0 and xyz_crop_new.shape[0] < num_points):
            xyz_crop_new = np.vstack((xyz_crop_new, xyz_crop_new))

        # print('xyz_crop_new',xyz_crop_new.shape)
        xyz_crop_new = fps(xyz_crop_new, num_points)
        xyz_crop_new=np.vstack((np.array(weld_seam.points),xyz_crop_new))
        norm=np.vstack((norm1,norm2))
        return xyz_crop_new, cropped_pc_large, weld_info
if __name__ == "__main__":
    xml_path = 'data/Reisch.xml'
    weld_infos=get_weld_info(xml_path)
    weld_infos = np.vstack(weld_infos)

