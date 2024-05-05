import argparse
import open3d as o3d
from utils import *
from models import Point_NN
import matplotlib.pyplot as plt
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mn40')
    # parser.add_argument('--dataset', type=str, default='scan')

    # parser.add_argument('--split', type=int, default=1)
    # parser.add_argument('--split', type=int, default=2)
    parser.add_argument('--split', type=int, default=3)

    parser.add_argument('--bz', type=int, default=16)  # Freeze as 16

    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--dim', type=int, default=72)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--alpha', type=int, default=1000)
    parser.add_argument('--beta', type=int, default=100)

    args = parser.parse_args()
    return args


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return torch.tensor(pc[np.newaxis,...],dtype=torch.float32)


def read_pc(path):
    point_cloud = o3d.io.read_point_cloud(path)

    points = np.asarray(point_cloud.points)
    return point_cloud, points


def vis_pc(pc):
    if isinstance(pc, np.ndarray):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pc)
        o3d.visualization.draw_geometries([point_cloud])
    else:
        o3d.visualization.draw_geometries([pc])
    return


def project_pc(pc):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(pc[:, 0], pc[:, 1], s=12, c='b')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    # axes[0].set_xlabel('X')
    # axes[0].set_ylabel('Y')
    # axes[0].set_title('Projection onto XY Plane')

    axes[1].scatter(pc[:, 0], pc[:, 2], s=12, c='r')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    # axes[1].set_xlabel('X')
    # axes[1].set_ylabel('Z')
    # axes[1].set_title('Projection onto XZ Plane')

    axes[2].scatter(pc[:, 1], pc[:, 2], s=12, c='g')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    # axes[2].set_xlabel('Y')
    # axes[2].set_ylabel('Z')
    # axes[2].set_title('Projection onto YZ Plane')

    plt.tight_layout()

    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer.buffer_rgba())
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    image_array = cv2.resize(image_array,(750,250))
    return image_array




@torch.no_grad()
def main():
    print('==> Loading args..')
    args = get_arguments()
    print(args)


    print('==> Preparing model..')
    point_nn = Point_NN(input_points=args.points, num_stages=args.stages,
                        embed_dim=args.dim, k_neighbors=args.k,
                        alpha=args.alpha, beta=args.beta).cuda()
    point_nn.eval()

    global_feature = []
    global_feature_names = []
    pcds = []

    folder_path = os.path.join(ROOT_DIR,'Aehn3Test')


    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pcd'):
                pcd_path = os.path.join(root, file)
                pcds.append(pcd_path)

    for pcd in pcds:

        pcdd = o3d.io.read_point_cloud(pcd)
        pcdata = np.asarray(pcdd.points)

        input_pts = pc_normalize(pcdata).cuda().permute(0, 2, 1)
        point_features = point_nn(input_pts)
        point_pro_features = project_pc(pcdata)
        point_pro_features = np.reshape(point_pro_features,-1)
        
        # cv2.imshow('t',point_pro_features)
        # cv2.waitKey(0)
        global_feature.append(point_features)
        global_feature_names.append(pcd)

    tmp_feature_memory = torch.cat(global_feature, dim=0)
    tmp_feature_memory /= tmp_feature_memory.norm(dim=-1, keepdim=True)
    tmp_feature_memory=tmp_feature_memory.cpu().numpy()

    np.savez('pcd_features_k100',feature=tmp_feature_memory,name=global_feature_names)
    print('saved feature!')



if __name__ == '__main__':
    main()
