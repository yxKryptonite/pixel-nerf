import torch
import numpy as np

def pose_matrix(p, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def sample_sphere(radius):
    '''sample points on a ball'''
    # 生成随机数u和v
    u = np.random.uniform(0, 1)
    v = np.random.uniform(0, 1)
    
    # 计算球面坐标
    x = np.cos(2 * np.pi * v) * np.sqrt(1 - u**2)
    y = np.sin(2 * np.pi * v) * np.sqrt(1 - u**2)
    z = u
    
    # 归一化得到单位向量
    n = np.array([x, y, z])
    n /= np.linalg.norm(n)
    
    # 计算旋转矩阵将n朝向球心
    theta = np.arccos(n[2])
    if n[0] == 0:
        if n[1] == 0:
            R = np.eye(3)
        else:
            R = np.array([[0, -n[2], n[1]], [0, 0, -n[0]], [n[1], n[0], 0]])
    else:
        R = np.array([[n[0]**2, n[0]*n[1]-n[2], n[0]*n[2]+n[1]],
                      [n[0]*n[1]+n[2], n[1]**2, n[1]*n[2]-n[0]],
                      [n[0]*n[2]-n[1], n[1]*n[2]+n[0], n[2]**2]])
    
    # 将坐标点沿着单位向量n平移至球面上
    p = np.array([0, 0, radius])
    p = np.dot(R, p)
    
    return p, R

def naive_search(space_info, image_info, render_info):
    '''
    space_info: {
        angle: float
        elevation: float
        scale: float
        radius: float
        focal: float
        z_near: float
        z_far: float
    }
    image_info: {
        target: torch.Tensor
    }
    render_info: {
        renderer: torch.nn.Module
        device: torch.device
    }
    
    return
    cam_pose:
    torch.tensor([[ a, b, c, x ],
                  [ d, e, f, y ],
                  [ g, h, i, z ],
                  [ 0, 0, 0, 1 ]], device=device)
    '''
    radius = space_info['radius'] if space_info['radius'] is not None \
        else (space_info['z_near'] + space_info['z_far']) / 2
    focal = space_info['focal']
    scale = space_info['scale']
    
    # sample points on a sphere of radius
    poses = []
    POINT_NUM = 200
    for i in range(POINT_NUM):
        p, R = sample_sphere(radius)
        pose = torch.tensor(pose_matrix(p, R))
        poses.append(pose)
        
    for pose in poses:
        pass