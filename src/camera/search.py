import torch
import numpy as np
import torch.nn as nn
np.random.seed(514)

def pose_matrix(p, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def sample_pose_sphere(radius):
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
    
    return torch.tensor(pose_matrix(p, R), dtype=torch.float32)


class PoseSampler(nn.Module):
    def __init__(self, radius=2.6):
        super().__init__()
        self.radius = radius
        self.pose = None
    
    def forward(self, batch_size):
        return sample_pose_sphere(self.radius).repeat(batch_size, 1, 1)