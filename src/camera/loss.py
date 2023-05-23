import torch
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as transforms
import clip
from PIL import Image
import os

def pixel_loss(pred, gt):
    return torch.nn.functional.mse_loss(pred, gt)


def vgg_loss(pred, gt, device):
    '''
    pred and gt: 64x64x3
    using VGG16 encoder to measure visual loss
    '''
    vgg = models.vgg16(pretrained=True).features
    vgg.eval().to(device=device)

    # 将图片转换为tensor并进行归一化
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
    with torch.no_grad():
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

        # 提取vgg16的特征，并计算L1 loss
        pred_feature = vgg(pred)
        gt_feature = vgg(gt)
        criterion = nn.L1Loss()
        loss = criterion(pred_feature, gt_feature)

    return loss
    
    
def clip_similarity(tmp_path, gt_path, device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    gt = preprocess(Image.open(gt_path)).unsqueeze(0).to(device)
    frames = []
    for file in os.listdir(tmp_path):
        if not file.endswith(".png"):
            continue
        frames.append(preprocess(Image.open(os.path.join(tmp_path, file))).unsqueeze(0).to(device))
            
    with torch.no_grad():
        gt_feature = model.encode_image(gt)
        pred_feature = model.encode_image(torch.cat(frames, dim=0))
        sim = torch.cosine_similarity(gt_feature, pred_feature, dim=-1).mean().item()
        
        return sim