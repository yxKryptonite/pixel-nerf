import torch
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as transforms

def pixel_loss(pred, gt):
    return torch.nn.functional.mse_loss(pred, gt)

def visual_loss(pred, gt, device):
    '''
    pred and gt: 64x64x3
    using visual encoder to measure visual loss
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
    
    
def visual_similarity(pred, gt, device):
    '''
    pred and gt: 64x64x3
    using visual encoder to measure visual cosine similarity
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

        # 提取vgg16的特征，并计算cosine similarity
        pred_feature = vgg(pred)
        gt_feature = vgg(gt)
        print(pred_feature.shape)
        return torch.nn.functional.cosine_similarity(pred_feature, gt_feature, dim=1)