import torch
import torch.nn as nn
from bilinear_sampler import apply_disparity
import pytorch_ssim


# https://github.com/alwynmathew/monodepth-pytorch/blob/master/depth_modelv2.py

def SSIM(x: torch.Tensor, y: torch.Tensor):
    ssim_loss = pytorch_ssim.SSIM()
    return torch.clamp(1 - ssim_loss(x, y) / 2, 0, 1)

def scalePyramidGreyscale(img: torch.Tensor, num_scales: int):
    img = torch.mean(img, 1)
    img = torch.unsqueeze(img, 1)
    scaled_imgs = [img]
    s = img.size()
    h = int(s[2])
    w = int(s[3])
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        temp = nn.functional.upsample(img, [nh, nw], mode='bilinear')
        scaled_imgs.append(temp)
    return scaled_imgs

def generateLeft(right: torch.Tensor, disp: torch.Tensor):
    return apply_disparity(right, -disp)

def generateRight(left: torch.Tensor, disp: torch.Tensor):
    return apply_disparity(left, -disp)

def gradX(img: torch.Tensor):
    gx = img[:,:,:,:-1] - img[:,:,:,1:]
    return gx

def gradY(img: torch.Tensor):
    gy = img[:,:,:-1,:] - img[:,:,1:,:]
    return gy


def disparitySmoothness(disp, input_img):
    disp_gradients_x = [gradX(d) for d in disp]
    disp_gradients_y = [gradY(d) for d in disp]

    image_gradients_x = [gradX(img) for img in input_img]
    image_gradients_y = [gradY(img) for img in input_img]

    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]

    smoothness_x = [torch.nn.functional.pad(k,(0,1,0,0,0,0,0,0),mode='constant') for k in smoothness_x]
    smoothness_y = [torch.nn.functional.pad(k,(0,0,0,1,0,0,0,0),mode='constant') for k in smoothness_y]

    return smoothness_x + smoothness_y


def depthLoss(left: torch.Tensor, right: torch.Tensor, disp: list):
    levels = len(disp)
    left_pyramid = scalePyramidGreyscale(left, levels)
    right_pyramid = scalePyramidGreyscale(right, levels)

    # we unpack the channels for disparity estimation
    left_disp = [torch.unsqueeze(d[:,0,:,:], 1) for d in disp]
    right_disp= [torch.unsqueeze(d[:,1,:,:], 1) for d in disp]

    left_est = [generateLeft(right, disp) for right,disp in zip(right_pyramid, left_disp)]
    right_est = [generateRight(left, disp) for left,disp in zip(left_pyramid, right_disp)]

    # LR consistency
    right_to_left_disp = [generateLeft(right, left) for right, left in zip(right_disp, left_disp)]
    left_to_right_disp = [generateRight(left, right) for right, left in zip(right_disp, left_disp)]

    # disparity smoothness
    left_disp_smoothness = disparitySmoothness(left_disp, left_pyramid)
    right_disp_smoothness = disparitySmoothness(right_disp, right_pyramid)

    # image reconstruction loss
    left_l1 = [torch.abs(left, est) for left, est in zip(left_pyramid, left_est)]
    left_l1_reconstruction_loss = [torch.mean(l) for l in left_l1]

    right_l1 = [torch.abs(right, est) for right, est in zip(right_pyramid, right_est)]
    right_l1_reconstruction_loss = [torch.mean(r) for r in right_l1]

    left_ssim_loss  = [SSIM(est, img) for est, img in zip(left_est, left_pyramid)]
    right_ssim_loss = [SSIM(est, img) for est, img in zip(right_est, right_pyramid)]

    # weighted sum of the things

    right_image_loss = [0.85 * ssim + 0.15 * l1 for ssim, l1 in zip(right_ssim_loss, right_l1_reconstruction_loss)]
    left_image_loss = [0.85 * ssim + 0.15 * l1 for ssim, l1 in zip(left_ssim_loss, left_l1_reconstruction_loss)]

    image_loss = [left + right for left, right in zip(left_image_loss, right_image_loss)]
    image_loss = sum(image_loss)

    # disparity smoothness
    left_disp_loss = [torch.mean(torch.abs(left_disp_smoothness[i])) / 2** i for i in range(levels)]
    right_disp_loss = [torch.mean(torch.abs(right_disp_smoothness[i])) / 2**i for i in range(levels)]
    disp_grad_loss = sum(left_disp_loss + right_disp_loss)

    # lr consistency
    left_lr_loss = [torch.mean(torch.abs(right_to_left_disp[i] - left_disp[i])) for i in range(4)]
    right_lr_loss = [torch.mean(torch.abs(left_to_right_disp[i] - right_disp[i])) for i in range(4)]
    lr_loss = sum(left_lr_loss + right_lr_loss)

    return image_loss, disp_grad_loss, lr_loss

