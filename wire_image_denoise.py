#!/usr/bin/env python
import sys
sys.path.append('../learnable_wavelet/')
sys.path.append('../learnable_wavelet/learn2D/')

import os
import sys
from tqdm import tqdm
import importlib
import time
from PIL import Image

import numpy as np
from scipy import io

import matplotlib.pyplot as plt
plt.gray()

import cv2
from skimage.metrics import structural_similarity as ssim_func

import torch
import torch.nn
from torch.optim.lr_scheduler import LambdaLR
from pytorch_msssim import ssim

from dataset import Single_Image_Dataset, INR_Single_Image_Dataset
from net import waveletNet
from torch.utils.data import DataLoader
from torchvision import transforms

from modules import models
from modules import utils

import argparse

def get_mgrid(sidelen, noise_channel=True, dim=2):
    '''
    Generates input to INR wavelet model with 2/3 channels.
    The mandatory two channels are for x-coordinate and y-coordinate projected in the range of -1 to 1
    The third channels is the optiional noise channel
    '''
    if type(sidelen) is tuple:
        tensors = (torch.linspace(-1, 1, steps=sidelen[0]), torch.linspace(-1, 1, steps=sidelen[1]))
    else:
        tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    if noise_channel:
        if type(sidelen) is tuple:
            noise = torch.randn(sidelen).unsqueeze(-1)
        else:
            noise = torch.randn((sidelen, sidelen)).unsqueeze(-1)
        mgrid = torch.cat((mgrid, noise), dim=-1).transpose(0,2).transpose(1,2)
    else:
        mgrid = mgrid.transpose(0,2).transpose(1,2)
    return mgrid

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description="Arguments for CDE in BCR")
    
    parser.add_argument("--seed", type=int, default=0, help="Setting seed for the entire experiment")
    parser.add_argument("--exp", type=str, default='learn2D_feb10', help="Adjusted in code: Experiment foler name")
    
    parser.add_argument('--noise_channel', default=True, action='store_false', help='Whether to use noise channel as input')
    parser.add_argument('--image_size', type=int, default=256, help='Size of square image')
    parser.add_argument('--erase_patch', default=False, action='store_true', help='Whether to earse patch in input image to wavelet model')
    
    parser.add_argument('--wave', default='db8', type=str, help='Type of pywavelet')
    parser.add_argument('--nlevel', default=1, type=int, help="Number of levels of wavelet decomposition")
    parser.add_argument('--transform', default=True, action='store_false', help='Whether to perform non-linear transform in the approx and detail coefficeints')
    parser.add_argument('--learn_wave', default=True, action='store_true', help='Whether to learn the wavelet transform')
    
    parser.add_argument('--num_classes', default=1, type=int, help="Output dimensionality of model")
    parser.add_argument('--nonlinearity', default='relu', type=str, help='Non lienarity to use')
    
    # training arguments
    parser.add_argument('--train_bs', default=1, type=int, help='Batchsize for train loader')
    parser.add_argument('--valid_bs', default=1, type=int, help='Batchsize for valid loader')
    parser.add_argument('--test_bs', default=1, type=int, help='Batchsize for test loader')
    parser.add_argument('--epoch', default=1000, type=int, help='Number of epochs to train')
    parser.add_argument('--lr', default=0.01, type=float, help="Learning rate for the BCR_DE model")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nonlin = 'lwave'            # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss', 'lwave'
    niters = 2000 # Number of SGD iterations
    learning_rate = 1e-3        # Learning rate. 
    
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3 
    
    tau = 3e1                   # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2               # Readout noise (dB)
    
    # Gabor filter constants.
    # We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
    omega0 = 4.0           # Frequency of sinusoid
    sigma0 = 4.0           # Sigma of Gaussian
    
    # Network parameters
    hidden_layers = 2       # Number of hidden layers in the MLP
    hidden_features = 256   # Number of hidden units per layer
    maxpoints = 256*256     # Batch size
    
    # Read image and scale. A scale of 0.5 for parrot image ensures that it
    # fits in a 12GB GPU
    im = utils.normalize(plt.imread('data/parrot.png').astype(np.float32), True)
    im = cv2.resize(im, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
    H, W, C = im.shape
    im = np.vstack( (np.zeros((1,W,C), dtype=np.float32), im))
    H, W, C = im.shape
    
    # Create a noisy image
    im_noisy = utils.measure(im, noise_snr, tau)
    
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
        
        if tau < 100:
            sidelength = int(max(H, W)/3)
        else:
            sidelength = int(max(H, W))
    elif nonlin == 'lwave':
        args = parse_args()
        arg_dict = vars(args)
        # set noise channel
        args.noise_channel = True
        if args.noise_channel:
                inchannel = 3
        else:
                inchannel = 2
        posencode = False
            
    else:
        posencode = False
        sidelength = H

    if nonlin == 'lwave':
        model = waveletNet(nlevel=args.nlevel, wave=args.wave, inchannel=inchannel, outchannel=3,
                                                learnable_wave=args.learn_wave, transform=args.transform, mode='periodization').to(device)
        # Send model to CUDA
        model.cuda()
        # loss criterion
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

        torch_image = transforms.ToTensor()(im_noisy).unsqueeze(0).float()
        grid_image = get_mgrid((H,W), noise_channel=args.noise_channel, dim=2).unsqueeze(0)
        dataset = INR_Single_Image_Dataset(grid_image, torch_image)
        dataloader = DataLoader(dataset, batch_size=args.train_bs)
        
    else:
        model = models.get_INR(
                         nonlin=nonlin,
                         in_features=2,
                         out_features=3, 
                         hidden_features=hidden_features,
                         hidden_layers=hidden_layers,
                         first_omega_0=omega0,
                         hidden_omega_0=omega0,
                         scale=sigma0,
                         pos_encode=posencode,
                         sidelength=sidelength)
        # Send model to CUDA
        model.cuda()
        # Create an optimizer
        optim = torch.optim.Adam(lr=learning_rate*min(1, maxpoints/(H*W)),
                                 params=model.parameters())
        # Schedule to reduce lr to 0.1 times the initial rate in final epoch
        scheduler = LambdaLR(optim, lambda x: 0.1**min(x/niters, 1))
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        
        X, Y = torch.meshgrid(x, y, indexing='xy')
        coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
        
    
    print('Number of parameters: ', utils.count_parameters(model))
    print('Input PSNR: %.2f dB'%utils.psnr(im, im_noisy))
    
    
    gt = torch.tensor(im).cuda().permute(2,0,1).unsqueeze(0)
    gt_noisy = torch.tensor(im_noisy).cuda().permute(2,0,1).unsqueeze(0)
    
    mse_array = torch.zeros(niters, device='cuda')
    mse_loss_array = torch.zeros(niters, device='cuda')
    time_array = torch.zeros_like(mse_array)
    
    best_mse = torch.tensor(float('inf'))
    best_img = None
    
    rec = torch.zeros_like(gt)
    
    tbar = tqdm(range(niters))
    init_time = time.time()
    for epoch in tbar:
        if nonlin == 'lwave':
            for inp_grid, true_image in dataloader:
                inp_grid= inp_grid.to(device)
                true_image = true_image.to(device)
                rec = model(inp_grid, verbose=False)
                loss = loss_fn(rec, true_image)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:

            indices = torch.randperm(H*W)
            
            for b_idx in range(0, H*W, maxpoints):
                b_indices = indices[b_idx:min(H*W, b_idx+maxpoints)]
                b_coords = coords[:, b_indices, ...].cuda()
                b_indices = b_indices.cuda()
                pixelvalues = model(b_coords)
                
                with torch.no_grad():
                    rec[:, b_indices, :] = pixelvalues
    
                loss = ((pixelvalues - gt_noisy[:, b_indices, :])**2).mean() 
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()
        
        time_array[epoch] = time.time() - init_time
        
        with torch.no_grad():
            mse_loss_array[epoch] = ((gt_noisy - rec)**2).mean().item()
            mse_array[epoch] = ((gt - rec)**2).mean().item()
        
            psnrval = -10*torch.log10(mse_array[epoch])
            tbar.set_description('%.1f'%psnrval)
            tbar.refresh()
        
        
        imrec = rec[0, ...].permute(1, 2, 0).detach().cpu().numpy()
        
        # im = Image.fromarray(imrec[..., ::-1])
        # im.save(f'results/epoch_{epoch}.png')
        # cv2.imshow('Reconstruction', imrec[..., ::-1])            
        # cv2.waitKey(1)
    
        if (mse_array[epoch] < best_mse) or (epoch == 0):
            best_mse = mse_array[epoch]
            best_img = imrec
    
    if posencode:
        nonlin = 'posenc'
        
    mdict = {'rec': best_img,
             'gt': im,
             'im_noisy': im_noisy,
             'mse_noisy_array': mse_loss_array.detach().cpu().numpy(), 
             'mse_array': mse_array.detach().cpu().numpy(),
             'time_array': time_array.detach().cpu().numpy()}
    
    os.makedirs('results/denoising', exist_ok=True)
    io.savemat('results/denoising/%s.mat'%nonlin, mdict)

    print('Best PSNR: %.2f dB'%utils.psnr(im, best_img))
