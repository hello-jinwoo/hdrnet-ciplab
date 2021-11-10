import os
import sys
from test import test
from tqdm import tqdm
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

# from dataset import HDRDataset
from dataset import MultiDataset
from metrics import psnr
from model import HDRPointwiseNN
from utils import load_image, save_params, get_latest_ckpt, load_params

epsilon = 1e-8

def train(params=None):
    writer = SummaryWriter("runs/" + params['ver'] + '/')

    os.makedirs(os.path.join(params['ckpt_path'], params['ver']), exist_ok=True)

    device = torch.device("cuda")

    max_value = 255.

    train_dataset = MultiDataset(params['dataset'], 'train', 'rgb', 'rgb')
    val_dataset = MultiDataset(params['dataset'], 'val', 'rgb', 'rgb')
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4)

    model = HDRPointwiseNN(params=params)
    model = torch.nn.DataParallel(model)
    ckpt = None
    if params['ckpt_file']:
        ckpt = params['ckpt_file']
    elif params['ckpt_path']:
        ckpt = get_latest_ckpt(params['ckpt_path'] + "/" + params['ver'], order_by='time')
    if ckpt:
        print('Loading previous state:', ckpt)
        state_dict = torch.load(ckpt)
        state_dict,_ = load_params(state_dict)
        model.load_state_dict(state_dict)
    model.to(device)
    
    mseloss = torch.nn.MSELoss(reduction='mean')
    cos = torch.nn.CosineSimilarity(dim=1)
    optimizer = Adam(model.parameters(), params['lr'])
    # optimizer = SGD(model.parameters(), params['lr'], momentum=0.9)

    prev_e = 0
    if ckpt:
        prev_e = int(ckpt.split('_')[-2])
    count = 0

    val_min_loss = 1000
    val_min_mae = 100
    val_max_psnr = 0
    val_min_cos = 1


    for e in range(params['epochs']):
        e = e + prev_e
        model.train()
        print("Train...")
        for i, (low, full, target, _, _) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            low = low.to(device)
            full = full.to(device)
            t = target.to(device)

            res = model(low, full)
            
            mse_loss = mseloss(res, t)
            cos_loss = 1 - cos(res, t).mean()
            # total_loss = mse_loss + cos_loss
            total_loss = mse_loss + 0.5 * cos_loss
            _psnr = float(psnr(res, t).item())

            writer.add_scalar("Loss/train_mse", mse_loss, count)
            writer.add_scalar("Loss/train_cos", cos_loss, count)
            writer.add_scalar("Loss/train_total_loss", total_loss, count)
            writer.add_scalar("Loss/train_psnr", _psnr, count)

            total_loss.backward()

            if (count+1) % params['log_interval'] == 0:
                loss = total_loss.item()
                print(e , "epoch,", count, "step")
                print("loss:", loss, "\tpsnr:", _psnr)
            
            optimizer.step()
            count += 1

        print("Validation...")
        val_psnr = 0
        val_count = 0
        val_mse_loss = 0
        val_cos_loss = 0
        model.eval()
        for i, (low_val, full_val, target_val, _, _) in enumerate(val_loader):
            low_val = low_val.to(device)
            full_val = full_val.to(device)
            t_val = target_val.to(device)
            
            res_val = model(low_val, full_val)

            val_psnr += float(psnr(res_val, t_val).item())
            val_count += 1

            val_mse_loss += float(mseloss(res_val, t_val))
            val_cos_loss += float(1 - cos(res_val, t_val).mean())
            

            if e % 20 == 0:
                # RGB to BGR
                for sample_i in range(1):
                    img_t = (t_val.cpu().detach().numpy()).transpose(0,2,3,1)[sample_i] * max_value
                    img_res = (res_val.cpu().detach().numpy()).transpose(0,2,3,1)[sample_i] * max_value

                    # RGB to BGR
                    img_t = img_t[:, :, ::-1]
                    img_res = img_res[:, :, ::-1]
                    # save images
                    cv2.imwrite("train_result_samples/" + params['ver'] + "/" + str(e) + "_" + str(count) + "_" + str(sample_i) + "_target.jpg", img_t)
                    cv2.imwrite("train_result_samples/" + params['ver'] + "/" + str(e) + "_" + str(count) + "_" + str(sample_i) + "_result.jpg", img_res)
        
        val_psnr /= val_count
        val_cos_loss /= val_count
        val_loss = float(val_mse_loss + 0.5 * val_cos_loss) / val_count

        print()
        print("=== Epoch", e, "Validation ===")
        print("loss:", val_loss, "\tpsnr:", val_psnr)
        best_loss = val_loss < val_min_loss
        best_psnr = val_psnr > val_max_psnr
        best_cos = val_cos_loss < val_min_cos

        # save model
        model.eval().cpu()
        state = save_params(model.state_dict(), params)
        if best_loss and best_cos and best_psnr:
            ckpt_model_filename = "ckpt_"+str(e)+'_' + str(count) + "_all.pth"
            ckpt_model_path = os.path.join(params['ckpt_path'], params['ver'], ckpt_model_filename)
            torch.save(state, ckpt_model_path)
            print("Loss decrease!", val_min_loss, "->", val_loss)
            print("Cos decrease!", val_min_cos, "->", val_cos_loss)
            print("PSNR increase!", val_max_psnr, "->", val_psnr)
            print("ckpt saved:", ckpt_model_filename)
            val_min_loss = val_loss
            val_min_cos = val_cos_loss
            val_max_psnr = val_psnr
        elif best_cos and best_loss:
            ckpt_model_filename = "ckpt_"+str(e)+'_' + str(count) + "_loss&cos.pth"
            ckpt_model_path = os.path.join(params['ckpt_path'], params['ver'], ckpt_model_filename)
            torch.save(state, ckpt_model_path)
            print("Loss decrease!", val_min_loss, "->", val_loss)
            print("Cos decrease!", val_min_cos, "->", val_cos_loss)
            print("ckpt saved:", ckpt_model_filename)
            val_min_loss = val_loss
            val_min_cos = val_cos_loss
        elif best_psnr and best_loss:
            ckpt_model_filename = "ckpt_"+str(e)+'_' + str(count) + "_loss&psnr.pth"
            ckpt_model_path = os.path.join(params['ckpt_path'], params['ver'], ckpt_model_filename)
            torch.save(state, ckpt_model_path)
            print("Loss decrease!", val_min_loss, "->", val_loss)
            print("PSNR increase!", val_max_psnr, "->", val_psnr)
            print("ckpt saved:", ckpt_model_filename)
            val_max_psnr = val_psnr
            val_min_loss = val_loss
        elif best_cos and best_psnr:
            ckpt_model_filename = "ckpt_"+str(e)+'_' + str(count) + "_cos&psnr.pth"
            ckpt_model_path = os.path.join(params['ckpt_path'], params['ver'], ckpt_model_filename)
            torch.save(state, ckpt_model_path)
            print("Cos decrease!", val_min_cos, "->", val_cos_loss)
            print("PSNR increase!", val_max_psnr, "->", val_psnr)
            print("ckpt saved:", ckpt_model_filename)
            val_max_psnr = val_psnr
            val_min_cos = val_cos_loss
        elif best_loss:
            ckpt_model_filename = "ckpt_"+str(e)+'_' + str(count) + "_loss.pth"
            ckpt_model_path = os.path.join(params['ckpt_path'], params['ver'], ckpt_model_filename)
            torch.save(state, ckpt_model_path)
            print("Loss decrease!", val_min_loss, "->", val_loss)
            print("ckpt saved:", ckpt_model_filename)
            val_min_loss = val_loss
        elif best_cos:
            ckpt_model_filename = "ckpt_"+str(e)+'_' + str(count) + "_cos.pth"
            ckpt_model_path = os.path.join(params['ckpt_path'], params['ver'], ckpt_model_filename)
            torch.save(state, ckpt_model_path)
            print("Cos decrease!", val_min_cos, "->", val_cos_loss)
            print("ckpt saved:", ckpt_model_filename)
            val_min_cos = val_cos_loss
        elif best_psnr:
            model.eval().cpu()
            ckpt_model_filename = "ckpt_"+str(e)+'_' + str(count) + "_psnr.pth"
            ckpt_model_path = os.path.join(params['ckpt_path'], params['ver'], ckpt_model_filename)
            state = save_params(model.state_dict(), params)
            torch.save(state, ckpt_model_path)
            model.to(device).train()
            print("PSNR increase!", val_max_psnr, "->", val_psnr)
            print("ckpt saved:", ckpt_model_filename)
            val_max_psnr = val_psnr
        
        model.to(device).train()
        print()
        print()


    # writer.flush()
    writer.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HDRNet Inference')
    parser.add_argument('--ckpt-path', type=str, default='./ch', help='Model checkpoint path')
    parser.add_argument('--ckpt-file', type=str, default=None, help='Model checkpoint file')
    parser.add_argument('--test-image', type=str, dest="test_image", help='Test image path')
    parser.add_argument('--test-out', type=str, default='out.png', dest="test_out", help='Output test image path')

    parser.add_argument('--luma-bins', type=int, default=8)
    parser.add_argument('--channel-multiplier', default=1, type=int)
    parser.add_argument('--spatial-bin', type=int, default=16)
    parser.add_argument('--batch-norm', action='store_true', help='If set use batch norm')
    parser.add_argument('--net-input-size', type=int, default=128, help='Size of low-res input')
    parser.add_argument('--net-output-size', type=int, default=256, help='Size of full-res input/output')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='', help='Dataset path with input/output dirs', required=True)
    parser.add_argument('--dataset-suffix', type=str, default='', help='Add suffix to input/output dirs. Useful when train on different dataset image sizes')
    
    parser.add_argument('--ver', type=str, default='', help='Experiment version information')
    
    

    params = vars(parser.parse_args())

    print('PARAMS:')
    print(params)

    train(params=params)

