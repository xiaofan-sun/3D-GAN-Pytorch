import argparse
import os
import time
import scipy.io
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter
from models.GAN3D import Discriminator, Generator, Predictor
from utils import get_vox_from_binvox, get_voxel_bbox, data_save, crop_voxel, recover_voxel
import binvox_rw

parser = argparse.ArgumentParser()
parser.add_argument("--num_epoch", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=22, help="size of the batches")
parser.add_argument("--discriminator_steps", type=int, default=1, help="Number of discriminator updates to do for each generator update.")
parser.add_argument("--generator_steps", type=int, default=1, help="Number of generator updates to do for each generator update.")
parser.add_argument("--g_lr", type=float, default=0.00006, help="adam: learning rate")
parser.add_argument("--d_lr", type=float, default=0.0000015, help="adam: learning rate")
parser.add_argument("--data_step", type=int, default=20, help="the epochs of saving data")
parser.add_argument("--model_step", type=int, default=100, help="the epochs of saving model")

parser.add_argument('--g_out_dim', type=int, default=256)
parser.add_argument('--g_in_channels', type=int, default=512)
parser.add_argument('--g_out_channels', type=int, default=1)
parser.add_argument('--noise_dim', type=int, default=256, help='Dimension of input z to the generator. ')
parser.add_argument('--d_dim', type=int, default=256)
parser.add_argument('--d_in_channels', type=int, default=1)
parser.add_argument('--d_out_conv_channels', type=int, default=512)
parser.add_argument('--mask_margin', default=(13,14,12))

parser.add_argument('--dir', type=str, help='Path to train result')
parser.add_argument('--train_data', default="../data_all/", type=str, help='Path to train data')

args = parser.parse_args()
print(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def dataloader(data_folder, mask_margin):
    file_list = os.listdir(data_folder)
    print('file_list:',file_list)
    data = []
    for data_name in file_list:
        print(data_name,end='  ')
        tmp_raw = get_vox_from_binvox(os.path.join(data_folder,data_name+"/model_depth_fusion_512.binvox")).astype(np.uint8)
        xmin,xmax,ymin,ymax,zmin,zmax = get_voxel_bbox(tmp_raw) # 13 243 70 186 116 140
        tmp = crop_voxel(tmp_raw, mask_margin, xmin,xmax,ymin,ymax,zmin,zmax)
        if tmp.shape==(256, 144, 48):
            data.append(tmp.astype(np.float32))
        else:
            print(' ')
    data = np.array(data)
    print("data:",data.shape)
    return data, xmin,xmax,ymin,ymax,zmin,zmax
 
def train(generator, discriminator, data, attribute, g_lr, d_lr, num_epoch, discriminator_steps, generator_steps, batch_size, noise_dim, model_pth, gen_data_folder, data_step, model_step):
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    loss_d = []
    loss_g = []
    
    adv_loss = nn.BCELoss()
    aux_loss = nn.CrossEntropyLoss().to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=g_lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=d_lr)

    print("start training......")
    batch_num = len(data)/batch_size
    if batch_num % 1 != 0 :
        batch_num = int(batch_num) + 1 
    else:
        batch_num = int(batch_num)
    d_loss = None
    g_loss = None
    train_data_list = []
    for epoch in range(num_epoch):
        start_time = time.time()
        for batch in range(batch_num):
            real_data = data[batch_size*batch:batch_size*(batch+1)]
            real_data = torch.Tensor(real_data).to(device).unsqueeze(1)
            real_att = attribute[batch_size*batch:batch_size*(batch+1)]
            real_att = torch.Tensor(attribute).to(device).unsqueeze(1)
            
            # Train discriminator
            for k in range(discriminator_steps):
                z = torch.randn(real_data.shape[0], noise_dim).to(device)
                fake_att = torch.rand(real_data.shape[0])
                fake_data = generator(z, fake_att)                
                optimizer_d.zero_grad()
                res_fake_adv, res_fake_aux = discriminator(fake_data)
                res_real_adv, res_real_aux = discriminator(real_data)
                if k == 0 and batch==batch_num-1:
                    print(f"fake_sum:{torch.sum(fake_data).view(-1).cpu().detach().numpy().tolist(), torch.sum(res_fake_adv).view(-1).cpu().detach().numpy().tolist()}, real_sum:{torch.sum(real_data).view(-1).cpu().detach().numpy().tolist(), torch.sum(res_real_adv).view(-1).cpu().detach().numpy().tolist()}",end="  ")
                
                loss_adv_real = adv_loss(res_real_adv, torch.ones_like(res_real_adv))
                loss_adv_fake = adv_loss(res_fake_adv, torch.zeros_like(res_fake_adv))
                loss_aux_real = aux_loss(res_real_aux, real_att)
                loss_aux_fake = aux_loss(res_fake_aux, fake_att)

                d_loss = (loss_adv_real + loss_adv_fake)*0.5 + (loss_aux_real + loss_aux_fake)*0.5
                # d_loss = (loss_adv_real + loss_adv_fake)*0.5 + loss_aux_real  # TODO
                
                d_loss.backward()
                optimizer_d.step()

            # Train generator
            for k in range(generator_steps):
                optimizer_g.zero_grad()
                z = torch.randn(real_data.shape[0], noise_dim).to(device)
                # for name, param in generator.named_parameters():
                #     print(f"{name} requires gradient: {param.requires_grad}")
                fake_att = torch.rand(real_data.shape[0])
                fake_data = generator(z, fake_att)
                res_fake_adv, res_fake_aux = discriminator(fake_data)
                if (epoch+1)%data_step==0 and batch==(batch_num-1):
                    train_fake_data = torch.squeeze(fake_data[0]).detach().cpu().numpy()
                    train_data_list.append(train_fake_data)
                    scipy.io.savemat(gen_data_folder+'mat_data/'+'gen_data'+str(epoch+1)+'.mat', {'data': train_fake_data, 'att':res_fake_aux})
                    # print("save_data "+ gen_data_folder+'mat_data/'+'gen_data'+str(epoch+1)+'.mat')
                res_adv, res_aux = discriminator(fake_data)
                loss_adv = adv_loss(res_adv, torch.ones_like(res_fake_adv))
                loss_aux = aux_loss(res_aux, torch.ones_like(res_fake_adv))

                g_loss  = loss_aux + loss_adv

                g_loss.backward()
                optimizer_g.step() # outof memory
                
        print(f"Epoch [{epoch+1}/{num_epoch}], time: {(time.time() - start_time):.0f}, G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")
        loss_d.append(d_loss.item())
        loss_g.append(g_loss.item())
        if (epoch+1)%100==0:
            torch.save(generator.state_dict(), model_pth + 'generator/generator'+str(epoch+1)+'.pth')
            torch.save(discriminator.state_dict(), model_pth + 'discriminator/discriminator'+str(epoch+1)+'.pth')
    print("Training completed!")
    return generator, discriminator, loss_d, loss_g, train_data_list

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

if __name__=='__main__':
    noise_dim = args.noise_dim
    batch_size = args.batch_size
    model_pth = './output/'+args.dir
    gen_data_folder = './output/'+args.dir+'gen_data/'
    mask_margin = args.mask_margin

    if not os.path.exists(model_pth+'generator/'):
        os.makedirs(model_pth+'generator/')
    if not os.path.exists(model_pth+'discriminator/'):
        os.makedirs(model_pth+'discriminator/')

    if not os.path.exists(gen_data_folder+'vox_data/'):
        os.makedirs(gen_data_folder+'vox_data/')
    if not os.path.exists(gen_data_folder+'mat_data/'):
        os.makedirs(gen_data_folder+'mat_data/')
    if not os.path.exists(gen_data_folder+'stl_data/'):
        os.makedirs(gen_data_folder+'stl_data/')

    generator = Generator()
    discriminator = Discriminator()
    predictor = Predictor()
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    data, xmin,xmax,ymin,ymax,zmin,zmax = dataloader(args.train_data, mask_margin)
    
    generator, discriminator, loss_d, loss_g, train_data_list = train(generator, discriminator, data, args.g_lr, args.d_lr, args.num_epoch, args.discriminator_steps, args.generator_steps, batch_size, noise_dim, model_pth, gen_data_folder, args.data_step, args.model_step)
    print("len(train_data_list)",len(train_data_list))
    z = torch.randn(2, noise_dim).to(device)
    z_att = torch.ones_like(2)
    gen_data_list = generator(z, z_att).squeeze(1).detach().cpu().numpy()
    gen_data_att = discriminator(gen_data_list).squeeze(1).detach().cpu().numpy()
    for sin_data in gen_data_list:
        train_data_list.append(sin_data)
    tmp_voxel_gen = np.array(train_data_list)
    print("tmp_voxel_gen.shape",tmp_voxel_gen.shape)
    scipy.io.savemat(gen_data_folder+'mat_data/data.mat', {'tmp_voxel_gen': tmp_voxel_gen, 'gen_data_att':gen_data_att, 'loss_d':loss_d, 'loss_g':loss_d})
    