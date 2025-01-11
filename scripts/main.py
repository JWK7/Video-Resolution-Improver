import torch.nn as nn
import torch
import numpy as np

import sys
import os
sys.path.append(os.getcwd())

from modeling.DCGAN import Generator, Discriminator
from utility.online_load import load_gif
from utility.sample import sample

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def rollout_disc(gen: nn.Module,disc: nn.Module,gen_opt:nn.Module,disc_opt:nn.Module,url: str,image_shape: tuple) -> None:
    criterion = nn.BCEWithLogitsLoss()
    data = load_gif(url = url).astype(np.float32).transpose(1,2,0)
    data = cv2.resize(data,(64,32))

    #Discriminator update
    disc_opt.zero_grad()
    data_before = data[:,:,:-2]
    data_after = data[:,:,2:]
    data_current = data[:,:,1:-1]

    gen_input  = torch.tensor( np.array([data_before,data_after]).transpose(3,0,1,2))
    gen_out = gen(gen_input).detach()
    gen_disc_input = torch.cat((gen_input,gen_out),1)
    gen_disc_out = disc(gen_disc_input)

    real_disc_input = torch.tensor( np.array([data_before,data_after,data_current]).transpose(3,0,1,2))
    real_disc_out = disc(real_disc_input)

    gen_disc_loss = criterion(gen_disc_out, torch.zeros_like(gen_disc_out))
    real_disc_loss = criterion(real_disc_out,torch.ones_like(real_disc_out)) 
    avg_disc_loss = ( gen_disc_loss + real_disc_loss ) / 2

    avg_disc_loss.backward(retain_graph=True)
    disc_opt.step()

    #Generator update
    gen_opt.zero_grad()
    gen_out = gen(gen_input)
    gen_disc_input = torch.cat((gen_input,gen_out),1)
    gen_disc_out = disc(gen_disc_input)
    gen_disc_loss = criterion(gen_disc_out, torch.zeros_like(gen_disc_out))
    gen_disc_loss.backward(retain_graph=True)
    gen_opt.step()

    return gen_disc_loss.mean().detach(), avg_disc_loss.mean().detach()



def rollout_gen(gen: nn.Module,disc: nn.Module,gen_opt:nn.Module,disc_opt:nn.Module,url: str,image_shape: tuple) -> None:
    pass

def train() -> None:
    batch_size = 5
    lr = 0.0002
    image_shape = 100,150

    beta_1 = 0.5 
    beta_2 = 0.999
    device = 'cpu'

    gen = Generator(2,1).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc = Discriminator(3).to(device) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))


    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)


    n_epochs = 50
    gen_loss_list = []
    disc_loss_list = []
    for epoch in tqdm(range(n_epochs)):
        for sample_url in sample(data_directory = "data/train.txt",sample_size = batch_size):
            gen_loss, disc_loss = rollout_disc(gen,disc,gen_opt,disc_opt,sample_url,image_shape)
            gen_loss_list.append(gen_loss)
            disc_loss_list.append(disc_loss)
    

    plt.plot(gen_loss_list)
    plt.savefig("Gen_Loss.png")

if __name__ == "__main__":
    train()

