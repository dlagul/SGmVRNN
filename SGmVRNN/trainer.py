import torch
import os
import time
import argparse
import torchvision
import torch.utils.data
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from model import *
from tqdm import *
from util import KpiReader
from logger import Logger

class Trainer(object):
    def __init__(self, model, train, trainloader, log_path='log_trainer',
                 log_file='loss', epochs=20, batch_size=1024, learning_rate=0.001,
                 checkpoints='kpi_model.path', checkpoints_interval = 1, device=torch.device('cuda:0')):
        self.trainloader = trainloader
        self.train = train
        self.log_path = log_path
        self.log_file = log_file
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.checkpoints_interval = checkpoints_interval
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.epoch_losses = []
        self.loss = {}
        self.logger = Logger(self.log_path, self.log_file)
 
    def save_checkpoint(self, epoch):
        torch.save({'epoch': epoch + 1,
                    'temperature': self.model.temperature,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'losses': self.epoch_losses},
                    self.checkpoints + '_epochs{}.pth'.format(epoch+1))

    def load_checkpoint(self, start_ep):
        try:
            print ("Loading Chechpoint from ' {} '".format(self.checkpoints+'_epochs{}.pth'.format(start_ep)))
            checkpoint = torch.load(self.checkpoints+'_epochs{}.pth'.format(start_ep))
            self.start_epoch = checkpoint['epoch']
            self.model.temperature = checkpoint['temperature']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print ("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print ("No Checkpoint Exists At '{}', Starting Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def train_model(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.epochs):
            losses = []
            llhs = []
            kld_zs = []
            kld_pis = []
            print ("Running Epoch : {}".format(epoch+1))
            for i, dataitem in tqdm(enumerate(self.trainloader,1)):
                _,_,data = dataitem
                data = data.to(self.device)
                self.optimizer.zero_grad()
                z_mean_post, z_logvar_post, z, z_mean_prior, z_logvar_prior, x_mu, x_logsigma, pi, logits, posterior_probs = self.model(data)
                loss, llh, kld_z, kld_pi = self.model.loss_fn(data, z, z_mean_post, z_logvar_post, z_mean_prior, z_logvar_prior,
                                                              x_mu, x_logsigma, pi, logits, posterior_probs)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                llhs.append(llh.item())
                kld_zs.append(kld_z.item())
                kld_pis.append(kld_pi.item())
            meanloss = np.mean(losses)
            meanllh = np.mean(llhs)
            meanz = np.mean(kld_zs)
            meanpi = np.mean(kld_pis)
            self.epoch_losses.append(meanloss)
            print ("Epoch {} : Average Loss: {} Loglikelihood: {} KL of z: {} KL of pi: {} Temp: {}".format(
                    epoch+1, meanloss, meanllh, meanz, meanpi, self.model.temperature))
            self.loss['Epoch'] = epoch+1
            self.loss['Avg_loss'] = meanloss
            self.loss['Llh'] = meanllh
            self.loss['KL_pi'] = meanpi
            self.loss['KL_z'] = meanz
            self.loss['Temp'] = self.model.temperature
            self.logger.log_trainer(epoch+1, self.loss)
            if (self.checkpoints_interval > 0
                and (epoch+1)  % self.checkpoints_interval == 0):
                self.save_checkpoint(epoch)

            # 
            if (epoch+1) % 1 == 0:
                self.model.temperature = np.maximum(self.model.temperature * np.exp(-self.model.anneal_rate * (epoch+1)), self.model.min_temperature)
                print("New Model Temperature: {}".format(self.model.temperature))
        print ("Training is complete!")


def main():
    parser = argparse.ArgumentParser()
    # GPU option
    parser.add_argument('--gpu_id', type=int, default=0)
    # Dataset options
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--win_size', type=int, default=1)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--n', type=int, default=36)

    # Model options
    parser.add_argument('--categorical_dims', type=int, default=5)
    parser.add_argument('--z_dims', type=int, default=10)
    parser.add_argument('--conv_dims', type=int, default=20)
    parser.add_argument('--hidden_dims', type=int, default=20)
    parser.add_argument('--enc_dec', type=str, default='CNN')

    # Training options
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--min_temperature', type=float, default=0.1)
    parser.add_argument('--anneal_rate', type=float, default=0.1)
    parser.add_argument("--hard_gumbel", action='store_true', help='hard gumbel or not')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--checkpoints_path', type=str, default='model')
    parser.add_argument('--checkpoints_file', type=str, default='')
    parser.add_argument('--checkpoints_interval', type=int, default=1)
    parser.add_argument('--log_path', type=str, default='log_trainer')
    parser.add_argument('--log_file', type=str, default='')

    args = parser.parse_args()

    # Set up GPU
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device('cuda:%d' % args.gpu_id)
    else:
        device = torch.device('cpu')
    
    if not os.path.exists(args.dataset_path):
        raise ValueError('Unknown dataset path: {}'.format(args.dataset_path))

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)
    if args.enc_dec == 'CNN': 
        if args.checkpoints_file == '':
            args.checkpoints_file = 'catdim{}_zdim{}_cdim{}_hdim{}_winsize{}_T{}_l{}'.format(
                                    args.categorical_dims,
                                    args.z_dims,
                                    args.conv_dims,
                                    args.hidden_dims,
                                    args.win_size,
                                    args.T,
                                    args.l)
        if args.log_file == '':
            args.log_file = 'catdim{}_zdim{}_cdim{}_hdim{}_winsize{}_T{}_l{}_loss'.format(
                             args.categorical_dims,
                             args.z_dims,
                             args.conv_dims,
                             args.hidden_dims,
                             args.win_size,
                             args.T,
                             args.l)
    else:
        raise ValueError('Unknown encoder or decoder: {}'.format(args.enc_dec))
    kpi_value_train = KpiReader(args.dataset_path)
    train_loader = torch.utils.data.DataLoader(kpi_value_train, 
                                               batch_size  = args.batch_size, 
                                               shuffle     = True, 
                                               num_workers = args.num_workers)

    sgmvrnn = SGmVRNN(cate_dim        = args.categorical_dims, 
                      z_dim           = args.z_dims, 
                      conv_dim        = args.conv_dims, 
                      hidden_dim      = args.hidden_dims, 
                      T               = args.T, 
                      w               = args.win_size, 
                      n               = args.n,
                      temperature     = args.temperature,
                      min_temperature = args.min_temperature, 
                      anneal_rate     = args.anneal_rate,
                      hard_gumbel     = args.hard_gumbel, 
                      enc             = args.enc_dec,  
                      dec             = args.enc_dec,
                      device          = device)
    trainer = Trainer(sgmvrnn, kpi_value_train, train_loader, 
                      log_path              = args.log_path, 
                      log_file              = args.log_file, 
                      batch_size            = args.batch_size, 
                      epochs                = args.epochs,
                      learning_rate         = args.learning_rate,
                      checkpoints           = os.path.join(args.checkpoints_path,args.checkpoints_file), 
                      checkpoints_interval  = args.checkpoints_interval, device=device)
    trainer.load_checkpoint(args.start_epoch)
    trainer.train_model()

if __name__ == '__main__':
    main() 
