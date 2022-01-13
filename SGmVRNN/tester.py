from time import *
import torch
import os
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from logger import Logger

class Tester(object):
    def __init__(self, model, device, test, testloader, log_path='log_tester', log_file='loss',
                 learning_rate=0.0002, nsamples=None, checkpoints=None):
        self.model = model
        self.model.to(device)
        self.device = device
        self.test = test
        self.testloader = testloader
        self.log_path = log_path
        self.log_file = log_file
        self.learning_rate = learning_rate
        self.nsamples = nsamples
        self.checkpoints = checkpoints
        self.start_epoch = 0
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.epoch_losses = []
        self.logger = Logger(self.log_path, self.log_file)         
        self.loss = {}


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
            print ("No Checkpoint Exists At '{}', Starting Fresh Training".format(self.checkpoints+'_epochs{}.pth'.format(start_ep)))
            self.start_epoch = 0

    def model_test(self):
        self.model.eval()
        print(self.model.temperature)
        # To collect the latent variable z at timestep T, as well as the corresponding max cate variable index
        for i, dataitem in enumerate(self.testloader,1):
            timestamps,labels,data = dataitem
            data = data.to(self.device)
            z_mean_post, z_logvar_post, z, z_mean_prior, z_logvar_prior, x_mu, x_logsigma, pi, logits, posterior_probs = self.forward_test(data)
            last_timestamp = timestamps[-1,-1,-1,-1]
            label_last_timestamp_tensor = labels[-1,-1,-1,-1]
            anomaly_index = (label_last_timestamp_tensor.numpy() == 1)
            anomaly_nums = len(label_last_timestamp_tensor.numpy()[anomaly_index])
            if anomaly_nums >= 1:
                isanomaly = "Anomaly"
            else:
                isanomaly = "Normaly"
            llh_last_timestamp = self.loglikelihood_last_timestamp(data[-1,-1,-1,:,-1], x_mu[-1,-1,-1,:,-1],x_logsigma[-1,-1,-1,:,-1])
            
            self.loss['Last_timestamp']=last_timestamp.item()
            self.loss['Llh_Lt'] = llh_last_timestamp.item()
            self.loss['IA'] = isanomaly
            self.logger.log_tester(self.start_epoch, self.loss)
        
        print ("Testing is complete!")
            
    def forward_test(self, data):
        with torch.no_grad():
            z_mean_post, z_logvar_post, z, z_mean_prior, z_logvar_prior, x_mu, x_logsigma, pi, logits, posterior_probs = self.model(data)
            return z_mean_post, z_logvar_post, z, z_mean_prior, z_logvar_prior, x_mu, x_logsigma, pi, logits, posterior_probs

    def loglikelihood_last_timestamp(self, x, recon_x_mu, recon_x_logsigma):
        llh = -0.5 * torch.sum(torch.pow(((x.float()-recon_x_mu.float())/torch.exp(recon_x_logsigma.float())), 2) + 2 * recon_x_logsigma.float() + np.log(np.pi*2))
        return llh

    
def main():
    parser = argparse.ArgumentParser()
    # GPU option
    parser.add_argument('--gpu_id', type=int, default=1)
    # Dataset options
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
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
    parser.add_argument('--start_epoch', type=int, default=20)
    parser.add_argument('--checkpoints_path', type=str, default='model')
    parser.add_argument('--checkpoints_file', type=str, default='')
    parser.add_argument('--checkpoints_interval', type=int, default=1)
    parser.add_argument('--log_path', type=str, default='log_tester')
    parser.add_argument('--log_file', type=str, default='') 
 
    # Testing option
    parser.add_argument('--nsamples', type=int, default=1)
    args = parser.parse_args()
    
    # Set up GPU
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device('cuda:%d' % args.gpu_id)
    else:
        device = torch.device('cpu')
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if not os.path.exists(args.dataset_path):
        raise ValueError('Unknown dataset path: {}'.format(args.dataset_path))

    if not os.path.exists(args.checkpoints_path):
        raise ValueError('Unknown checkpoints path: {}'.format(checkpoints_path))

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
            args.log_file = 'catdim{}_zdim{}_cdim{}_hdim{}_winsize{}_T{}_l{}_epochs{}_loss'.format(
                             args.categorical_dims,
                             args.z_dims,
                             args.conv_dims,
                             args.hidden_dims,
                             args.win_size,
                             args.T,
                             args.l,
                             args.start_epoch)
    
    else:
        raise ValueError('Unknown encoder or decoder: {}'.format(args.enc_dec))

    kpi_value_test = KpiReader(args.dataset_path)
    test_loader = torch.utils.data.DataLoader(kpi_value_test, 
                                              batch_size  = args.batch_size, 
                                              shuffle     = False, 
                                              num_workers = args.num_workers)

    sgmvrnn = SGmVRNN(cate_dim = args.categorical_dims,  
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

    tester = Tester(sgmvrnn,device,kpi_value_test,test_loader,
                    log_path       = args.log_path,
                    log_file       = args.log_file,
                    learning_rate  = args.learning_rate,
                    nsamples       = args.nsamples,
                    checkpoints    = os.path.join(args.checkpoints_path,args.checkpoints_file))
    tester.load_checkpoint(args.start_epoch)
    tester.model_test()

if __name__ == '__main__':
    main()
