import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math

class ConvUnit1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnit1d, self).__init__()
        self.model = nn.Sequential(
                     nn.Conv1d(in_channels, out_channels, kernel, stride, padding), nonlinearity)
    def forward(self, x):
        return self.model(x)

class ConvUnitTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, out_padding=0, nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnitTranspose1d, self).__init__()
        self.model = nn.Sequential(
                     nn.ConvTranspose1d(in_channels, out_channels, kernel, stride, padding), nonlinearity)
    def forward(self, x):
        return self.model(x)

class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        self.model = nn.Sequential(
                     nn.Linear(in_features, out_features),nonlinearity)
    def forward(self, x):
        return self.model(x)

class EncX(nn.Module):
    '''
    Input: x
        shape: [batch_size, 1, T, n, w]
    Output: x_hat
        shape: [batch_size, 1, T, n, w] 
    Describe: get the summarized x 
    '''
    def __init__(self, enc_dim, enc='CNN', n=38, w=1, T=20):
        super(EncX, self).__init__()
        self.enc = enc
        self.n = n
        self.w = w
        self.T = T
        self.conv_dim = enc_dim
        
        if self.enc == 'CNN' and self.w == 1:
            if self.n == 36:
                k0,k1,k2=3,2,2
                s0,s1,s2=3,2,2
                p0,p1,p2=0,0,0
                s_d=(int(self.n/(k0*k1*k2)))
            elif self.n == 38:
                k0,k1,k2=2,2,2
                s0,s1,s2=2,2,2
                p0,p1,p2=1,0,0
                s_d=(int((self.n+s0*p0)/(k0*k1*k2)))
            else:
                raise ValueError('Invalid kpi numbers: please choose from the set [36,38]')
            self.k0,self.k1,self.k2=k0,k1,k2
            self.s0,self.s1,self.s2=s0,s1,s2
            self.p0,self.p1,self.p2=p0,p1,p2
            self.cd = [32,s_d]
            self.conv = nn.Sequential(
                    ConvUnit1d(1, 8, kernel=self.k0, 
                                    stride=self.s0, 
                                    padding=self.p0), 
                    ConvUnit1d(8, 16, kernel=self.k1,  
                                      stride=self.s1, 
                                      padding=self.p1), 
                    ConvUnit1d(16, 32, kernel=self.k2, 
                                       stride=self.s2, 
                                       padding=self.p2)  
                    )  
            
            self.conv_fc = nn.Sequential(
                       LinearUnit(self.cd[0]*self.cd[1], self.conv_dim*2),
                       LinearUnit(self.conv_dim*2, self.conv_dim))
        else:
            raise ValueError('Unknown encoder and decoder: {}'.format(self.enc))

    def enc_x(self, x):
        if self.enc == 'CNN' and self.w == 1:
            x = x.view(-1, 1, self.n)
            x = self.conv(x)
            x = x.view(-1, self.cd[0]*self.cd[1])
            x = self.conv_fc(x)
            x = x.view(-1, self.T, self.conv_dim)
        else:
            raise ValueError('Unknown encoder and decoder: {}'.format(self.enc))
        return x

    def forward(self, x):
        x_hat =  self.enc_x(x)
        return x_hat


class DecX(nn.Module):
    '''
    Input: z
        shape: [batch_size, 1, z_dim]
    Output: x_mu and x_logsigma
        shape: [batch_size, 1, 1, n, w] 
    Describe: Obtain the paramaters, i.e., mu & logsigma of likelihood function 
    '''
    def __init__(self, enc_dim, dec_init_dim, dec='CNN', n=38, w=1, T=20):
        super(DecX, self).__init__()
        self.dec = dec
        self.n = n
        self.w = w
        self.T = T
        self.conv_dim = enc_dim
        self.dec_init_dim = dec_init_dim
        
        if self.dec == 'CNN' and self.w == 1:
            if self.n == 36:
                k0,k1,k2=3,2,2
                s0,s1,s2=3,2,2
                p0,p1,p2=0,0,0
                s_d=(int(self.n/(k0*k1*k2)))
            elif self.n == 38:
                k0,k1,k2=2,2,2
                s0,s1,s2=2,2,2
                p0,p1,p2=1,0,0
                s_d=(int((self.n+s0*p0)/(k0*k1*k2)))
            else:
                raise ValueError('Invalid kpi numbers: please choose from the set [36,38]')
            self.k0,self.k1,self.k2=k0,k1,k2
            self.s0,self.s1,self.s2=s0,s1,s2
            self.p0,self.p1,self.p2=p0,p1,p2
            self.cd = [32,s_d]
            self.deconv_fc_mu = nn.Sequential(
                         LinearUnit(self.dec_init_dim, self.conv_dim*2),
                         LinearUnit(self.conv_dim*2, self.cd[0]*self.cd[1]))
            self.deconv_mu = nn.Sequential(
                      ConvUnitTranspose1d(32, 16, kernel=self.k2, 
                                                  stride=self.s2, 
                                                  padding=self.p2),  
                      ConvUnitTranspose1d(16, 8, kernel=self.k1, 
                                                 stride=self.s1, 
                                                 padding=self.p1), 
                      ConvUnitTranspose1d(8, 1, kernel=self.k0, 
                                               stride=self.s0, 
                                               padding=self.p0, 
                                               nonlinearity=nn.Tanh()) 
                      ) 
            self.deconv_fc_logsigma = nn.Sequential(
                         LinearUnit(self.dec_init_dim, self.conv_dim*2),
                         LinearUnit(self.conv_dim*2, self.cd[0]*self.cd[1]))
            self.deconv_logsigma = nn.Sequential(
                      ConvUnitTranspose1d(32, 16, kernel=self.k2,                               
                                                  stride=self.s2,                             
                                                  padding=self.p2),  
                      ConvUnitTranspose1d(16, 8, kernel=self.k1,                               
                                                 stride=self.s1,              
                                                 padding=self.p1), 
                      ConvUnitTranspose1d(8, 1, kernel=self.k0,                              
                                               stride=self.s0,    
                                               padding=self.p0,                           
                                               nonlinearity=nn.Tanh()) 
                      )
        else:
            raise ValueError('Unknown decoder: {}'.format(self.dec))

    def dec_x_mu(self, x):
        if self.dec == 'CNN' and self.w == 1:
            x = self.deconv_fc_mu(x)
            x = x.view(-1, self.cd[0], self.cd[1])
            x = self.deconv_mu(x)
            x = x.view(-1, 1, 1, self.n, self.w)
        else:
            raise ValueError('Unknown decoder: {}'.format(self.dec))
        return x
    
 
    def dec_x_logsigma(self, x):
        if self.dec == 'CNN' and self.w == 1:
            x = self.deconv_fc_logsigma(x)
            x = x.view(-1, self.cd[0], self.cd[1])
            x = self.deconv_logsigma(x)
            x = x.view(-1, 1, 1, self.n, self.w)
        else:
            raise ValueError('Unknown  decoder: {}'.format(self.dec))
        return x

    def forward(self, x):
        x_mu =  self.dec_x_mu(x)
        x_logsigma = self.dec_x_logsigma(x)
        return x_mu, x_logsigma

class LossFunctions:
    eps = 1e-8

    def log_normal(self, x, mu, var):
        """Logarithm of normal distribution with mean=mu and variance=var
           log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
           x: (array) corresponding array containing the input
           mu: (array) corresponding array containing the mean 
           var: (array) corresponding array containing the variance

        Returns:
           output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
        """
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)

    def entropy(self, logits, targets):
        """Entropy loss
            loss = (1/n) * -Σ targets*log(predicted)

         Args:
            logits: (array) corresponding array containing the logits of the categorical variable
            real: (array) corresponding array containing the true labels
 
         Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
        """
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))


class ReparameterizeTrick: 
    def reparameterize_gaussian(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    # gumbel-softmax 2nd version
    def sample_gumbel(self, shape, device, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.to(device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, device):
        y = logits + self.sample_gumbel(logits.size(), device, logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, device, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, device)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

# Generation Network
class GenerationNet(nn.Module):
    def __init__(self, hidden_dim, cate_dim, z_dim, enc_dim, dec_init_dim, dec='CNN', T=20, w=1, n=38, device=torch.device('cuda:0')):
        super(GenerationNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.cate_dim = cate_dim
        self.z_dim = z_dim 
        self.enc_dim = enc_dim
        self.dec_init_dim = dec_init_dim
        self.dec = dec
        self.T = T
        self.w = w
        self.n = n 
        self.device = device

        self.rt = ReparameterizeTrick()
       
        self.Pz_hc_mean = nn.Sequential(
             LinearUnit(self.hidden_dim + self.cate_dim, self.hidden_dim),
             LinearUnit(self.hidden_dim, self.z_dim)
        )
        
        self.Pz_hc_logvar = nn.Sequential(
             LinearUnit(self.hidden_dim + self.cate_dim, self.hidden_dim),
             LinearUnit(self.hidden_dim, self.z_dim)
        )

        self.Gen_net = DecX(self.enc_dim, self.dec_init_dim, dec=self.dec, n=self.n, w=self.w, T=self.T)    

    # The Gaussian mixture prior of z
    def Pz_prior(self, h, cate_posterior, batch_size, random_sampling=True):
        '''
        # Input:
            h:
                the state variables of recurrent structure
                shape: [batch_size, T, hidden_dim] 
            cate_posterior: 
                the posterior of latent categorical variables c which are returned from Inference Network
                shape: [batch_size, T, cate_dim]            
        # Output:
            z_mean_prior:
                mu for the prior gaussian of latent variables z
                shape: [batch_size, T, cate_dim, z_dim]
            z_logvar_prior:
                log(sigma^2) for the prior gaussian of latent variables z
                shape: [batch_size, T, cate_dim, z_dim]
        # Describe: Obtain the prior of latent variables z at each time step via the posterior of z and cate variables 
        '''
        z_mean_prior = None
        z_logvar_prior = None

        for t in range(self.T):
            h_cate_posterior_t = torch.cat((h[:,t,:], cate_posterior[:,t,:].view(-1, self.cate_dim)), dim=1) 
            z_mean_prior_t = self.Pz_hc_mean(h_cate_posterior_t) 
            z_logvar_prior_t = self.Pz_hc_logvar(h_cate_posterior_t) 
                
            if z_mean_prior is None:
                z_mean_prior = z_mean_prior_t.unsqueeze(1) 
                z_logvar_prior = z_logvar_prior_t.unsqueeze(1) 
            else:
                z_mean_prior = torch.cat((z_mean_prior, z_mean_prior_t.unsqueeze(1)), dim=1)
                z_logvar_prior = torch.cat((z_logvar_prior, z_logvar_prior_t.unsqueeze(1)), dim=1)
        return z_mean_prior, z_logvar_prior
 
    # P(x_t|z_t)
    def gen_px_hz(self, h, z_posterior, batch_size):
        '''
        # Input:
            h:
                the state variables of recurrent structure
                shape: [batch_size, T, hidden_dim] 
            z_posterior: 
                the posterior of latent variables z which are returned from Inference Network
                shape: [batch_size, T, z_dim]
        # Output:
            x_mu:
                mu for the gaussian of likelihood
                shape: [batch_size, 1, T, n, w]
            x_logsigma:
                log(sigma) for the gaussian of likelihood
                shape: [batch_size, 1, T, n, w]
        # Describe: Obtain the parameters of likelihood at each time step 
        '''
        x_mu = None
        x_logsigma = None
        for t in range(self.T):
            h_t = h[:,t,:] 
            z_posterior_t = z_posterior[:,t,:]
            h_z_t = torch.cat((h_t,z_posterior_t), dim=1) 
            x_mu_t, x_logsigma_t = self.Gen_net(h_z_t)

            if x_mu is None:
                x_mu = x_mu_t
                x_logsigma = x_logsigma_t 
            else:
                x_mu = torch.cat((x_mu,x_mu_t), dim=1)
                x_logsigma = torch.cat((x_logsigma, x_logsigma_t), dim=1) 
        return x_mu, x_logsigma 

    def forward(self, h, z_posterior, cate_posterior):
        z_mean_prior, z_logvar_prior =  self.Pz_prior(h, cate_posterior, z_posterior.size(0))
        x_mu, x_logsigma = self.gen_px_hz(h, z_posterior, z_posterior.size(0))
        return z_mean_prior, z_logvar_prior, x_mu, x_logsigma 

# Inference Network
class InferenceNet(nn.Module):
    def __init__(self, cate_dim, z_dim, hidden_dim, enc_dim, enc='CNN', T=20, w=1, n=38, 
                       hard_gumbel=False, device=torch.device('cuda:0')):
        super(InferenceNet, self).__init__()

        self.cate_dim = cate_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.enc = enc 
        self.T = T
        self.w = w
        self.n = n
        self.device = device
        self.hard_gumbel = hard_gumbel

        self.rt = ReparameterizeTrick() 

        self.enc_x = EncX(self.enc_dim, enc=self.enc, n=self.n, w=self.w, T=self.T)
      
        self.xh_c_layer = nn.Sequential(
            LinearUnit(self.enc_dim + self.hidden_dim, self.hidden_dim),
            LinearUnit(self.hidden_dim, self.cate_dim)
        )

        self.phi_xz = LinearUnit(self.enc_dim + self.z_dim, 2*self.hidden_dim)
        self.rnn_enc = nn.LSTMCell(2*self.hidden_dim, self.hidden_dim, bias=True) 
 
        self.Pz_xhc_mean = nn.Sequential(
            LinearUnit(self.enc_dim + self.hidden_dim + self.cate_dim, self.hidden_dim),
            LinearUnit(self.hidden_dim, self.z_dim)
        )          
        
        self.Pz_xhc_logvar = nn.Sequential(
            LinearUnit(self.enc_dim + self.hidden_dim + self.cate_dim, self.hidden_dim),
            LinearUnit(self.hidden_dim, self.z_dim)
        )

    def infer_qzc_x(self, x, batch_size, temperature):
        '''
        Input: 
            x: the observed samples
                shape: [batch_size, 1, T, n, w]
 
        Output:
            z_posterior: the posterior latent variable z sampled via reparameterization trick 
                shape: [batch_size, T, z_dim]
            z_mean_posterior: the mu of posterior distribution for latent variables z
                shape: [batch_size, T, cate_dim, z_dim] 
            z_logvar_posterior: the log(sigma^2) of posterior distribution for latent variables z
                shape: [batch_size, T, cate_dim, z_dim]
            h_out: hidden state variable of lstm 
                shape: [batch_size, T, hidden_dim]
            cate_posterior: the posterior of latent categorical variable c sample via gumbel-softmax
                shape: [batch_size, T, cate_dim] 
            logits_out: self.h_c_layer(h2_t).view(-1, self.cate_dim) 
                shape: [batch_size, T, cate_dim], where the shape of each logits is [batch_size, cate_dim]
            posterior_prob_out: F.softmax(logits, dim=-1)
                shape: [batch_size, T, cate_dim], where the shape of each logits is [batch_size, cate_dim]

        Describe;: Inference Network for latent variables z and categorical variables c
        '''
        logits_out = None
        posterior_prob_out = None
        
        h_out = None
        cate_posterior = None
        z_posterior = None

        z_mean_posterior = None
        z_logvar_posterior = None
       
        z_posterior_t = torch.zeros(batch_size, self.z_dim, device=self.device) 
        z_mean_posterior_t = torch.zeros(batch_size, self.z_dim, device=self.device) 
        z_logvar_posterior_t = torch.zeros(batch_size, self.z_dim, device=self.device) 
 
        cate_t = torch.zeros(batch_size, self.cate_dim, device=self.device)

        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.device) 
        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        
        for t in range(self.T):
            x_h_t = torch.cat((x[:,t,:], h_t), dim=1)  
            logits = self.xh_c_layer(x_h_t).view(-1, self.cate_dim) 
            posterior_prob = F.softmax(logits, dim=-1)  
            cate_t = self.rt.gumbel_softmax(logits, temperature, self.device, self.hard_gumbel).view(-1, 1, self.cate_dim)
            
            x_h_cate_t = torch.cat((x_h_t, cate_t.view(-1, self.cate_dim)), dim=1) 
            z_mean_posterior_t = self.Pz_xhc_mean(x_h_cate_t) 
            z_logvar_posterior_t = self.Pz_xhc_logvar(x_h_cate_t) 
            z_posterior_t = self.rt.reparameterize_gaussian(z_mean_posterior_t, 
                                                            z_logvar_posterior_t, self.training)

            if z_posterior is None:
                cate_posterior = cate_t.view(-1, self.cate_dim).unsqueeze(1)
                logits_out = logits.unsqueeze(1) 
                posterior_prob_out = posterior_prob.unsqueeze(1)  
                z_posterior = z_posterior_t.unsqueeze(1) 
                z_mean_posterior = z_mean_posterior_t.unsqueeze(1) 
                z_logvar_posterior = z_logvar_posterior_t.unsqueeze(1) 
                h_out = h_t.unsqueeze(1) 
            else:
                cate_posterior = torch.cat((cate_posterior, cate_t.view(-1, self.cate_dim).unsqueeze(1)), dim=1) 
                logits_out = torch.cat((logits_out,logits.unsqueeze(1)), dim=1) 
                posterior_prob_out = torch.cat((posterior_prob_out, posterior_prob.unsqueeze(1)), dim=1)
                z_posterior = torch.cat((z_posterior, z_posterior_t.unsqueeze(1)), dim=1) 
                z_mean_posterior = torch.cat((z_mean_posterior, z_mean_posterior_t.unsqueeze(1)), dim=1) 
                z_logvar_posterior = torch.cat((z_logvar_posterior, z_logvar_posterior_t.unsqueeze(1)), dim=1) 
                h_out = torch.cat((h_out, h_t.unsqueeze(1)), dim=1)
            
            x_z_posterior_t = torch.cat((x[:,t,:], z_posterior_t.view(-1, self.z_dim)), dim=1) 
            phi_x_z_posterior_t = self.phi_xz(x_z_posterior_t) 
            h_t, c_t = self.rnn_enc(phi_x_z_posterior_t, (h_t, c_t))
            
        return z_posterior, z_mean_posterior, z_logvar_posterior, h_out, cate_posterior, logits_out, posterior_prob_out

    def forward(self, x, temperature):
        x = x.float()
        x_hat = self.enc_x(x)
        z_posterior, z_mean_posterior, z_logvar_posterior, h_out, cate_posterior, logits_out, posterior_prob_out = self.infer_qzc_x(
        x_hat, x_hat.size(0), temperature)
        return z_posterior, z_mean_posterior, z_logvar_posterior, h_out, cate_posterior, logits_out, posterior_prob_out 

class SGmVRNN(nn.Module):
    def __init__(self, cate_dim=5, z_dim = 10, conv_dim=20, hidden_dim=20,
                 T=20, w=1, n=36, 
                 temperature=1.0, min_temperature=0.1, anneal_rate=0.1, hard_gumbel = False,
                 enc='CNN', dec='CNN', nonlinearity=None, device=torch.device('cuda:0')):
        super(SGmVRNN, self).__init__()
        self.cate_dim = cate_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.T = T
        self.w = w
        self.n = n

        self.enc = enc
        self.dec = dec
        if self.enc == 'CNN':
            self.enc_dim = conv_dim
        else:
            raise ValueError('Unknown encoder: {}'.format(self.enc))

        self.device = device
        self.nonlinearity = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate
        self.hard_gumbel = hard_gumbel
        self.dec_init_dim = self.hidden_dim + self.z_dim

        self.losses = LossFunctions()

        self.inference = InferenceNet(self.cate_dim, self.z_dim, self.hidden_dim, self.enc_dim, enc=self.enc, 
                                      T=self.T, w=self.w, n=self.n,
                                      hard_gumbel=self.hard_gumbel, device=self.device)

        self.generation = GenerationNet(self.hidden_dim, self.cate_dim, self.z_dim, self.enc_dim, self.dec_init_dim, dec=self.dec, 
                                        T=self.T, w=self.w, n=self.n, 
                                        device=self.device) 
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)


    def loss_fn(self, x, z, z_mean_posterior, z_logvar_posterior, z_mean_prior, z_logvar_prior, 
                      x_mu, x_logsigma, cate, logits, posterior_probs):
        batch_size = x.size(0)
        
        loglikelihood = self.losses.log_normal(x.float(), x_mu.float(), torch.pow(torch.exp(x_logsigma.float()), 2))
        
        z_var_posterior = torch.exp(z_logvar_posterior)
        z_var_prior = torch.exp(z_logvar_prior)
        kld_z = 0.5 * torch.sum(z_logvar_prior - z_logvar_posterior
                        + ((z_var_posterior + torch.pow(z_mean_posterior - z_mean_prior, 2)) / z_var_prior)
                        -1)

        kld_cate = -self.losses.entropy(logits, posterior_probs) - np.log(1/self.cate_dim)

        return (-loglikelihood + kld_cate + kld_z)/batch_size, loglikelihood/batch_size, kld_z/batch_size, kld_cate/batch_size
 
    def forward(self, x):
        # Conduct Inference
        z_post, z_mean_post, z_logvar_post, hidden_state, cate_post, logits, post_probs = self.inference(x, self.temperature)
        # Conduct generation
        z_mean_prior, z_logvar_prior, x_mu, x_logsigma = self.generation(hidden_state, z_post, cate_post)

        return z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, x_mu, x_logsigma, cate_post, logits, post_probs
    
