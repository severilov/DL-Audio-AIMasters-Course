#!/usr/bin/env python
# coding: utf-8

# # VAE Vocoder

# In[1]:


# ! pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2  -f https://download.pytorch.org/whl/torch_stable.html
# ! pip install numpy==1.17.5 matplotlib==3.3.3 tqdm==4.54.0


# In[29]:


import torch
from torch import nn
from torch.nn import functional as F
from typing import Union
from math import log, pi, sqrt
from IPython.display import display, Audio
from torch.distributions.normal import Normal
import numpy as np
from collections import OrderedDict

import os

device = torch.device("cpu")
if torch.cuda.is_available():
    print('GPU found! 🎉')
    device = torch.device("cuda")


# ## Masked Autoregressive Flow - MAF
# 
# ### Sampling
# 
# During sampling process of **MAF** network we use previous $x_{1:i-1}$ states to predict $\mu_i$ and $\alpha_i$ to sample $x_i$ as $x_i = u_i \cdot \exp(\alpha_i) + \mu_i$, we can do it in **autoregressive** manner.
# 
# $$p(x) = \prod_{i}p(x_i|x_{1:i-1})$$
# 
# $$p(x_i|x_{1:i-1}) = \mathcal{N}(x_i | \mu_i, (\exp{\alpha_i})^2)$$
# 
# $$\mu_i = f_{\mu_i}(x_{1:i-1})$$
# 
# $$\alpha_i = f_{\alpha_i}(x_{1:i-1})$$
# 
# $$u_i \sim \mathcal{N}(0, 1)$$
# 
# <img src="./maf.PNG" width="80%">
# 
# ### Training
# 
# During training process of **MAF** network we restore $u_i, i=1...D$, using the whole signal $x$ as following:
# 
# $$
# u_i = \frac{x_i - \mu_i}{\exp{\alpha_i}}
# $$, where 
# 
# $$\mu_i = f_{\mu_i}(x_{1:i-1})$$
# 
# $$\alpha_i = f_{\alpha_i}(x_{1:i-1})$$
# 
# 
# 
# <img src="./maf_inv.PNG" width="80%">

# ## Inversed Autoregressive Flow - IAF
# 
# ### Sampling
# 
# Here we want to sample in а **non-autoregressive** manner, so we employ predicting $\mu_i$ and $\alpha_i$ from noise $u$:
# 
# $$\mu_i = f_{\mu_i}(u_{1:i-1})$$
# 
# $$\alpha_i = f_{\alpha_i}(u_{1:i-1})$$
# 
# $$x_i = u_i \cdot \exp{\alpha} + \mu_i$$
# 
# <img src="./iaf.PNG" width="80%">
# 
# ### Training
# 
# Here must be a fancy picture, but it's not...
# 
# The problem is that we have achieved non-autoregressiveness on inference, but on training step we use following equation to restore noise $u$ fro signal $x$:
# 
# $$
# u_i = \frac{x_i - \mu_i}{\exp{\alpha_i}}
# $$, similar to one in **MAF** inference, but now we need to predict $\mu_i$ and $\sigma_i$ from  $u_{1:i-1}$, so we need previous states of noise $u$ to restore the next one. That is the problem of **IAF**.

# ## [Parallel WaveNet](https://arxiv.org/pdf/1711.10433.pdf)
# 
# While the convolutional structure of WaveNet allows for rapid parallel training, sample generation
# remains inherently sequential and therefore slow, as it is for all autoregressive models which use
# ancestral sampling. We therefore seek an alternative architecture that will allow for rapid, parallel
# generation.
# 
# <img src="./parallel_wavenet1.PNG" width="100%">
# 
# <img src="./parallel_wavenet2.PNG" width="80%">

# ## Autoregressive Flow
# 
# ### What do we want to obtain?
# 
# Consider transformations of random variable $z^{(0)} \sim \mathcal{N}(0, I)$: 
# $$z^{(0)} \rightarrow z^{(1)} \rightarrow \dots \rightarrow z^{(n)} \rightarrow x.$$
# 
# Each transformation has the form: 
# $$ 
# z^{(k)} = f^{(k)}(z^{(k-1)}) = z{(k-1)} \cdot \sigma^{(k)} + \mu^{(k)},
# $$ 
# where $\mu^{(k)}_t = \mu(z_{<t}^{(k-1)}; \theta_k)$ and $\sigma^{(k)}_t = \sigma(z_{<t}^{(k-1)}; \theta_k)$ – are shifting and scaling variables modeled by a Gaussian WaveNet. 
# 
# It is easy to deduce that the whole transformation $f^{(k)} \circ \dots \circ f^{(2)} \circ f^{(1)}$ can be represented as $f^{(\mathrm{total})}(z) = z \cdot \sigma^{(\mathrm{total})} + \mu^{(\mathrm{total})}$, where
# $$\sigma^{(\mathrm{total})} = \prod_{k=1}^n \sigma^{(k)}, ~ ~ ~ \mu^{(\mathrm{total})} = \sum_{k=1}^n \mu^{(k)} \prod_{j > k}^n \sigma^{(j)} $$
# 
# $\mu^{(\mathrm{total})}$ and $\sigma^{(\mathrm{total})}$ we will need in the future for $p(x | z)$ estimation.

# ### Helpful notes
# 
# [Jacobian matrix and determinant](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Jacobian_determinant):
# 
# $
# \mathbf J = \begin{bmatrix}
#   \dfrac{\partial \mathbf{f}}{\partial x_1} & \cdots & \dfrac{\partial \mathbf{f}}{\partial x_n}
# \end{bmatrix}
# = \begin{bmatrix}
#   \nabla^{\mathrm T} f_1 \\  
#   \vdots \\
#   \nabla^{\mathrm T} f_m   
# \end{bmatrix}
# = \begin{bmatrix}
#     \dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n}\\
#     \vdots                             & \ddots & \vdots\\
#     \dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
# \end{bmatrix}
# $
# 
# **Lemma:**
# 
# Suppose $z \in \mathbb{R}^n$ an n-dimensional random variable with joint density $p$. If $x = f(z)$, where $f$ is a bijective, differentiable function, then $x$ has density $p_x(x)$:
# 
# $$
# p_x(\mathbf x) = 
# p_z(\mathbf z) \cdot \left| \det\frac{d\mathbf{z}}{d\mathbf{x}} \right| = 
# p_z(\mathbf z) \cdot \left| \det\frac{df^{-1}(\mathbf{x})}{d\mathbf{x}} \right| 
# $$, where $\dfrac{df^{-1}(\mathbf{x})}{d\mathbf{x}}$ is a **Jacobian** of the inverse of $f: f^{-1}$
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Jacobian_determinant_and_distortion.svg/1920px-Jacobian_determinant_and_distortion.svg.png" width="80%">
# 
# Physical interpretation of this image is that $\left| \det\frac{d\mathbf{z}}{d\mathbf{x}} \right|$ shows us how the new probability density $p_x(\mathbf x)$ changes from $p_z(\mathbf z)$, how it stretches.
# 
# In logarithmic scale we have:
# $$
# \log p_x(\mathbf x) = \log p_z(f^{-1}(\mathbf{x})) + \log{\left| \det\dfrac{df^{-1}(\mathbf{x})}{d\mathbf{x}} \right|}
# $$
# 
# so we want to maximize this logarithmic likelihood, we can compute it numerically knowing function $f$ and initial distribution $z$

# ### How to compute Jacobian for IAF?
# 
# During training we compute $z$ as:
# 
# $$z_i = \frac{x_i - \mu(z_{<t}^{(k-1)}; \theta_k)}{\sigma(z_{<t}^{(k-1)}; \theta_k)}$$
# 
# $\mu, \sigma$ have no dependence on the current or latter variables in our sequence, so: 
# 
# $$
# \dfrac{d \mathbf{z}}{d\mathbf{x}} = \begin{bmatrix}
#     \dfrac{1}{\sigma_0} & 0 & \cdots & 0\\
#     \dfrac{\partial z_1}{\partial x_0} & \dfrac{1}{\sigma_1} &  \ddots & 0\\
#     \vdots & \ddots & \ddots & 0 \\
#     \dfrac{\partial z_n}{\partial x_0} & \cdots & \dfrac{\partial z_{n-1}}{\partial x_n} & \dfrac{1}{\sigma_n}
# \end{bmatrix}
# $$
# 
# Knowing that the determinant of a triangular matrix is the product of its diagonals, this gives us our final result for the log determinant which is incredibly simple to compute:
# 
# $$
# \log{\left| \det\dfrac{df^{-1}(\mathbf{x})}{d\mathbf{x}} \right|} = - \sum_{i=0}^{n} \log{\sigma_i(z)}
# $$

# Introduce auxiliary modules:
# 1. causal convolution – simple convolution with `kernel_size` and `dilation` hyper-parameters, but working in causal way (does not look in the future)
# 2. residual block – main building component of WaveNet architecture
# 
# Yes, WaveNet is everywhere. We can build MAF and IAF with any architecture, but WaveNet declared oneself as simple yet powerfull architecture. We will use WaveNet with conditioning on mel spectrograms, because we are building a vocoder.

# In[30]:


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv, self).__init__()

        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size, dilation, cin_channels):
        super(ResBlock, self).__init__()
        self.cin_channels = cin_channels

        self.filter_conv = CausalConv(in_channels, out_channels, kernel_size, dilation)
        self.gate_conv = CausalConv(in_channels, out_channels, kernel_size, dilation)
        self.res_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(out_channels, skip_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        nn.init.kaiming_normal_(self.skip_conv.weight)

        self.filter_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
        self.gate_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
        self.filter_conv_c = nn.utils.weight_norm(self.filter_conv_c)
        self.gate_conv_c = nn.utils.weight_norm(self.gate_conv_c)
        nn.init.kaiming_normal_(self.filter_conv_c.weight)
        nn.init.kaiming_normal_(self.gate_conv_c.weight)

    def forward(self, x, c=None):
        h_filter = self.filter_conv(x)
        h_gate = self.gate_conv(x)
        h_filter += self.filter_conv_c(c)
        h_gate += self.gate_conv_c(c)
        out = torch.tanh(h_filter) * torch.sigmoid(h_gate)
        res = self.res_conv(out)
        skip = self.skip_conv(out)
        return (x + res) * sqrt(0.5), skip


# For WaveNet it doesn't matter what it is used for: MAF or IAF - it all depends on our interpretation of the input and output variables.

# In[31]:


class WaveNet(nn.Module):
    def __init__(self, 
                 out_channels=2,
                 num_blocks=2,
                 num_layers=10,
                 residual_channels=128,
                 gate_channels=256,
                 skip_channels=128,
                 kernel_size=2,
                 cin_channels=80,
                 upsample_scales=[16, 16]):
        super(WaveNet, self). __init__()

        self.front_conv = nn.Sequential(
            CausalConv(1, residual_channels, 32),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        for b in range(num_blocks):
            for n in range(num_layers):
                self.res_blocks.append(ResBlock(
                    in_channels=residual_channels,
                    out_channels=gate_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** n,
                    cin_channels=cin_channels))

        self.final_conv = nn.Sequential(
            nn.ReLU(),
            CausalConv(skip_channels, skip_channels, 1),
            nn.ReLU(),
            CausalConv(skip_channels, out_channels, 1)
        )
        
        self.upsample_conv = nn.ModuleList()
        for s in upsample_scales:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))
    
    def forward(self, x, c):
        # x: input tensor with signal or noise [B, 1, T]
        # c: local conditioning [B, C_mel, T]
        
        c = self.upsample(c)
        out = 0
        
        x = self.front_conv(x)
        
        for block in self.res_blocks:
            x, skip = block(x, c)
            out += skip

        out = self.final_conv(out)
        return out

    def upsample(self, c):
        if self.upsample_conv is not None:
            # B x 1 x C x T'
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            c = c.squeeze(1)
        return c


# In[32]:


net = WaveNet().to(device).eval()

with torch.no_grad():
    z = torch.FloatTensor(5, 1, 4096).normal_().to(device)
    c = torch.FloatTensor(5, 80, 4096 // 256).zero_().to(device)
    assert list(net(z, c).size()) == [5, 2, 4096]


# 📝 Notes: 
# 1. WaveNet outputs tensor `output` of size `[B, 2, T]`, where `output[:, 0, :]` is $\mu$ and `output[:, 1, :]` is $\log \sigma$. We model logarithms of $\sigma$ insead of $\sigma$ for stable gradients. 
# 2. As we model $\mu(z_{<t}^{(k-1)}; \theta_k)$ and $\sigma(z_{<t}^{(k-1)}; \theta_k)$ – their output we have length `T - 1`. To keep constant length `T` of modelled noise variable we need to pad it on the left side (with zero).
# 3. $\mu^{(\mathrm{total})}$ and $\sigma^{(\mathrm{total})}$ wil have length `T - 1`, because we do not pad distribution parameters.

# In[47]:


class Wavenet_Flow(nn.Module):
    def __init__(self, 
                 out_channels=1, 
                 num_blocks=1, 
                 num_layers=10,
                 front_channels=32, 
                 residual_channels=64, 
                 gate_channels=32, 
                 skip_channels=64,
                 kernel_size=3, 
                 cin_channels=80):
        super(Wavenet_Flow, self). __init__()

        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.front_channels = front_channels
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.cin_channels = cin_channels
        self.kernel_size = kernel_size

        self.front_conv = nn.Sequential(
            CausalConv(1, self.residual_channels, self.front_channels),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList()
        self.res_blocks_fast = nn.ModuleList()
        for b in range(self.num_blocks):
            for n in range(self.num_layers):
                self.res_blocks.append(
                    ResBlock(
                        self.residual_channels, 
                        self.gate_channels, 
                        self.skip_channels,
                        self.kernel_size, 
                        dilation=2**n,
                        cin_channels=self.cin_channels
                    )
                )
                
        self.final_conv = nn.Sequential(
            nn.ReLU(),
            CausalConv(self.skip_channels, self.skip_channels, 1),
            nn.ReLU(),
            CausalConv(self.skip_channels, self.out_channels, 1)
        )

    def forward(self, x, c=None):
        x = self.front_conv(x)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            x, s = f(x, c)
            skip += s
        out = self.final_conv(skip)
        return out

class Wavenet_Decoder(nn.Module):
    def __init__(self, 
                 num_blocks_student=[1, 1, 1, 1, 1, 1], 
                 num_layers=10,
                 front_channels=32, 
                 residual_channels=64, 
                 gate_channels=128, 
                 skip_channels=64,
                 kernel_size=3, 
                 cin_channels=80):
        super(Wavenet_Decoder, self).__init__()
        self.num_blocks = num_blocks_student
        self.num_flow = len(self.num_blocks)
        self.num_layers = num_layers

        self.iafs = nn.ModuleList()
        for i in range(self.num_flow):
            self.iafs.append(
                Wavenet_Flow(
                    out_channels=2,
                    num_blocks=self.num_blocks[i], 
                    num_layers=self.num_layers,
                    front_channels=front_channels, 
                    residual_channels=residual_channels,
                    gate_channels=gate_channels, 
                    skip_channels=skip_channels,
                    kernel_size=kernel_size, 
                    cin_channels=cin_channels
                )
            )
    
    def forward(self, z, c_up):
        mu_tot, logs_tot = 0., 0.
        for i, iaf in enumerate(self.iafs):
            mu_logs = iaf(z, c_up)
            mu = mu_logs[:, 0:1, :-1]
            logs = mu_logs[:, 1:, :-1]
            mu_tot = mu_tot * torch.exp(logs) + mu
            logs_tot = logs_tot + logs
            z = z[:, :, 1:] * torch.exp(logs) + mu
            z = F.pad(z, pad=(1, 0), mode='constant', value=0)
        return z, mu_tot, logs_tot

    def generate(self, z, c_up):
        x, _, _ = self(z, c_up)
        return x


# In[48]:


net = Wavenet_Decoder(
    num_blocks_student=[1, 1, 1, 1, 1, 1],
    num_layers=10
).to(device)

with torch.no_grad():
    z = torch.FloatTensor(3, 1, 4096).normal_().to(device)
    c = torch.FloatTensor(3, 80, 4096).zero_().to(device)
    z_hat, mu, log_sigma = net(z, c)
    assert list(z_hat.size()) == [3, 1, 4096]         # same length as input
    assert list(mu.size()) == [3, 1, 4096 - 1]        # shorter by one sample
    assert list(log_sigma.size()) == [3, 1, 4096 - 1] # shorted by one sample


# If you are not familiar with VAE framework, please try to figure it out. For example, please familiarize with this [blog post](https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/).
# 
# 
# In short, VAE – is just "modification" of AutoEncoder, which consists of encoder and decoder. VAE allows you to sample from data distribution $p(x)$ as $p(x|z)$ via its decoder, where $p(z)$ is simple and known, e.g. $\mathcal{N}(0, I)$. The interesting part is that $p(x | z)$ cannot be optimized with Maximum Likelihood Estimation, because $p(x | z)$ is not tractable. 
# 
# But we can maximize Evidence Lower Bound (ELBO) which has a form:
# 
# $$\max_{\phi, \theta} \mathbb{E}_{q_{\phi}(z | x)} \log p_{\theta}(x | z) - \mathbb{D}_{KL}(q_{\phi}(z | x) || p(z))$$
# 
# where $p_{\theta}(x | z)$ is VAE decoder and $q_{\phi}(z | x)$ is VAE encoder. For more details please read mentioned blog post or any other materials on this theme.
# 
# In our case $q_{\phi}(z | x)$ is represented by MAF WaveNet, and $p_{\theta}(x | z)$ – by IAF build with WaveNet stack. To be more precise our decoder $p_{\theta}(x | z)$ is parametrised by the **one-step-ahead prediction** from an IAF.
# 
# `generate` method, which accepts mel spectrogram as conditioning tensor. Inside this method random tensor from standart distribution N(0, I) is sampled. This tensor than transformed to tensor from audio distribution via `encoder`. In the cell bellow you will see code for loading pretrained model and mel spectrogram. Listen to result – it should sound okay.

# The `forward` method will return the loss. But lets talk more precisly about our architecture and how it was trained.
# 
# The encoder of our model $q_{\phi}(z|x)$ is parametrerized by a Gaussian autoregressive WaveNet, which maps the audio $\mathbf{x}$ into the sample length latent representation $\mathbf{z}$. Specifically, the Gaussian WaveNet (if we talk about **real MAF**) models $x_t$ given the previous samples $x_{<t}$ with $x_t ∼ \mathcal{N}(\mu(x_{<t}; \phi), \sigma(x_{<t}; \phi))$, where the mean $\mu(x_{<t}; \phi)$ and log-scale $\log \sigma(x_{<t}; \phi)$ are predicted by WaveNet, respectively.
# 
# Our **encoder** posterior is constructed as
# 
# $$q_{\phi}(z | x) = \prod_{t} q_{\phi}(z_t | x_{\leq t})$$
# 
# where
# 
# $$q_{\phi}(z_t | x_{\leq t}) = \mathcal{N}(\frac{x_t - \mu(x_{<t}; \phi)}{\sigma(x_{<t}; \phi)}, \varepsilon)$$
# 
# We apply the mean $\mu(x_{<t}; \phi)$ and scale $\sigma(x_{<t})$ for "whitening" the posterior distribution. Also we introduce a trainable scalar $\varepsilon > 0$ to decouple the global variation, which will make optimization process easier.
# 
# Substitution of our model formulas in $\mathbb{D}_{KL}$ formula gives `loss_kld` in `forward` method as KL divergence:
# 
# $$\mathbb{D}_{KL}(q_{\phi}(z | x) || p(z)) = \sum_t \log\frac{1}{\varepsilon} + \frac{1}{2}(\varepsilon^2 - 1 + (\frac{x_t - \mu(x_{<t})}{\sigma(x_{<t})})^2)$$
# 
# ---
# 
# The other term in ELBO formula can be interpreted as reconstruction loss. It can be evaluated by sampling from $p_{\theta}(x | z)$, where $z$ is from $q_{\phi}(z | \hat x)$, $\hat x$ is our ground truth audio. But sampling is not differential operation! We can apply reparametrization trick!
# 
# `loss_rec` in `forward` method is a recontruction loss – which is just log likelihood of ground truth sample $x$ in predicted by IAF distribution $p_{\theta}(x | \hat z)$ where $\hat z \sim q_{\phi}(z | \hat x)$.
# 
# $$
# \log p_{\theta}(x | z) = \log p(z) - \sum_{t=1}^{T} \sum_{i=0}^{N} \log{\sigma_{i,t}} = 
# - \dfrac{1}{2} \sum_{i=0}^{N} \left[ \log{2\pi} + \frac{x_{i,t} - \mu(x_{<t}; \phi)}{\sigma(x_{<t})} + \sum_{t=1}^{T}\log{\sigma_{i,t}} \right]
# $$
# 
# --- 
# 
# Vocoders without MLE are still not able to train without auxilary losses. We studied many of them, but STFT-loss is our favourite!
# 
# `loss_frame_rec` stands for MSE loss in STFT domain between original audio and its reconstruction.
# 
# --- 
# 
# We can go even further and calculate STFT loss with random sample from $p_\theta(x | z)$. Conditioning on mel spectrogram allows us to do so.
# 
# `loss_frame_prior` stands for MSE loss in STFT domain between original audio and sample from prior.

# In[ ]:


class WaveNetVAE(nn.Module):
    def __init__(self):
        super(WaveNetVAE, self).__init__()

        self.encoder = WaveNet(out_channels=2,
                               num_blocks=2,
                               num_layers=10,
                               residual_channels=128,
                               gate_channels=256,
                               skip_channels=128,
                               kernel_size=2,
                               cin_channels=80,
                               upsample_scales=[16, 16])
        self.decoder = Wavenet_Decoder(num_blocks_student=[1, 1, 1, 1, 1, 1],
                                       num_layers=10)
        self.log_eps = nn.Parameter(torch.zeros(1))
    
    @staticmethod
    def stft(y):
        D = torch.stft(y, n_fft=1024, hop_length=256, win_length=1024, window=torch.hann_window(1024), return_complex=False)
        D = torch.sqrt(D.pow(2).sum(-1) + 1e-10)
        S = 2 * torch.log(torch.clamp(D, 1e-10, float("inf")))
        return D, S
    
    def forward(self, x, c):
        # x: audio signal [B, 1, T]
        # c: mel spectrogram [B, mel_c, T / HOP_SIZE]

        # Encode
        # mu_logs = [μ, logσ] used to define the posterior distribution q(z|x).
        # mu, logs ~ [B, 1, T-1]
        # target x[:, :, 1:] ~ [B, 1, T-1] as well

        mu_logs = self.encoder(x, c) # -> [B, 2, T] 
        mu = mu_logs[:, 0:1, :-1] # μ
        logs = mu_logs[:, 1:, :-1] # logσ

        # Base normal distribution
        q_0 = Normal(mu.new_zeros(mu.size()), mu.new_ones(mu.size()))
        
        # "Whitening” the posterior. This centers and scales x relative to encoder output. 
        # We use this as the mean of the posterior.
        # We want mean_q to be similar to N(0, 1), ideally mean_q ~ N(0, 1)
        mean_q = (x[:, :, 1:] - mu) * torch.exp(-logs)

        # Reparameterization, Sampling from prior
        z = q_0.sample() * torch.exp(self.log_eps) + mean_q
        z_prior = q_0.sample()

        # First token = 0
        z = F.pad(z, pad=(1, 0), mode='constant', value=0)
        z_prior = F.pad(z_prior, pad=(1, 0), mode='constant', value=0)
        c_up = self.encoder.upsample(c)

        # Decode
        # x_rec : [B, 1, T] (first time step zero-padded)
        # mu_tot, logs_tot : [B, 1, T-1]
        x_rec, mu_p, log_p = self.decoder(z, c_up)
        x_prior, _, _ = self.decoder(z_prior, c_up)

        loss_recon = -0.5 * (- log(2.0 * pi) - 2. * log_p - torch.pow(x[:, :, 1:] - mu_p, 2) * torch.exp((-2.0 * log_p)))
        loss_kl = 0.5 * (mean_q ** 2 + torch.exp(self.log_eps) ** 2 - 1) - self.log_eps
        
        # For annealing during training:
        # slowly increase the weight of KL loss over training to avoid posterior collapse.
        # early in training, let model focus on reconstruction; 
        # later encourage meaningful latent structure.
        global_step = 35000
        alpha = 1 / (1 + np.exp(-5e-5 * (global_step - 5e+5))) 
        
        stft_rec, stft_rec_log = self.stft(x_rec[:, 0, 1:])
        stft_truth, stft_truth_log = self.stft(x[:, 0, 1:])
        stft_prior, stft_prior_log = self.stft(x_prior[:, 0, 1:])
        
        loss_frame_rec = F.mse_loss(stft_rec, stft_truth) + F.mse_loss(stft_rec_log, stft_truth_log)
        loss_frame_prior = F.mse_loss(stft_prior, stft_truth) + F.mse_loss(stft_prior_log, stft_truth_log)
        
        return  loss_recon.mean() + alpha * loss_kl.mean() + loss_frame_rec + loss_frame_prior

    def generate(self, c):
        # c: mel spectrogram [B, 80, L] where L - number of mel frames
        # outputs: audio [B, 1, L * HOP_SIZE]
        c_up = self.encoder.upsample(c)
        
        q_0 = Normal(torch.zeros(1, 1, c_up.size(2)), torch.ones(1, 1, c_up.size(2)))
        z = q_0.sample()
        
        x_sample, _, _ = self.decoder(z, c_up)
        return x_sample

#     def upsample(self, c):
#         c = c.unsqueeze(1) # [B, 1, C, L]
#         for f in self.upsample_conv:
#             c = f(c)
#         c = c.squeeze(1) # [B, C, T], where T = L * HOP_SIZE
#         return c


# In[52]:


from matplotlib import pyplot as plt
alpha = [1 / (1 + np.exp(-5e-5 * (i - 5e+5))) for i in range(1, 100000)]
plt.plot(alpha)
plt.title("annealing during training")
plt.show()


# In[40]:


def load_checkpoint(model, checkpoint: OrderedDict):
    if 'module' not in list(checkpoint["state_dict"].keys())[0]:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("INFO: this model is trained with DataParallel. Creating new state_dict without module...")
        state_dict = checkpoint["state_dict"]
        
        new_state_dict = OrderedDict()
    
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(dict(new_state_dict))


# Link to checkpoint: https://disk.yandex.ru/d/Yl2Ie7ZNJRa5HQ

# In[41]:


# saved checkpoint model has following architecture parameters
        
# load checkpoint
ckpt_path = 'checkpoint.pth'
model = WaveNetVAE().eval().to(device)
checkpoint = torch.load(ckpt_path, map_location='cpu')
load_checkpoint(model, checkpoint)

# load original audio and it's mel
x = torch.load('data/x.pth').to(device)
c = torch.load('data/c.pth').to(device)

# generate audio from 
with torch.no_grad():
    x_prior = model.generate(c.unsqueeze(0)).squeeze()

display(Audio(x_prior.cpu(), rate=22050))


# In[42]:


x = torch.load('data/x.pth').to(device)
c = torch.load('data/c.pth').to(device)

model = WaveNetVAE().to(device).train()

x = x[:64 * 256]
c = c[:, :64]

model.zero_grad()
loss = model.forward(x.unsqueeze(0).unsqueeze(0), c.unsqueeze(0))
loss.backward()
print(f"Initial loss: {loss.item():.2f}")

checkpoint = torch.load(ckpt_path, map_location='cpu')
load_checkpoint(model, checkpoint)

model.zero_grad()
loss = model.forward(x.unsqueeze(0).unsqueeze(0), c.unsqueeze(0))
loss.backward()
print(f"Optimized loss: {loss.item():.2f}")

