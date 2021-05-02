import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm,  beta
#from sklearn.cluster.k_means_ import _k_init # kmeans++ initialization
import matplotlib.pyplot as plt

class SOPL_Layer(nn.Module):
    ### soft ordered prototype learning layer
    def __init__(self, K, D, n_iter=5, alpha=1, time_prior_weight=1, seed=None, dtype=torch.float, device='cuda'):
        super(SOPL_Layer, self).__init__()
        self.K = K
        self.D = D
        self.n_iter = n_iter
        self.alpha = alpha
        self.time_prior_weight = time_prior_weight
        self.seed = seed
        self.dtype = dtype
        self.device = device
        if seed is not None:
            torch.manual_seed(seed)
        self.init_params()
        self.center_init = False

    def init_params(self):
        centers = torch.randn(self.K, self.D, dtype=self.dtype, device=self.device) * 0.1
        time_means = torch.zeros(self.K, dtype=self.dtype, device=self.device)
        time_vars = torch.zeros(self.K, dtype=self.dtype, device=self.device)
        self.register_buffer('centers_', centers)
        self.register_buffer('time_means_', time_means)
        self.register_buffer('time_vars_', time_vars)

    def sort_centers(self):
        centers = self.centers_
        time_means = self.time_means_
        #print('time means:', time_means)
        time_order = time_means.argsort()
        #print('order:', time_order)
        sorted_centers = centers[time_order]
        return sorted_centers
            

    def beta_generate(self, x, mu, sigma):
        #print(mu, sigma)
        if sigma <= 1e-4 or mu <= 0:
            return np.ones_like(x) / x.shape[0]
        a = (1 - mu) * mu**2 / sigma**2 - mu
        b = a / mu - a
        y = beta.pdf(x, a, b)
        y = y / max(y.sum(), 1e-10)
        return y.astype(np.float32)

    def acquire_time_prior(self, t_range):
        T = t_range.shape[0]
        time_prior = np.ones([T, self.K], dtype=np.float32) / T
        """
           assume background and actions are beta distribution
        """
        for k in range(0, self.K):
            y = self.beta_generate(t_range, self.time_means_[k].item(), np.sqrt(self.time_vars_[k].item())) 
            time_prior[:, k] = np.clip(y, a_min=1e-30, a_max=1)
        return time_prior

    def acquire_time_prior_for_all(self, t_range_list):
        time_prior_list = []
        for t_range in t_range_list:
            time_prior = self.acquire_time_prior(t_range)
            time_prior_list.append(time_prior)

        all_time_prior = np.concatenate(time_prior_list, axis=0)
        return torch.tensor(all_time_prior, dtype=self.dtype, device=self.device)

    def update_time_params(self, weight, t_range_list):
        t_ranges = torch.tensor(np.concatenate(t_range_list), dtype=self.dtype, device=self.device)
        time_means = torch.matmul(weight.t(), t_ranges) / (torch.sum(weight.t(), dim=-1) + 1e-10)
        time_vars =  (weight * (t_ranges[:,None]-time_means[None,:])**2).sum(dim=0) / (torch.sum(weight, dim=0) + 1e-10)
        self.register_buffer('time_means_', time_means)
        self.register_buffer('time_vars_', time_vars)

    def forward(self, x_list, sort_centers=True, return_dist=False, init=True):
        T_list = [x.shape[0] for x in x_list]
        t_range_list = [((np.arange(T) + 0.5) / T).astype(np.float32) for T in T_list]
        x = torch.cat(x_list, 0)
        #concat_t = np.concatenate(t_list)
        #concat_t = torch.tensor(np.concatenate(t_list), dtype=self.dtype, device=self.device)

        if init:
            self.init_params()
            #data = x.detach().cpu().numpy()
            #squared_norm = (data ** 2).sum(axis=1)
            #init_centers = _k_init(data, self.K, squared_norm, np.random.RandomState(self.seed))
            #centers = torch.tensor(init_centers).to(x.device).type(x.dtype)
        centers = self.centers_.data

        x_ = torch.unsqueeze(x, 1)
        #x_ = x_.repeat(1, self.K, 1)
        #x_norm = (x**2).sum(1).view(-1, 1)
        mse_list = []
        for i in range(self.n_iter):
            time_priors = -torch.log(self.acquire_time_prior_for_all(t_range_list))
            if torch.isnan(time_priors).any():
                print('***Time NAN!')
            if torch.isinf(time_priors).any():
                print('***Time INF!')
            #print(time_priors)
            dist = torch.sum((x_ - centers.unsqueeze(0))**2, dim=-1) + self.time_prior_weight * time_priors #- torch.log(time_priors)
            weight = F.softmax(-dist/self.alpha, dim=-1)
            #print('weight:', weight.detach().cpu().numpy())
            if torch.isnan(weight).any():
                print('===Weight NAN!')
            centers = torch.mm(weight.t(), x) / (torch.sum(weight.t(), dim=-1, keepdim=True) + 1e-10)
            self.update_time_params(weight, t_range_list)
            if torch.isnan(centers).any():
                print('nan!')
                exit()
            weighted_dist = (dist * weight).sum(0).detach().cpu().numpy()
            #print(i, weighted_dist.sum())
            #print(i, dist.detach().cpu().numpy().min(1).sum())
            mse_list.append([weighted_dist.sum(), dist.detach().cpu().numpy().min(1).sum()])
        self.register_buffer('centers_', centers)
        if sort_centers:
            centers = self.sort_centers()

        return centers if not return_dist else (centers, mse_list)


if __name__ == '__main__':
    K = 5
    D = 16
    plt.figure()
    ax = plt.subplot(1,1,1)
    colors = ['r','g','b','y','orange']

    np.random.seed(1)
    torch.manual_seed(1)
    x_list = []
    N = np.random.randint(10,15,1)[0]
    t_list = np.random.randint(30,50,N)
    print(t_list)
    for i in range(N):
        x = torch.randn(t_list[i], D).float().cuda()
        x_list.append(x)
        print(x[0,0])
    for n in range(5):
        sopl_layer = SOPL_Layer(K, D, n_iter=50, alpha=1, seed=None, dtype=torch.float, device='cuda')
        #print("===Init kmeans++")
        #centers, mse = sopl_layer(x_list, True, first=True)
        #plt.plot(np.array(mse)[:,1], label='kmeans '+str(n), color=colors[n])
        sopl_layer2 = SOPL_Layer(K, D, n_iter=50, alpha=1, seed=None, dtype=torch.float, device='cuda')
        print("===Init random")
        centers, mse = sopl_layer(x_list, True, init=False)
        plt.plot(np.array(mse)[:,1], '--', label='random '+str(n), color=colors[n])
    plt.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('2.pdf')

