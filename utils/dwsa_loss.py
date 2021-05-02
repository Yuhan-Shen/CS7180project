import numpy as np
import torch
import torch.nn.functional as F
from numba import jit
from torch.autograd import Function
import matplotlib.pyplot as plt

@jit(nopython = True)
def compute_dwsa_loss(C, gamma):
    '''
       compute differentiable weak sequence alignment loss
    '''
    M, N = C.shape
    D = np.ones((M, N)) * 100 #np.inf
    D[0, :] = C[0, :]

    for i in range(1, M):
        for j in range(0, N):
            if j % 2 == 0:
                last_row = D[i-1, :j+1]
            else:
                last_row = D[i-1, :j]

            rmin = np.min(last_row)
            rsum = np.sum(np.exp( - (last_row - rmin) / gamma))
            softmin = -gamma * np.log(rsum) + rmin #+ gamma * np.log(last_row.shape[0])
            D[i, j] = softmin + C[i, j]
    #print('D:', D)

    last_row = D[-1, :]
    rmin = np.min(last_row)
    rsum = np.sum(np.exp( - (last_row - rmin) / gamma))
    softmin = -gamma * np.log(rsum) + rmin
    #print(softmin)

    return D, softmin

@jit(nopython = True)
def compute_dwsa_loss_backward(C, D, gamma):
    M, N = C.shape
    E = np.exp( -D / gamma)
    #print('E:', E)
    G = np.zeros((M, N), dtype=C.dtype)
    final_row = D[-1, :] - D[-1, :].min()
    exp_final_row = np.exp( -final_row / gamma)
    G[-1, :] = exp_final_row / (exp_final_row.sum()+1e-4)
    #G[-1, :] = E[-1, :] / (E[-1, :].sum()+1e-4)
    #G[:-1, -1] = 0

    for i in range(M-2, -1, -1):
        for j in range(N-1, -1, -1):
            if j % 2 == 0:
                delta = D[i+1, j:] - C[i+1, j:] - D[i, j] #- gamma * np.log(np.arange(j, N))
                delta = G[i+1, j:] * np.exp(delta/gamma)
                G[i, j] = delta.sum()
            else:
                delta = D[i+1, j+1:] - C[i+1, j+1:] - D[i, j] #- gamma * np.log(np.arange(j+1, N))
                delta = G[i+1, j+1:] * np.exp(delta/gamma)
                G[i, j] = delta.sum()
            #g_sum = 0
            #for l in range(j+1, N):
            #    g_sum += G[i+1, l] / (np.sum(E[i, :l])+1e-4)
            #G[i, j] = E[i, j] * g_sum
    #print('G:', G)
    return G

class _DWSALoss(Function):
    @staticmethod
    def forward(ctx, C, gamma):
        dev = C.device
        dtype = C.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
        C_ = C.detach().cpu().numpy()
        g_ = gamma.item()
        D, softmin = compute_dwsa_loss(C_, g_)
        D = torch.Tensor(D).to(dev).type(dtype)
        softmin = torch.Tensor([softmin]).to(dev).type(dtype)
        ctx.save_for_backward(C, D, softmin, gamma)
        return softmin

    @staticmethod
    def backward(ctx, grad_output):
        #print('grad:', grad_output)
        dev = grad_output.device
        dtype = grad_output.dtype
        C, D, softmin, gamma = ctx.saved_tensors
        C_ = C.detach().cpu().numpy()
        D_ = D.detach().cpu().numpy()
        softmin_ = softmin.item()
        g_ = gamma.item()
        G = torch.Tensor(compute_dwsa_loss_backward(C_, D_, g_)).to(dev).type(dtype)
        return grad_output.view(-1, 1).expand_as(G) * G, None

class DWSA_Loss(torch.nn.Module):
    def __init__(self, alpha=0.01, center_norm=False, sim='cos', threshold=2, softmax='row'):
        super(DWSA_Loss, self).__init__()

        self.alpha = alpha
        self.center_norm = center_norm
        self.func_apply = _DWSALoss.apply
        self.sim = sim
        self.threshold = threshold
        self.softmax = softmax

    def forward(self, centers_a, centers_b):
        sorted_centers_a = centers_a
        sorted_centers_b = centers_b
        
        if self.center_norm:
            sorted_centers_a = sorted_centers_a / torch.sqrt(torch.sum(sorted_centers_a**2, axis=-1, keepdims=True) + 1e-10) 
            sorted_centers_b = sorted_centers_b / torch.sqrt(torch.sum(sorted_centers_b**2, axis=-1, keepdims=True) + 1e-10)
        
        if self.sim == 'cos':
            matching = 1 - torch.matmul(sorted_centers_a, sorted_centers_b.t())
            C_aa = 1 - torch.matmul(sorted_centers_a, sorted_centers_a.t())
            C_bb = 1 - torch.matmul(sorted_centers_b, sorted_centers_b.t())
        elif self.sim == 'exp':
            matching = torch.exp(-torch.matmul(sorted_centers_a, sorted_centers_b.t()))
            C_aa = torch.exp(-torch.matmul(sorted_centers_a, sorted_centers_a.t()))
            C_bb = torch.exp(-torch.matmul(sorted_centers_b, sorted_centers_b.t()))
        elif self.sim == 'euc':
            a_ = torch.unsqueeze(sorted_centers_a, 1)
            matching = torch.sum((a_ - sorted_centers_b.unsqueeze(0))**2, axis=-1)
        else:
            print('Invalid similarity metric! {}'.format(self.sim))
            exit()

        #print('matching:', matching.shape, matching.min().item(), matching.max().item())
        #print(matching.detach().cpu().numpy())
        threshold = self.threshold
        L_a, L_b = sorted_centers_a.shape[0], sorted_centers_b.shape[0]
        #print('L_a, L_b:', L_a, L_b)
        matching = torch.cat((matching, torch.ones_like(matching) * threshold))
        #print('matching 2:', matching.shape)
        matching = matching.t().reshape(2 * L_b, L_a).t()
        matching = torch.cat((threshold * torch.ones(L_a, 1, dtype=matching.dtype).to(matching.device), matching), dim=1)

        if self.softmax == 'row':
            matching = F.softmax(matching, -1)
        elif self.softmax == 'col':
            matching = F.softmax(matching, 0)
        elif self.softmax == 'all':
            matching = F.softmax(matching.view(-1), 0).view(L_a, 2*L_b+1)
        #print('after:', matching)
        loss = self.func_apply(matching, self.alpha) / L_a

        return loss 
