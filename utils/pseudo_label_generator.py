import numpy as np
#import torch
#import torch.nn.functional as F
from numba import jit
#from torch.autograd import Function

@jit(nopython = True)
def compute_softdtw(C, gamma, ratio):
    M, N = C.shape
    bandwidth = max(N / M, 1)
    D = np.ones((M+2, N+2)) * np.inf
    Q = np.zeros((M+2, N+2, 2))
    D[0, 0] = 0

    for i in range(1, M+1):
        for j in range(1, N+1):
            if ratio > 0 and np.abs(j/bandwidth - i + 0.5) > ratio/2:
                #print(i,j,'continue')
                continue
            r0 = -D[i - 1, j - 1] / gamma
            #r1 = -D[i - 1, j] / gamma
            r2 = -D[i, j - 1]  / gamma
            #print(i,j,r0, r2)

            rmax = max(r0, r2)
            #rmax = max(max(r0, r2), r1)
            #print(r0, r2)
            if rmax == -np.inf:
                #print('inf:', rmax, r0, r2)
                softmin = np.inf
            else:
                rsum = max(np.exp(r0 - rmax) + np.exp(r2 - rmax), 1e-10)
                #rsum = max(np.exp(r0 - rmax) + np.exp(r2 - rmax) + np.exp(r1 - rmax), 1e-10)
                softmin = - gamma * (np.log(rsum) + rmax)
            D[i, j] = C[i-1, j-1] + softmin
            if not rmax == -np.inf:
                e0 = np.exp(r0 - rmax)
                e2 = np.exp(r2 - rmax)
                #print(e0, e2)
                Q[i, j, 0] = e0 / (e0+e2)
                Q[i, j, 1] = e2 / (e0+e2)
    return D, D[-2, -2], Q


@jit(nopython = True)
def compute_softdtw_backward(C_, D, Q, gamma, ratio):
    M, N = C_.shape
    bandwidth = max(N / M, 1)
    C = np.zeros((M+2, N+2), dtype=C_.dtype)
    G = np.zeros((M+2, N+2), dtype=C_.dtype)

    C[1:M+1, 1:N+1] = C_
    G[-1, -1] = 1
    Q[-1, -1, 0] = 1
    D[:, -1] = -np.inf
    D[-1, :] = -np.inf
    D[-1, -1] = D[-2, -2]

    
    for i in range(M, 0, -1):
        for j in range(N, 0, -1):

            #if np.isinf(D[i, j]):
            #    D[i, j] = -np.inf
            if ratio > 0 and np.abs(j/bandwidth - i + 0.5) > ratio/2:
                continue
            a = Q[i+1, j+1, 0]
            b = Q[i, j+1, 1]
            G[i, j] = G[i+1, j+1] * a + G[i, j+1] * b
            #G[i, j] = G[i+1, j+1] * a + G[i, j+1] * b + G[i+1, j] * c
            #if np.isnan(G[i, j]):
            #    G[i, j] = 0
   #         if np.isnan(G[i, j]):
   #             #print('a,b',G[i+1,j+1], a, a0, G[i,j+1], b, b0)
   #             print(i,j)
   #             print('NAN!')
   #             print(G)
   #             #1/0

    return G[1:M+1, 1:N+1]

@jit(nopython = True)
def compute_dtw(C, ratio):
    M, N = C.shape
    bandwidth = max(N / M, 1)
    D = np.ones((M, N)) * np.inf
    for j in range(N):
        if ratio > 0 and np.abs((j+1)/bandwidth - 0.5) > ratio/2:
            continue
        D[0, j] = C[0, :j+1].sum()


    for i in range(1, M):
        for j in range(1, N):
            if ratio > 0 and np.abs((j+1)/bandwidth - i - 0.5) > ratio/2:
                continue
            r0 = D[i - 1, j - 1] 
            #r1 = D[i - 1, j]
            r2 = D[i, j - 1]
            min_val = min(r0, r2)
            D[i, j] = C[i, j] + min_val
    return D
    
@jit(nopython = True)
def dtw_warping_path(D):
    M, N = D.shape
    m = M - 1
    n = N - 1
    P = [(m, n)]
    while n > 0 or m > 0:
        if m == 0:
            cell = (0, n - 1)
        elif n == 0:
            cell = (m - 1, 0)
        else:
            val = min(D[m-1, n-1], D[m, n-1])
            if val == D[m-1, n-1]:
                cell = (m-1, n-1)
            else:
                cell = (m, n-1)
        P.append(cell)
        (m, n) = cell
    P.reverse()
    return np.array(P)

def dtw_decode(C, ratio=0, return_loss=False):
    D = compute_dtw(C, ratio)
    #print('hard D:', D)
    path = dtw_warping_path(D)
    if return_loss:
        return path[:, 0], D[-1,-1]
    return path[:,0]


def generate_soft_label(C, gamma=0.1, ratio=0):
    D, loss, Q = compute_softdtw(C, gamma, ratio)
    #print('D:', D)
    G = compute_softdtw_backward(C, D, Q, gamma, ratio)
    #print('G:', G)
    return G


if __name__ == '__main__':
    np.random.seed(3)
    C = np.random.rand(6, 5)
    C = C - C.min() + 1e-10
    C = np.log(C)
    C = C - C.max()
    C = -C
    #print('C:', C)
    G = generate_soft_label(C, 0.0001, 2)
    print('soft:', G.argmax(0))
    path= dtw_decode(C, ratio=2, return_loss=False)
    print('path:', path)
