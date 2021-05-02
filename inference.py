#!/usr/bin/python3.7

import os
import numpy as np
import torch
from utils.dataset import Dataset
from utils.joint_sdtw_network import Forwarder
from utils.grammar import PathGrammar
from utils.pseudo_label_generator import compute_softdtw, dtw_decode
from scipy.stats import truncnorm, beta, norm
import argparse



def truncate_gauss(x, mean, std, clip_a=0, clip_b=1):
    if mean < clip_a or mean > clip_b or std <= 0:
        return np.ones_like(x)
    y =  norm.pdf(x, loc = mean, scale = std)
    y = y / max(y.sum(), 1e-10)
    return y

def beta_generate(x, mu, sigma):
    if sigma <= 0:
        return np.ones_like(x) / x.shape[0]
    a = (1 - mu) * mu**2 / sigma**2 - mu
    b = a / mu - a
    y = beta.pdf(x, a, b)
    y = y / max(y.sum(), 1e-10)
    return y.astype(np.float32)

def acquire_time_prior(T, n_classes, time_dist_param, time_prior_type='beta'):
    time_prior = np.ones([T, n_classes], dtype=np.float32) / T
    if time_dist_param is None:
        return time_prior
    t_range = ((np.arange(T) + 0.5) / T).astype(np.float32)
    """
       assume background and actions are beta distribution
    """
    for k in range(0, n_classes):
        if time_prior_type == 'beta':
            y = beta_generate(t_range, time_dist_param[k][0], time_dist_param[k][1]) 
        elif time_prior_type == 'gauss':
            y = truncate_gauss(t_range, time_dist_param[k][0], time_dist_param[k][1]) 
        time_prior[:, k] = np.clip(y, a_min=1e-10, a_max=1)
    return time_prior

def remove_cons_dup(array):
    '''
       remove consecutive duplicates in an array
    '''
    if array.shape[0] == 0:
        return array
    return array[np.where(np.insert(np.diff(array), 0, 1)!=0)]

def write_result(result, index2label, result_dir):
    for video in result:
        # save result
        stn_score = result[video][0]
        stn_labels = result[video][1]
        stn_segments = remove_cons_dup(np.array(stn_labels))
        with open(result_dir + '/' + video, 'w') as f:
            f.write( '### Recognized sequence: ###\n' )
            f.write( ' '.join( [index2label[s] for s in stn_segments] ) + '\n' )
            f.write( '### Score: ###\n' + str(stn_score) + '\n')
            f.write( '### Frame level recognition: ###\n')
            f.write( ' '.join( [index2label[l] for l in stn_labels] ) + '\n' )

### read label2index mapping and index2label mapping ###########################

def infer_task(task):
    label2index = dict()
    index2label = dict()
    with open(os.path.join(data_path, 'mapping', task, 'mapping.txt'), 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]
    
    ### read test data #############################################################
    with open(os.path.join(data_path, 'splits', task, split+'.test'), 'r') as f:
        video_list = f.read().split('\n')[0:-1]
    #print('Inference on %d videos' % len(video_list))
    dataset = Dataset(data_path, task, video_list, label2index, shuffle = False)
    
    # load prior, length model, grammar, and network
    forwarder = Forwarder(dataset.input_dimension, dataset.n_classes)
    result_dir = os.path.join('results_' + split, task)
    grammar = PathGrammar(result_dir+'/grammar_wl{}.txt'.format(wl_ratio), label2index)
    fname_pre = '/' + fname_format

    try:
        time_dist_param =  np.loadtxt(result_dir+fname_pre+'_time.epoch-' + str(load_epoch) + '.txt')
    except:
        time_dist_param = None
    forwarder.load_model(result_dir+fname_pre+'_network.epoch-' + str(load_epoch) + '.net')
    window = 10
    step = 5
    
    transcripts = grammar._read_transcripts(result_dir+'/grammar_wl{}.txt'.format(wl_ratio), label2index)
    
    log_probs = dict()
    result = dict()
    for i, data in enumerate(dataset):
        sequence, true_transcript = data
        video_length = sequence.shape[1]
        sequence = torch.unsqueeze(torch.tensor(sequence).t(), 0).float().cuda()
        video = list(dataset.features.keys())[i]
        features = forwarder.forward(sequence)[0]
        distance_origin = torch.sum((features[:,None,:] - forwarder.net.prototypes)**2, axis=-1).data.cpu().numpy()
        #print(task, 'before:', distance_origin.argmax(1))
        if time_dist_param is not None:
            time_prior = acquire_time_prior(video_length, dataset.n_classes, time_dist_param, time_prior_type=time_prior_type)
            distance_origin = distance_origin - time_prior_weight * np.log(time_prior)

        best_record = ([], 1e5, [])
        for transcript in transcripts:
            distance = distance_origin[:, np.array(transcript)] 
            _, loss, _ = compute_softdtw(distance.T, gamma=0.1, ratio=ratio) 
            loss = loss / distance.shape[0]
            if loss < best_record[1]:
                path = dtw_decode(distance.T, ratio=ratio)
                labels = np.array(transcript)[path]
                best_record = (transcript, loss, labels)
        result[video] = (best_record[1], best_record[2])
    
    write_result(result, index2label, result_dir)

data_path = '/mnt/raptor/yuhan/Breakfast_per_task/'
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('-e', '--load_epoch', default=500)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--cuda_device', default='7')
parser.add_argument('--band_ratio', type=int, default=4)
parser.add_argument('--wl_ratio', type=float, default=0.5)
parser.add_argument('--dwsa_type', default=None)
parser.add_argument('--time_prior_type', default='beta')
parser.add_argument('--time_prior_weight', type=float, default=0.2)
parser.add_argument('--split', default='split1')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
learning_rate = args.lr
load_epoch = args.load_epoch
batch_size = args.batch_size
ratio = args.band_ratio
wl_ratio = args.wl_ratio
dwsa_type = args.dwsa_type
time_prior_type = args.time_prior_type
time_prior_weight = args.time_prior_weight
use_time_prior = True if time_prior_weight > 0 else False
split = args.split

if dwsa_type is None:
    fname_format = 'joint_sdtw_{}wldata_0.2pos_hinge_m20_{}time_r{}_lr{:.0e}'.format(wl_ratio, time_prior_weight, ratio, learning_rate)
elif dwsa_type == 'min':
    fname_format = 'joint_sdtw_{}wldata_0.2pos_hinge_m20_{}time_r{}_dwsamin_lr{:.0e}'.format(wl_ratio, time_prior_weight, ratio, learning_rate)
elif dwsa_type == 'mean':
    fname_format = 'joint_sdtw_{}wldata_0.2pos_hinge_m20_{}time_r{}_dwsamean_lr{:.0e}'.format(wl_ratio, time_prior_weight, ratio, learning_rate)
if __name__ == "__main__":

    for task in sorted(os.listdir(data_path+'/features')):
        infer_task(task)
    #os.system("python eval.py --recog_dir results_{}/".format(split))
