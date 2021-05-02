#!/usr/bin/python3.7

import numpy as np
from utils.dataset import SemiDataset
from utils.joint_sdtw_network import Trainer, Forwarder
import os
import math
import torch.optim as optim
import time
import argparse



class Logger():
    def __init__(self, log_name, printout=True):
        self.log_name = log_name
        self.printout = printout
        log_dir = os.path.dirname(log_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if os.path.exists(log_name):
            print("Existing log name:", log_name)
            self.log_name = log_name + "_" + str(time.time())

    def add_log(self, *log_info):
        if self.printout:
            print(*log_info)
        with open(self.log_name, 'a') as f:
            f.write(" ".join([str(i) for i in log_info]) + "\n")

data_path = '/mnt/raptor/yuhan/Breakfast_per_task/'

"""
Note: all results using DWSA before Thu Apr  8 02:38:34 UTC 2021 are wrong!

"""
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--max_epochs', type=int, default=500)
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
max_epochs = args.max_epochs
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


for task in sorted(os.listdir(data_path+'features')):#[1:]:
    ### read label2index mapping and index2label mapping ###########################
    log_name = 'logs_' + split + '/' + task + '/log_' + fname_format
    logger = Logger(log_name, printout=True)
    label2index = dict()
    index2label = dict()
    with open(os.path.join(data_path, 'mapping', task, 'mapping.txt'), 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]
    
    ### read training data #########################################################
    with open(os.path.join(data_path, 'splits', task, split + '.train'), 'r') as f:
        video_list = f.read().split('\n')[0:-1]
    #print('Loading {} videos...'.format(len(video_list)))
    logger.add_log('Loading {} videos...'.format(len(video_list)))
    labeled_num = int(len(video_list) * wl_ratio)
    un_video_list = video_list[labeled_num:]
    video_list = video_list[:labeled_num]
    ul_to_wl = int((1 - wl_ratio) / wl_ratio)
    dataset = SemiDataset(data_path, task, video_list, un_video_list, label2index, shuffle=True, batch_size=batch_size, un_batch_size=batch_size*ul_to_wl)
    logger.add_log('Videos loaded!')
    
    ### generate path grammar for inference ########################################
    paths = set()
    for _, transcripts, _ in dataset:
        for transcript in transcripts:
            paths.add( ' '.join([index2label[index] for index in transcript]) )
    result_dir = os.path.join('results_' + split, task)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    with open(result_dir+'/grammar_wl{}.txt'.format(wl_ratio), 'w') as f:
        f.write('\n'.join(paths) + '\n')
    transcripts = [[label2index[l] for l in path.split()] for path in paths]
    
    trainer = Trainer(dataset.input_dimension, dataset.n_classes, buffer_size = len(dataset), buffered_frame_ratio = 25, transcripts=transcripts, use_time_prior=use_time_prior, time_prior_type=time_prior_type, time_prior_weight=time_prior_weight)
    
    n_videos = len(video_list)
    iter_per_epoch = math.ceil(n_videos/batch_size)
    #print('ratio =', ratio)
    logger.add_log('ratio =', ratio)
    for name, param in trainer.net.named_parameters():
        print(name, param.shape)
    optimizer = optim.Adam(trainer.net.parameters(), lr = learning_rate, weight_decay=1e-6)
    # train for 100000 iterations
    loss_list = []
    best_loss = 1e5
    for i in range(iter_per_epoch * max_epochs):
        n_epoch = (i+1) // iter_per_epoch
        if i+1 == int(0.4 * max_epochs) * iter_per_epoch:
            optimizer = optim.Adam(trainer.net.parameters(), lr = 0.2*learning_rate, weight_decay=1e-6)
        features, transcripts, un_features = dataset.get()
        loss, loss_hinge, loss_pos, loss_neg, loss_match = trainer.train(features, transcripts, un_features, optimizer, ratio=ratio, iteration=i, dwsa_type=dwsa_type)
        #loss = loss1-loss2 #np.abs(loss1-loss2+10000/sequence.shape[1])
        #loss_list.append([loss1, loss2, loss])
        loss_list.append([loss, loss_hinge, loss_pos, loss_neg, loss_match])
        # print some progress information
        if (i+1) % 5 == 0:
            #print('Iteration %d, loss1: %f, loss2: %f, loss: %f' % (i+1, loss1, loss2, loss))
            logger.add_log('Iteration %d, loss hinge: %f, loss pos: %f, loss neg: %f, loss dwsa: %f, loss: %f' % (i+1, loss_hinge, loss_pos, loss_neg, loss_match, loss))
        if (i+1) % iter_per_epoch == 0:
            avg_loss = np.mean(loss_list, axis=0)
            #print('Epoch %d -- Average loss, avg loss1: %f, avg loss2: %f, avg loss: %f' % (n_epoch, avg_loss[0], avg_loss[1], avg_loss[2]))
            logger.add_log('Epoch %d -- Average loss, avg loss hinge: %f, avg loss pos: %f, avg loss neg: %f, avg loss dwsa: %f, avg loss: %f' % (n_epoch, avg_loss[1], avg_loss[2], avg_loss[3], avg_loss[4], avg_loss[0]))
            loss_list = []
        # save model every 1000 iterations
        if (i+1) % (10 * iter_per_epoch) == 0:
            file_pre = result_dir + '/' + fname_format 
            network_file = file_pre + '_network.epoch-' + str(n_epoch) + '.net'
            length_file =  file_pre + '_lengths.epoch-' + str(n_epoch) + '.txt'
            prior_file =   file_pre + '_prior.epoch-' + str(n_epoch) + '.txt'
            time_file  =   file_pre + '_time.epoch-' + str(n_epoch) + '.txt'
            trainer.save_model(network_file, length_file, prior_file, time_file)
            if avg_loss[0] < best_loss:
                network_file = file_pre + '_network.epoch-best.net'
                length_file =  file_pre + '_lengths.epoch-best.txt'
                prior_file =   file_pre + '_prior.epoch-best.txt'
                time_file  =   file_pre + '_time.epoch-best.txt'
                trainer.save_model(network_file, length_file, prior_file, time_file)
                best_loss = avg_loss[0]
                logger.add_log('New best model: Epoch %d -- Average loss, avg loss hinge: %f, avg loss pos: %f, avg loss neg: %f, avg loss dwsa: %f, avg loss: %f' % (n_epoch, avg_loss[1], avg_loss[2], avg_loss[3], avg_loss[4], avg_loss[0]))

        # adjust learning rate after 60000 iterations
        #if (i+1) == 60000:
        #    learning_rate = learning_rate * 0.1
