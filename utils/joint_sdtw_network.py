#!/usr/bin/python3.7

import random
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from .pseudo_label_generator import generate_soft_label, dtw_decode
from .soft_dtw_loss import SoftDTW_Loss
from scipy.stats import truncnorm,  beta, norm
import time
from .sopl import SOPL_Layer
from .dwsa_loss import DWSA_Loss


# buffer for old sequences (robustness enhancement: old frames are sampled from the buffer during training)
class Buffer(object):

    def __init__(self, buffer_size, n_classes):
        self.features = []
        self.transcript = []
        self.framelabels = []
        self.instance_counts = []
        self.label_counts = []
        self.buffer_size = buffer_size
        self.n_classes = n_classes
        self.next_position = 0
        self.frame_selectors = []

    def add_sequence(self, features, transcript, framelabels):
        if len(self.framelabels) < self.buffer_size:
            # sequence data 
            #self.features.append(features)
            self.transcript.append(transcript)
            self.framelabels.append(framelabels)
            # statistics for prior and mean lengths
            #self.instance_counts.append( np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] ) )
            self.label_counts.append( np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] ) )
            self.next_position = (self.next_position + 1) % self.buffer_size
        else:
            # sequence data
            #self.features[self.next_position] = features
            self.transcript[self.next_position] = transcript
            self.framelabels[self.next_position] = framelabels
            # statistics for prior and mean lengths
            #self.instance_counts[self.next_position] = np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] )
            self.label_counts[self.next_position] = np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] )
            self.next_position = (self.next_position + 1) % self.buffer_size
        # update frame selectors
        #self.frame_selectors = []
        #for seq_idx in range(len(self.features)):
        #    self.frame_selectors += [ (seq_idx, frame) for frame in range(self.features[seq_idx].shape[1]) ]

    def random(self):
        return random.choice(self.frame_selectors) # return sequence_idx and frame_idx within the sequence

    def n_frames(self):
        return len(self.frame_selectors)


class GRUNet(nn.Module):

    def __init__(self, input_dim, hidden_size, n_classes):
        super(GRUNet, self).__init__()
        self.n_classes = n_classes
        self.gru = nn.GRU(input_dim, hidden_size, 1, bidirectional = False, batch_first = True)
        self.prototypes = nn.Parameter(torch.randn(n_classes, hidden_size))

    def forward(self, x):
        output, h_n = self.gru(x)
        return output


class Forwarder(object):

    def __init__(self, input_dimension, n_classes, hidden_size=64):
        self.n_classes = n_classes
        self.net = GRUNet(input_dimension, hidden_size, n_classes)
        self.net.cuda()

    def forward(self, sequence):
        sequence = sequence.cuda()
        return self.net(sequence)

    def load_model(self, model_file):
        self.net.cpu()
        self.net.load_state_dict( torch.load(model_file) )
        self.net.cuda()


class Trainer(Forwarder):

    def __init__(self, input_dimension, n_classes, buffer_size, hidden_size=64, buffered_frame_ratio = 25, transcripts=None, use_time_prior=True, time_prior_type='beta', time_prior_weight=1):
        super(Trainer, self).__init__(input_dimension, n_classes, hidden_size)
        self.buffer = Buffer(buffer_size, n_classes)
        self.buffered_frame_ratio = buffered_frame_ratio
        self.criterion = nn.NLLLoss()
        self.prior = np.ones((self.n_classes), dtype=np.float32) / self.n_classes
        self.mean_lengths = np.ones((self.n_classes), dtype=np.float32)
        self.grad = None
        self.transcripts = transcripts
        self.time_dist_param = None
        self.use_time_prior = use_time_prior
        self.sopl_layer = SOPL_Layer(n_classes, hidden_size, n_iter=20, alpha=1, time_prior_weight=time_prior_weight, seed=None, dtype=torch.float, device='cuda')
        self.dwsa_loss_fn = DWSA_Loss(alpha=0.01, center_norm=False, sim='euc', threshold=100, softmax='no')
        self.time_prior_type = time_prior_type
        self.time_prior_weight = time_prior_weight

    def update_mean_lengths(self):
        self.mean_lengths = np.zeros( (self.n_classes), dtype=np.float32 )
        for label_count in self.buffer.label_counts:
            self.mean_lengths += label_count
        instances = np.zeros((self.n_classes), dtype=np.float32)
        for instance_count in self.buffer.instance_counts:
            instances += instance_count
        # compute mean lengths (backup to average length for unseen classes)
        self.mean_lengths = np.array( [ self.mean_lengths[i] / instances[i] if instances[i] > 0 else sum(self.mean_lengths) / sum(instances) for i in range(self.n_classes) ] )


    def update_prior(self):
        # count labels
        self.prior = np.zeros((self.n_classes), dtype=np.float32)
        for label_count in self.buffer.label_counts:
            self.prior += label_count
        self.prior = self.prior / np.sum(self.prior)
        # backup to uniform probability for unseen classes
        n_unseen = sum(self.prior == 0)
        self.prior = self.prior * (1.0 - float(n_unseen) / self.n_classes)
        self.prior = np.array( [ self.prior[i] if self.prior[i] > 0 else 1.0 / self.n_classes for i in range(self.n_classes) ] )

    def update_time_dist(self):
        self.time_dist = {k: [] for k in range(self.n_classes)}
        for framelabel in self.buffer.framelabels:
            time = np.arange(len(framelabel)) / len(framelabel)
            for k in range(self.n_classes):
                time_k = time[np.array(framelabel)==k]
                self.time_dist[k].extend(time_k)
        self.time_dist_param = dict()
        for k in range(self.n_classes):
            time_dist_k = self.time_dist[k]
            if len(time_dist_k) == 0:
                #mean_k, std_k = 0.5, 0.5 / np.sqrt(3)
                mean_k, std_k = 0.5, -1 
            else:
                mean_k = np.mean(time_dist_k)
                std_k = np.std(time_dist_k)
            self.time_dist_param[k] = (mean_k, std_k)

    def truncate_gauss(self, x, mean, std, clip_a=0, clip_b=1):
        if mean < clip_a or mean > clip_b or std <= 0:
            return np.ones_like(x)
        #a, b = (clip_a - mean) / std, (clip_b - mean) / std
        #y =  truncnorm.pdf(x, a, b, loc = mean, scale = std)
        y =  norm.pdf(x, loc = mean, scale = std)
        y = y / max(y.sum(), 1e-10)
        return y

    def beta_generate(self, x, mu, sigma):
        if sigma <= 0:
            return np.ones_like(x) / x.shape[0]
        a = (1 - mu) * mu**2 / sigma**2 - mu
        b = a / mu - a
        y = beta.pdf(x, a, b)
        y = y / max(y.sum(), 1e-10)
        return y.astype(np.float32)

    def acquire_time_prior(self, T):
        time_prior = np.ones([T, self.n_classes], dtype=np.float32) / T
        if self.time_dist_param is None:
            return time_prior
        t_range = ((np.arange(T) + 0.5) / T).astype(np.float32)
        """
           assume background and actions are beta distribution
        """
        for k in range(0, self.n_classes):
            if self.time_prior_type == 'beta':
                y = self.beta_generate(t_range, self.time_dist_param[k][0], self.time_dist_param[k][1]) 
            elif self.time_prior_type == 'gauss':
                y = self.truncate_gauss(t_range, self.time_dist_param[k][0], self.time_dist_param[k][1]) 
            time_prior[:, k] = np.clip(y, a_min=1e-40, a_max=1)
        return time_prior

    def train(self, sequences, transcripts, un_sequences, optimizer, ratio=0, iteration=0, dwsa_type='min'):
        #print('--------------------new video-----------------')
        # forwarding and Viterbi decoding
        optimizer.zero_grad()
        prev_time = time.time()
        #print('before forward:', time.time())
        loss = 0
        loss_hinge_sum = 0
        loss_pos_sum = 0
        loss_neg_sum = 0
        for sequence, transcript in zip(sequences, transcripts):
            sequence = torch.unsqueeze(torch.tensor(sequence).t(), 0).float().cuda()
            features = self.net(sequence)[0] 
            distance_origin = torch.sum((features[:,None,:] - self.net.prototypes)**2, axis=-1)
            #log_distance = nn.functional.log_softmax(distance, dim=-1) 
            #print('after forward:', time.time())
            #print('distance_origin:', distance_origin)
            #print('distance_origin:', distance_origin.min(), distance_origin.max(), distance_origin.mean(), distance_origin.std())


            video_length = features.shape[0]
            #print('before acquire time prior:', time.time())
            if self.use_time_prior:# and iteration>300:
                time_prior = - torch.log(torch.tensor(self.acquire_time_prior(video_length)).cuda())
                #print('time_prior:', time_prior)
                #print('time_prior:', time_prior.min(), time_prior.max(), time_prior.mean(), time_prior.std())
                distance_origin = distance_origin + self.time_prior_weight * time_prior # - torch.log(torch.tensor(time_prior).cuda())

            distance = distance_origin[:, torch.tensor(transcript)].t()
            #distance = nn.functional.softmax(distance, 0)

            loss_fn = SoftDTW_Loss(gamma=0.1, ratio=ratio)
            loss_pos = loss_fn(distance) / video_length
            neg_transcripts = random.sample([t for t in self.transcripts if not t==transcript], min(5, len(self.transcripts)-1))
            loss_neg = 0
            loss_hinge = 0
            for neg_transcript in neg_transcripts:
                loss_neg = loss_fn(distance_origin[:, torch.tensor(neg_transcript)].t()) / video_length
                loss_hinge += nn.ReLU()(loss_pos - loss_neg + 20)
                loss_neg_sum += loss_neg
            loss += 0.2*loss_pos + loss_hinge
            loss_hinge_sum += loss_hinge
            loss_pos_sum += loss_pos
            path = dtw_decode(distance.detach().cpu().numpy(), ratio=ratio)
            labels = np.array(transcript)[path]

            # add sequence to buffer
            self.buffer.add_sequence(features, transcript, labels)
            self.update_prior()
            if self.use_time_prior:# and iteration>0:
                self.update_time_dist()
        loss = loss / 5
        loss_hinge_sum = loss_hinge_sum / 5
        loss_pos_sum = loss_pos_sum / 5
        loss_neg_sum = loss_neg_sum / 25
        
        if dwsa_type is not None:
            un_features = []
            for un_sequence in un_sequences: 
                un_sequence = torch.unsqueeze(torch.tensor(un_sequence).t(), 0).float().cuda()
                un_feature = self.net(un_sequence)[0] 
                un_features.append(un_feature)
            un_centers = self.sopl_layer(un_features, init=True, sort_centers=True)
            #print(un_centers)
            loss_match_list = []
            for transcript in transcripts:
                #transcript = torch.tensor(transcript)
                transcript = torch.tensor(transcript[1:-1])
                #transcript = transcript[transcript>0]
                #print('transcript:', transcript)
                centers = self.net.prototypes[transcript]
                #print('centers:', centers.shape, un_centers.shape)
                loss_match = self.dwsa_loss_fn(un_centers, centers)
                loss_match_list.append(loss_match)
                #print('loss_match:', loss_match)
            if dwsa_type == 'min':
                loss_match = torch.stack(loss_match_list).min()
            elif dwsa_type == 'mean':
                loss_match = torch.stack(loss_match_list).mean()
            else:
                print('Invalid DWSA Type:', dwsa_type)
                exit()
            #print('loss_match_min:', loss_match_min)
            #print('loss_match_mean:', loss_match_mean)
            #print('loss:', loss)
            lambda_match = 1 * min(iteration / 200, 1)
            loss = loss + lambda_match * loss_match
        else:
            loss_match = torch.tensor([0])

        loss.backward()
        optimizer.step()
        #print('after updating time dist:', time.time())
        #return loss.item() / video_length, 0
        return loss.item(), loss_hinge_sum.item(), loss_pos_sum.item(), loss_neg_sum.item(), loss_match.item()


    def save_model(self, network_file, length_file, prior_file, time_file=None):
        self.net.cpu()
        try:
            torch.save(self.net.state_dict(), network_file, _use_new_zipfile_serialization=False)
        except:
            torch.save(self.net.state_dict(), network_file)
        self.net.cuda()
        np.savetxt(length_file, self.mean_lengths)
        np.savetxt(prior_file, self.prior)
        if self.use_time_prior:
            try:
                time_params = [self.time_dist_param[k] for k in range(self.n_classes)]
                np.savetxt(time_file, time_params)
            except:
                print('Failure: did not save time params!')

        self.net.cuda()
        np.savetxt(length_file, self.mean_lengths)
        np.savetxt(prior_file, self.prior)
        if self.use_time_prior:
            try:
                time_params = [self.time_dist_param[k] for k in range(self.n_classes)]
                np.savetxt(time_file, time_params)
            except:
                print('Failure: did not save time params!')

