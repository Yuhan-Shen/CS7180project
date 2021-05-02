#!/usr/bin/python

import argparse
import glob
import re
import os
import numpy as np
from utils.metrics import ComputeMetrics




def collect_list(array):
    res = []
    start = 0
    i = -1
    for i in range(len(array)-1):
        if not array[i+1] == array[i]:
            res.append((array[i], i-start+1))
            start = i+1
    if i >= 0:
        res.append((array[i+1], i-start+2))
    return res

action_count_dict = dict()
gt_action_count_dict = dict()
def recog_file(filename, ground_truth_path):

    # read ground truth
    gt_file = ground_truth_path + re.sub('.*/','/',filename+'.txt')
    with open(gt_file, 'r') as f:
        ground_truth = f.read().split('\n')[0:-1:15]
        f.close()
    # read recognized sequence
    with open(filename, 'r') as f:
        recognized = f.read().split('\n')[5].split() # framelevel recognition is in 6-th line of file
        f.close()

    col_gt  = collect_list(ground_truth)
    col_rec = collect_list(recognized)
    #print('gt:', col_gt)
    #print('pred:', col_rec)
    for step in col_rec:
        action_count_dict[step[0]] = action_count_dict.get(step[0], 0) + step[1]
    for step in col_gt:
        gt_action_count_dict[step[0]] = gt_action_count_dict.get(step[0], 0) + step[1]

    nonbg_correct = 0
    nonbg_frames = 0
    n_frame_correct = 0
    n_pred_bg_frames = len([r for r in recognized if r=='SIL'])
    for i in range(len(recognized)):
        if recognized[i] == ground_truth[i]:
            n_frame_correct += 1
        if not ground_truth[i] == 'SIL':
            nonbg_frames += 1
            if recognized[i] == ground_truth[i]:
                nonbg_correct += 1

    return n_frame_correct, len(recognized), nonbg_correct, nonbg_frames, n_pred_bg_frames

def get_gt_pred(filename, ground_truth_path, label2index):

    # read ground truth
    gt_file = ground_truth_path + re.sub('.*/','/',filename+'.txt')
    with open(gt_file, 'r') as f:
        ground_truth = f.read().split('\n')[0:-1]
        gt = [label2index[label] for label in ground_truth][::15]
        f.close()
    # read recognized sequence
    with open(filename, 'r') as f:
        recognized = f.read().split('\n')[5].split() # framelevel recognition is in 6-th line of file
        pred = [label2index[label] for label in recognized]
        f.close()
    
    return gt, pred

### MAIN #######################################################################

### arguments ###
### --recog_dir: the directory where the recognition files from inferency.py are placed
### --ground_truth_dir: the directory where the framelevel ground truth can be found
parser = argparse.ArgumentParser()
parser.add_argument('--recog_dir', default='results/')
parser.add_argument('--ground_truth_dir', default='/mnt/raptor/yuhan/Breakfast_per_task/groundTruth/')
args = parser.parse_args()
n_steps_dict = {'cereals': 4, 'coffee': 6, 'friedegg': 8, 'juice': 7, 'milk': 4, 'pancake': 13, 'salat': 7, 'sandwich': 8, 'scrambledegg': 11, 'tea': 6}

def eval_task(task):
    label2index = dict()
    index2label = dict()
    with open(os.path.join(data_path, 'mapping', task, 'mapping.txt'), 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]
                
    filelist = glob.glob(args.recog_dir + task + '/' + 'P*')
    
    rslt_dict = dict()
    
    n = 0
    
    metric_types = ["accuracy", "accuracy_wo_bg", "overlap_score", "IoD", "IoD_wo_bg", "IoU", "IoU_wo_bg", "macro_accuracy", "overlap_f1"]#, "IoU_mAP"]
    metricer = ComputeMetrics(metric_types, bg_class=0, n_classes=n_steps_dict[task]+1)
    gt_list = []
    pred_list = []
    for filename in filelist:
        gt, pred = get_gt_pred(filename, args.ground_truth_dir+task+'/', label2index)
        assert len(gt) == len(pred)
        gt_list += gt
        pred_list += pred
        n += len(gt)
        rslt_dict[filename] = (gt, pred)
        metricer.add_predictions(filename, np.array(pred), np.array(gt))
    
    n_classes=n_steps_dict[task]+1
    gt_array = np.array(gt_list)
    pred_array = np.array(pred_list)

    metric_list = []
    for k in range(n_classes):
        gt_K = (gt_array == k).sum()
        pred_K = (pred_array == k).sum()
        correct_K = ((gt_array == k) * (pred_array == k)).sum()
        pre_K = correct_K / max(pred_K, 1e-10)
        rec_K = correct_K / max(gt_K, 1e-10)
        F1_K = 2 * pre_K * rec_K / max(pre_K + rec_K, 1e-10)
        #print('class {}: precision = {:.2%}, recall = {:.2%}, F1 = {:.2%}, sum = {}, {}'.format(k, pre_K, rec_K, F1_K, gt_K, pred_K))
        metric_list.append([pre_K, rec_K, F1_K])
    avg_metric = np.mean(metric_list, axis=0)
    nonbg_avg_metric = np.mean(metric_list[1:], axis=0)

    #metricer.print_scores(metric_types=["IoD", "IoD_wo_bg", "IoU", "IoU_wo_bg"])
    scores = metricer.get_scores(metric_types=["IoD", "IoD_wo_bg", "IoU", "IoU_wo_bg"])
    
    n_frames = 0
    n_correct = 0
    n_nonbg_frames = 0
    n_nonbg_correct = 0
    n_pred_bg_frames = 0
    # loop over all recognition files and evaluate the frame error
    for filename in filelist:
        correct, frames, nonbg_correct, nonbg_frames, pred_bg_frames = recog_file(filename, args.ground_truth_dir+task+'/')
        n_correct += correct
        n_frames += frames
        n_nonbg_correct += nonbg_correct
        n_nonbg_frames += nonbg_frames
        n_pred_bg_frames += pred_bg_frames
    
    recall = float(n_nonbg_correct) / n_nonbg_frames
    prec = float(n_nonbg_correct) / (n_frames - n_pred_bg_frames)
    F1 = 2 * recall * prec / (max(recall + prec, 1e-10))
    my_scores = [100 * n_correct / n_frames, 100 * recall, 100 * prec, 100 * F1, 100 * float(n_pred_bg_frames) / n_frames]
    scores =  my_scores[:-1] + [avg_metric[-1] * 100, nonbg_avg_metric[-1] * 100] + [my_scores[-1]] + scores[2:] + scores[:2]
    return scores


if __name__ == "__main__":

    scores_list = []
    metrics_types = ["MoF", "Recall", "Prec", "F1-ag", "F-sp+bg", "F1-sp", "bgratio", "IoU", "IoU-bg", "IoD", "IoD-bg"]
    print("{:12s}, {:7s}, {:7s}, {:7s}, {:7s}, {:7s}, {:7s}, {:7s}, {:7s}, {:7s}, {:7s}, {:7s}".format("task", *metrics_types))
    for task in sorted(os.listdir(data_path+'/features')):
        scores = eval_task(task)
        print("{:12s},".format(task), ', '.join(['{:7.4f}'.format(score) for score in scores]))
        scores_list.append(scores)
    avg_scores = np.mean(scores_list, axis=0)
    print("{:12s},".format("Average"), ', '.join(['{:7.4f}'.format(score) for score in avg_scores]))
