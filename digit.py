import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from numpy import linalg as LA

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent


import sys

import torch


def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
  # has had softmax applied
  _, k = x_out.size()
  # p_i_j = compute_joint(x_out, x_tf_out)




  bn_, k_ = x_out.size()
  assert (x_tf_out.size(0) == bn_ and x_tf_out.size(1) == k_)

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  p_i_j = p_i_j / p_i_j.sum()  # normalise




  assert (p_i_j.size() == (k, k))

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                           k)  # but should be same, symmetric

  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  # p_j[(p_j < EPS).data] = EPS
  # p_i[(p_i < EPS).data] = EPS

  loss = - p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j) \
                    - lamb * torch.log(p_i))

  loss = loss.sum()

  # loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
  #                           - torch.log(p_j) \
  #                           - torch.log(p_i))

  # loss_no_lamb = loss_no_lamb.sum()

  return loss


def compute_joint(x_out, x_tf_out):
  # produces variable that requires grad (since args require grad)

  bn, k = x_out.size()
  assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  return p_i_j


def obtain_label_ts(feas_F, outputs):
    start_test = True
    with torch.no_grad():
        if start_test:
            all_fea_F = feas_F.float().cpu()
            #all_fea = feas.float().cpu()
            all_output = outputs.float().cpu()
            #all_label = labels.float()
            start_test = False
            #print("vvvvvvvvvvvvvvvv")
            #print(all_output.shape)
        else:
            all_fea_F = torch.cat((all_fea_F, feas_F.float().cpu()), 0)
            #all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
            all_output = torch.cat((all_output, outputs.float().cpu()), 0)
            #all_label = torch.cat((all_label, labels.float()), 0)

    # all_logis = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    #print("ccccccccccccc")
    #print(all_output.shape)
    #all_output = nn.Softmax(dim=0)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
    #ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=0)
    unknown_weight = 1 - ent / np.log(5)
    #_, predict = torch.max(all_output, 1)
    #predict = predict.cuda()
    #all_label = all_label.cuda()

    len_unconfi = int(ent.shape[0]*0.5)
    idx_unconfi = ent.topk(len_unconfi, largest=True)[-1]
    idx_unconfi_list_ent = idx_unconfi.cpu().numpy().tolist()
    #accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    
    #all_fea_F = torch.cat((all_fea_F, torch.ones(all_fea_F.size(0), 1)), 1)
    all_fea_F = (all_fea_F.t() / torch.norm(all_fea_F, p=2, dim=1)).t()
    all_fea_F = all_fea_F.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    #print("aaaaaaaaaaaaaaaaaa")
    initc = aff.transpose().dot(all_fea_F)
    #print(initc)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    #print(initc)
    #cls_count = np.eye(K)[predict.cpu()].sum(axis=0)
    #labelset = np.where(cls_count>0)
    #labelset = labelset[0]
    #print(labelset)

    #dd = cdist(all_fea_F, initc[labelset], 'cosine')
    dd = cdist(all_fea_F, initc, 'cosine')
    #pred_label = dd.argmin(axis=1)
    #pred_label = labelset[pred_label]

    # --------------------use dd to get confi_idx and unconfi_idx-------------
    dd_min = dd.min(axis = 1)
    dd_min_tsr = torch.from_numpy(dd_min).detach()
    dd_t_confi = dd_min_tsr.topk(int((dd.shape[0]*0.6)), largest = False)[-1]
    dd_confi_list = dd_t_confi.cpu().numpy().tolist()
    dd_confi_list.sort()
    idx_confi = dd_confi_list

    idx_all_arr = np.zeros(shape = dd.shape[0], dtype = np.int64)
    idx_all_arr[idx_confi] = 1
    idx_unconfi_arr = np.where(idx_all_arr == 0)
    idx_unconfi_list_dd = list(idx_unconfi_arr[0])

    # Get intersection
    #idx_unconfi_list = list(set(idx_unconfi_list_dd).intersection(set(idx_unconfi_list_ent)))
    #idx_unconfi_list = list(set(idx_unconfi_list_dd))   # 聚类排序
    idx_unconfi_list = list(set(idx_unconfi_list_ent))   # 熵排序
    # ------------------------------------------------------------------------
    # idx_unconfi_list = idx_unconfi_list_dd # idx_unconfi_list_dd

    label_confi = np.ones(ent.shape[0], dtype="int64")
    label_confi[idx_unconfi_list] = 0

    #acc = np.sum(pred_label == all_label.float().cpu().numpy()) / len(all_fea_F)
    #log_str = '{:.1f} AccuracyEpoch = {:.2f}% -> {:.2f}%'.format(iter_num_update_f, accuracy * 100, acc * 100)

    #args.out_file.write(log_str + '\n')
    #args.out_file.flush()
    #print(log_str+'\n')

    return all_fea_F, label_confi



def obtain_nearest_trace(data_q, data_all, lab_confi):
    data_q_ = data_q.detach()
    data_all_ = torch.tensor(data_all)
    #data_all_ = data_all.detach()
    data_q_ = data_q_.cpu().numpy()
    data_all_ = data_all_.cpu().numpy()
    num_sam = data_q.shape[0]
    LN_MEM = 70

    flag_is_done = 0         # indicate whether the trace process has done over the target dataset 
    ctr_oper = 0             # counter the operation time
    idx_left = np.arange(0, num_sam, 1)
    mtx_mem_rlt = -3*np.ones((num_sam, LN_MEM), dtype='int64')
    mtx_mem_ignore = np.zeros((num_sam, LN_MEM), dtype='int64')
    is_mem = 0
    mtx_log = np.zeros((num_sam, LN_MEM), dtype='int64')
    indices_row = np.arange(0, num_sam, 1)
    flag_sw_bad = 0 
    nearest_idx_last = np.array([-7])

    while flag_is_done == 0:

        nearest_idx_tmp, idx_last_tmp = get_nearest_sam_idx(data_q_, data_all_, is_mem, ctr_oper, mtx_mem_ignore, nearest_idx_last)
        is_mem = 1
        nearest_idx_last = nearest_idx_tmp

        if ctr_oper == (LN_MEM-1):    
            flag_sw_bad = 1
        else:
            flag_sw_bad = 0 

        mtx_mem_rlt[:, ctr_oper] = nearest_idx_tmp
        mtx_mem_ignore[:, ctr_oper] = idx_last_tmp
        
        lab_confi_tmp = lab_confi[nearest_idx_tmp]
        idx_done_tmp = np.where(lab_confi_tmp == 1)[0]
        idx_left[idx_done_tmp] = -1

        if flag_sw_bad == 1:
            idx_bad = np.where(idx_left >= 0)[0]
            mtx_log[idx_bad, 0] = 1
        else:
            mtx_log[:, ctr_oper] = lab_confi_tmp

        flag_len = len(np.where(idx_left >= 0)[0])
        # print("{}--the number of left:{}".format(str(ctr_oper), flag_len))
        
        if flag_len == 0 or flag_sw_bad == 1:
            # idx_nn_tmp = [list(mtx_log[k, :]).index(1) for k in range(num_sam)]
            idx_nn_step = []
            for k in range(num_sam):
                try:
                    idx_ts = list(mtx_log[k, :]).index(1)
                    idx_nn_step.append(idx_ts)
                except:
                    print("ts:", k, mtx_log[k, :])
                    # mtx_log[k, 0] = 1
                    idx_nn_step.append(0)

            idx_nn_re = mtx_mem_rlt[indices_row, idx_nn_step]
            data_re = data_all[idx_nn_re, :]
            flag_is_done = 1
        else:
            data_q_ = data_all_[nearest_idx_tmp, :]
        ctr_oper += 1

    return data_re, idx_nn_re, idx_nn_step # array



def get_nearest_sam_idx(Q, X, is_mem_f, step_num, mtx_ignore, nearest_idx_last_f): # Q、X arranged in format of row-vector
    Xt = np.transpose(X)
    Simo = np.dot(Q, Xt)               
    nq = np.expand_dims(LA.norm(Q, axis=1), axis=1)
    nx = np.expand_dims(LA.norm(X, axis=1), axis=0)
    Nor = np.dot(nq, nx)
    Sim = 1 - (Simo / Nor)

    # Sim = cdist(Q, X, "cosine") # too slow
    # print('eeeeee \n', Sim)

    indices_min = np.argmin(Sim, axis=1)
    indices_row = np.arange(0, Q.shape[0], 1)
    
    idx_change = np.where((indices_min - nearest_idx_last_f)!=0)[0] 
    if is_mem_f == 1:
        if idx_change.shape[0] != 0:
            indices_min[idx_change] = nearest_idx_last_f[idx_change]  
    Sim[indices_row, indices_min] = 1000

    # mytst = np.eye(795)[indices_min]
    # mytst_log = np.sum(mytst, axis=0)
    # haha = np.where(mytst_log > 1)[0]
    # if haha.size != 0:
    #     print(haha)

    # Ignore the history elements. 
    if is_mem_f == 1:
        for k in range(step_num):
            indices_ingore = mtx_ignore[:, k]
            Sim[indices_row, indices_ingore] = 1000
    
    indices_min_cur = np.argmin(Sim, axis=1)
    indices_self = indices_min
    return indices_min_cur, indices_self

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='u2m', choices=['u2m', 'm2u','s2m'])
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.05)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1) 
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--threshold', type=int, default=0)  
    parser.add_argument('--output', type=str, default='ckps_digits_iic')
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()
    args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir = osp.join(args.output, 'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
        args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_source(args)
        test_target(args)

    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(osp.join(args.output_dir, 'log_tar_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args)
'''