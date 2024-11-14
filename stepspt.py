import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import os, sys
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import sqrtm
from torch.hub import load_state_dict_from_url
import torchvision.models as models
import requests
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import configs
from models.ming import VisionTransformer, CFG

from io_utils import model_dict, parse_args

from datasets import ISIC_few_shot_da, EuroSAT_few_shot_da, CropDisease_few_shot_da, Chest_few_shot_da, Pattern_few_shot_da

from PIL import Image
from timm.models import create_model

from digit import obtain_label_ts, obtain_nearest_trace, IID_loss

import argparse
import os.path as osp
import torchvision
from torchvision import transforms
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from numpy import linalg as LA

torch.hub.set_dir('/scratch/project_2002243/huali/prompt/Code/')

scaler = torch.cuda.amp.GradScaler()


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))
 
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class CustomLayerNorm(nn.Module):
    def __init__(self, wrapped_layer):
        super(CustomLayerNorm, self).__init__()
        self.wrapped_layer = wrapped_layer
        self.scale = nn.Parameter(torch.ones_like(wrapped_layer.weight))
        self.shift = nn.Parameter(torch.zeros_like(wrapped_layer.bias))

    def forward(self, x):
        normalized = self.wrapped_layer(x)
        return normalized * self.scale + self.shift
        
class BNModel(nn.Module):
    def __init__(self, num_channels):
        super(BNModel, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        x = self.bn(x)
        return x
        
class FineTuneModel(nn.Module):
    def __init__(self, bn_model, pretrained_model, clf):
        super(FineTuneModel, self).__init__()
        self.bn = bn_model.module.bn 
        self.pretrained_model = pretrained_model
        self.clf = clf

    def forward(self, x):
        x = self.bn(x)  
        x = self.pretrained_model(x)#[0]
        out = self.clf(x)
        return x, out

def chain(features_test, outputs_test, feas_all, label_confi, classifier):
    softmax_out = nn.Softmax(dim=1)(outputs_test)

    features_test_N, _, _ = obtain_nearest_trace(features_test, feas_all, label_confi)  # equal to the "data_n" 
    features_test_N = torch.tensor(features_test_N)
    features_test_N = features_test_N.cuda()
    #features_test_N = pretrained_model(features_test_N)
    outputs_test_N = classifier(features_test_N)
    softmax_out_hyper = nn.Softmax(dim=1)(outputs_test_N)
    
    classifier_loss = torch.tensor(0.0).cuda()
    # -----------------hyper-dou------------------
    iic_loss = IID_loss(softmax_out, softmax_out_hyper)
    classifier_loss = classifier_loss + 1.0 * iic_loss

    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    gentropy_loss = gentropy_loss * 1.0
    classifier_loss = classifier_loss - gentropy_loss
    
    return classifier_loss

def finetune(img_size, novel_loader, n_query=15, freeze_backbone=False, mt='ViT-B_16', n_way=5, n_support=5):

    iter_num = len(novel_loader)

    acc_all_ori = []
    acc_all_lp = []

    loss_values_runs = []
    for _, (t_all, x_all, y_all) in enumerate(novel_loader):

        ###############################################################################################

        # download ConvNeXt
        pretrained_model = create_model('convnext_xlarge.fb_in22k', pretrained=True, num_classes=1000, 
        drop_path_rate=0.2,
        #layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        )
        pretrained_model = nn.DataParallel(pretrained_model)
        
        ###############################################################################################

        classifier = Classifier(1000, n_way)
        classifier = nn.DataParallel(classifier)

        ###############################################################################################
        
        batch_size = 5
        support_size = n_way * n_support
        n_samples = n_support + n_query
        all_size = n_way * n_samples
        x_b_i = []
        #t_b_i = []
        #t_a_i = []
        t_a_list = []
        yy = []
        
        for aug, (t, x, y) in enumerate(zip(t_all, x_all, y_all)):
            #image_input = preprocess(x).unsqueeze(0).cuda()
            #text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in t]).cuda()

            #t_b_i_list = [item for item in t[0] for _ in range(n_query)]
            #t_a_i_list = [item for item in t[0] for _ in range(n_support)]
            #t_a_i.append(t_a_i_list)
            #t_a_list = [item for sublist in t_a_i for item in sublist]

            n_query = x.size(1) - n_support
            x = x.cuda()
            x_var = Variable(x)
            x_var_i = x_var[:, :, :, :, :].contiguous().view(all_size, *x.size()[2:])
            y_var_tmp = Variable(torch.from_numpy(np.repeat(range(n_way), (n_support+n_query)))).cuda()

            y_a_i_tmp = Variable(torch.from_numpy(np.repeat(range(n_way), n_support))).cuda()

            x_b_i.append(x_var[:, n_support:, :, :, :].contiguous().view(n_way * n_query, *x.size()[2:]))
            
            
            x_a_i_tmp = x_var[:, :n_support, :, :, :].contiguous().view(n_way * n_support, *x.size()[2:])
            if aug == 0:
                x_a_i = x_a_i_tmp
                y_a_i = y_a_i_tmp
                #x_var_i = x_var_i_tmp
                y_var = y_var_tmp
            else:
                x_a_i = torch.cat((x_a_i, x_a_i_tmp), 0)
                y_a_i = torch.cat((y_a_i, y_a_i_tmp), 0)
                #x_var_i = torch.cat((x_var_i, x_var_i_tmp), 0)
                y_var = torch.cat((y_var, y_var_tmp), 0)

        
        ###############################################################################################
        bn_model = BNModel(x_a_i.shape[1])
        bn_model = nn.DataParallel(bn_model)
        finetune_model = FineTuneModel(bn_model, pretrained_model, classifier)
        #prompt_prefix = " ".join(["X"] * 12)
        loss_fn = nn.CrossEntropyLoss().cuda()
        classifier_opt = torch.optim.SGD(finetune_model.clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        bn_opt = torch.optim.SGD(finetune_model.bn.parameters(), lr=0.01)
        
        if freeze_backbone is False:
            delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr = 0.01)
        
        
        pretrained_model.cuda()
        classifier.cuda()
        finetune_model.cuda()
        
        ###############################################################################################
        
        total_epoch = 100
        support_size_all = support_size * 5
        
        finetune_model.train()
        
        loss_values = []
        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size_all)
            
            
            with torch.no_grad():
                x_var_i.cuda()
                var_features, output_var = finetune_model(x_var_i)
                feas_all, label_confi = obtain_label_ts(var_features, output_var)
                
            finetune_model.clf.train()
            finetune_model.pretrained_model.eval()
            
            for j in range(0, support_size_all, batch_size):
                classifier_opt.zero_grad()
                if freeze_backbone is False:
                    delta_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, support_size_all)]).cuda()
               
                z_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id]
                #####################################
                var_f, output_v = finetune_model(z_batch)

                loss = loss_fn(output_v, y_batch)
                
                loss = loss 

                #####################################
                scaler.scale(loss).backward()
                
                scaler.unscale_(classifier_opt)

                scaler.step(classifier_opt)
                if freeze_backbone is False:
                    scaler.step(delta_opt)
                scaler.update()
                
            
            finetune_model.clf.eval()
            finetune_model.pretrained_model.eval()
            finetune_model.bn.train()
            
            if freeze_backbone is False:
                delta_opt.zero_grad()
            bn_opt.zero_grad()
            
            var_f, output_v = finetune_model(x_var_i)
            
            loss_chain = chain(var_f, output_v, feas_all, label_confi, finetune_model.clf)
            scaler.scale(loss_chain).backward()
            
            scaler.unscale_(bn_opt)

            if freeze_backbone is False:
                scaler.step(delta_opt)
            scaler.step(bn_opt)
            scaler.update()
            
            loss_values.append(loss_chain.cpu().detach().numpy())
            loss_values_runs.append(loss_values)
        
        finetune_model.eval()
        

        scores_ori = 0
        scores_lp = 0

        y_query = np.repeat(range(n_way), n_query)

        #text_list = [item for item in t_all[0][0] for _ in range(15)]

        n_lp = len(y_query)
        del_n = int(n_lp * (1.0 - params.delta))

        out = []
        text_inputs = []
        with torch.no_grad():
            for x_b_i_tmp in x_b_i:
                img_features, output = finetune_model(x_b_i_tmp)
                scores_tmp = F.softmax(output, 1)
                scores_ori += scores_tmp

                x_lp = output.cpu().numpy()
                y_lp = scores_tmp.cpu().numpy()
                neigh = NearestNeighbors(n_neighbors=params.k_lp)
                neigh.fit(x_lp)
                d_lp, idx_lp = neigh.kneighbors(x_lp)
                d_lp = np.power(d_lp, 2)
                sigma2_lp = np.mean(d_lp)

                for i in range(n_way):
                    yi = y_lp[:, i]
                    top_del_idx = np.argsort(yi)[0:del_n]
                    y_lp[top_del_idx, i] = 0

                w_lp = np.zeros((n_lp, n_lp))
                for i in range(n_lp):
                    for j in range(params.k_lp):
                        xj = idx_lp[i, j]
                        w_lp[i, xj] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
                        w_lp[xj, i] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
                q_lp = np.diag(np.sum(w_lp, axis=1))
                q2_lp = sqrtm(q_lp)
                q2_lp = np.linalg.inv(q2_lp)
                L_lp = np.matmul(np.matmul(q2_lp, w_lp), q2_lp)
                a_lp = np.eye(n_lp) - params.alpha * L_lp
                a_lp = np.linalg.inv(a_lp)
                ynew_lp = np.matmul(a_lp, y_lp)

                scores_lp += ynew_lp
                
            #assert 1==2

        count_this = len(y_query)

        topk_scores, topk_labels = scores_ori.data.topk(1, 1, True, True)
        topk_ind_ori = topk_labels.cpu().numpy()
        top1_correct_ori = np.sum(topk_ind_ori[:, 0] == y_query)
        correct_ori = float(top1_correct_ori)
        print('BSR+DA: %f' % (correct_ori / count_this * 100))
        acc_all_ori.append((correct_ori / count_this * 100))

        
        topk_ind_lp = np.argmax(scores_lp, 1)
        top1_correct_lp = np.sum(topk_ind_lp == y_query)
        correct_lp = float(top1_correct_lp)
        print('BSR+LP+DA: %f' % (correct_lp / count_this * 100))
        acc_all_lp.append((correct_lp / count_this * 100))
        
        ###############################################################################################
    
    acc_all_ori = np.asarray(acc_all_ori)
    acc_mean_ori = np.mean(acc_all_ori)
    acc_std_ori = np.std(acc_all_ori)
    print('BSR+DA: %d Test Acc = %4.2f%% +- %4.2f%%' %
          (iter_num, acc_mean_ori, 1.96 * acc_std_ori / np.sqrt(iter_num)))

    acc_all_lp = np.asarray(acc_all_lp)
    acc_mean_lp = np.mean(acc_all_lp)
    acc_std_lp = np.std(acc_all_lp)
    print('BSR+LP+DA: %d Test Acc = %4.2f%% +- %4.2f%%' %
          (iter_num, acc_mean_lp, 1.96 * acc_std_lp / np.sqrt(iter_num)))


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('finetune')
    print(params.test_n_way)
    print(params.n_shot)

    image_size = 224
    iter_num = 600
    params.method = 'ce'

    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, n_query=15)
    params.freeze_backbone = True
    freeze_backbone = params.freeze_backbone
    model_type = params.model_type

    if params.dtarget == 'ISIC':
        print ("Loading ISIC")
        datamgr = ISIC_few_shot_da.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=True)
    elif params.dtarget == 'EuroSAT':
        print ("Loading EuroSAT")
        datamgr = EuroSAT_few_shot_da.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=True)
    elif params.dtarget == 'CropDisease':
        print ("Loading CropDisease")
        datamgr = CropDisease_few_shot_da.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=True)
    elif params.dtarget == 'ChestX':
        print ("Loading ChestX")
        datamgr = Chest_few_shot_da.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=True)
    elif params.dtarget == 'Pattern':
        print ("Loading Pattern")
        datamgr = Pattern_few_shot_da.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=True)

    print (params.dtarget)
    print (freeze_backbone)
    finetune(image_size, novel_loader, freeze_backbone=freeze_backbone, **few_shot_params)
