from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import time
import copy
from evaluate import fx_calc_map_label, fx_calc_recall
import torch.nn.functional as F
import numpy as np
import itertools
import os
#from Smooth_AP_loss import SmoothAP
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim

# def cos(x, y):
#     return x.mm(y.t())

def CrossModel_triplet_loss(view1_feature, view2_feature, margin, num_per_cls):
    loss = torch.tensor(0.0).cuda()
    loss.requires_grad = True
    
    cls_num = len(view1_feature) // num_per_cls
    
    label_indices = np.arange(cls_num)
    triplets = list(itertools.combinations(label_indices, 2))
    
    length = len(triplets)
    
    view1_feature_avg = []
    view2_feature_avg = []
    for i in range(cls_num):
        view1_group = view1_feature[i * num_per_cls:(i + 1) * num_per_cls]
        view1_group_avg = view1_group.mean(0)
        view1_feature_avg.append(view1_group_avg.unsqueeze(0))
        
        view2_group = view2_feature[i * num_per_cls:(i + 1) * num_per_cls]
        view2_group_avg = view2_group.mean(0)
        view2_feature_avg.append(view2_group_avg.unsqueeze(0))
    
    view1_feature_avg = torch.cat(view1_feature_avg)
    view2_feature_avg = torch.cat(view2_feature_avg)

    if length:
        triplets = np.array(triplets)
        # print('triplet', triplets.shape)

        # cross model triplet loss
        # anchor-image    negative,positive-text
        image_text_I_ap_0 = (view1_feature_avg[triplets[:, 0]] - view2_feature_avg[triplets[:, 0]]).pow(2).sum(-1)
        image_text_I_an_0 = (view1_feature_avg[triplets[:, 0]] - view2_feature_avg[triplets[:, 1]]).pow(2).sum(-1)
        image_text_triplet_loss_0 = F.relu(margin + image_text_I_ap_0 - image_text_I_an_0).mean()
        
        image_text_I_ap_1 = (view1_feature_avg[triplets[:, 1]] - view2_feature_avg[triplets[:, 1]]).pow(2).sum(-1)
        image_text_I_an_1 = (view1_feature_avg[triplets[:, 1]] - view2_feature_avg[triplets[:, 0]]).pow(2).sum(-1)
        image_text_triplet_loss_1 = F.relu(margin + image_text_I_ap_1 - image_text_I_an_1).mean()
        
        text_image_I_ap_0 = (view2_feature_avg[triplets[:, 0]] - view1_feature_avg[triplets[:, 0]]).pow(2).sum(-1)
        text_image_I_an_0 = (view2_feature_avg[triplets[:, 0]] - view1_feature_avg[triplets[:, 1]]).pow(2).sum(-1)
        text_image_triplet_loss_0 = F.relu(margin + text_image_I_ap_0 - text_image_I_an_0).mean()
        
        text_image_I_ap_1 = (view2_feature_avg[triplets[:, 1]] - view1_feature_avg[triplets[:, 1]]).pow(2).sum(-1)
        text_image_I_an_1 = (view2_feature_avg[triplets[:, 1]] - view1_feature_avg[triplets[:, 0]]).pow(2).sum(-1)
        text_image_triplet_loss_1 = F.relu(margin + text_image_I_ap_1 - text_image_I_an_1).mean()

        loss = image_text_triplet_loss_0 + image_text_triplet_loss_1 + text_image_triplet_loss_0 + text_image_triplet_loss_1
    
    return loss

def CrossModel_triplet_loss_hard_center(view1_feature, view2_feature, margin, num_per_cls):
    loss = torch.tensor(0.0).cuda()
    loss.requires_grad = True
    
    cls_num = len(view1_feature) // num_per_cls
    
    view1_feature_avg = []
    view2_feature_avg = []
    for i in range(cls_num):
        view1_group = view1_feature[i * num_per_cls:(i + 1) * num_per_cls]
        view1_group_avg = view1_group.mean(0)
        view1_feature_avg.append(view1_group_avg.unsqueeze(0))
        
        view2_group = view2_feature[i * num_per_cls:(i + 1) * num_per_cls]
        view2_group_avg = view2_group.mean(0)
        view2_feature_avg.append(view2_group_avg.unsqueeze(0))
    
    view1_feature_avg = torch.cat(view1_feature_avg)
    view2_feature_avg = torch.cat(view2_feature_avg)
    
    view1_tri_loss = []
    for index, view1 in enumerate(view1_feature_avg):
        d_all = (view1 - view2_feature_avg).pow(2).sum(-1)
        d_p = d_all[index].max()
        if index == 0:
            d_n = d_all[(index + 1):].min()
        elif index == cls_num - 1:
            d_n = d_all[:index].min()
        else:
            d_n1 = d_all[:index].min()
            d_n2 = d_all[(index + 1):].min()
            if d_n1 > d_n2:
                d_n = d_n2
            else:
                d_n = d_n1
        view1_tri_loss.append(F.relu(margin + d_p - d_n).unsqueeze(0))
    view1_tri_loss = torch.cat(view1_tri_loss)
    
    loss = view1_tri_loss.mean()
    
    return loss

def CrossModel_triplet_loss_hard(view1_feature, view2_feature, margin, num_per_cls):
    loss = torch.tensor(0.0).cuda()
    loss.requires_grad = True
    
    cls_num = len(view1_feature) // num_per_cls
    
    view1_tri_loss = []
    for index, view1 in enumerate(view1_feature):
        cur_cls = index // num_per_cls
        view2_index = cur_cls * num_per_cls
        d_all = (view1 - view2_feature).pow(2).sum(-1)
        d_p = d_all[view2_index:(view2_index + num_per_cls)].max()
        if cur_cls == 0:
            d_n = d_all[(view2_index + num_per_cls):].min()
        elif cur_cls == cls_num - 1:
            d_n = d_all[:view2_index].min()
        else:
            d_n1 = d_all[:view2_index].min()
            d_n2 = d_all[(view2_index + num_per_cls):].min()
            if d_n1 > d_n2:
                d_n = d_n2
            else:
                d_n = d_n1
        view1_tri_loss.append(F.relu(margin + d_p - d_n).unsqueeze(0))
    view1_tri_loss = torch.cat(view1_tri_loss)
    
    loss = view1_tri_loss.mean()
    
    return loss

def CrossModel_quadruplet_loss_hard(view1_feature, view2_feature, margin_pn, margin_nn, num_per_cls):
    loss = torch.tensor(0.0).cuda()
    loss.requires_grad = True
    
    cls_num = len(view1_feature) // num_per_cls
    
    view1_tri_loss = []
    for index, view1 in enumerate(view1_feature):
        cur_cls = index // num_per_cls
        view2_index = cur_cls * num_per_cls
        d_all = (view1 - view2_feature).pow(2).sum(-1)
        d_p = d_all[view2_index:(view2_index + num_per_cls)].max()
        if cur_cls == 0:
            d_n = d_all[(view2_index + num_per_cls):].min()
            d_n_index = d_all[(view2_index + num_per_cls):].argmin() + (view2_index + num_per_cls)
        elif cur_cls == cls_num - 1:
            d_n = d_all[:view2_index].min()
            d_n_index = d_all[:view2_index].argmin()
        else:
            d_n1 = d_all[:view2_index].min()
            d_n2 = d_all[(view2_index + num_per_cls):].min()
            if d_n1 > d_n2:
                d_n = d_n2
                d_n_index = d_all[(view2_index + num_per_cls):].argmin() + (view2_index + num_per_cls)
            else:
                d_n = d_n1
                d_n_index = d_all[:view2_index].argmin()
        
        cur_cls_n = d_n_index // num_per_cls
        view2_index_n = cur_cls_n * num_per_cls
        view2_n = view2_feature[d_n_index]
        d_all_n = (view2_n - view2_feature).pow(2).sum(-1)
        if cur_cls_n == 0:
            d_nn = d_all_n[(view2_index_n + num_per_cls):].min()
        elif cur_cls_n == cls_num - 1:
            d_nn = d_all_n[:view2_index_n].min()
        else:
            d_nn1 = d_all_n[:view2_index_n].min()
            d_nn2 = d_all_n[(view2_index_n + num_per_cls):].min()
            if d_nn1 > d_nn2:
                d_nn = d_nn2
            else:
                d_nn = d_nn1
        
        view1_tri_loss.append(F.relu(margin_pn + d_p - d_n).unsqueeze(0) + F.relu(margin_nn + d_p - d_nn).unsqueeze(0))
    view1_tri_loss = torch.cat(view1_tri_loss)
    
    loss = view1_tri_loss.mean()
    
    return loss

def CrossModel_center_loss(view1_feature, view2_feature, num_per_cls):
    loss = torch.tensor(0.0).cuda()
    loss.requires_grad = True
    
    cls_num = len(view1_feature) // num_per_cls
    
    view1_feature_var = []
    view2_feature_var = []
    for i in range(cls_num):
        view1_group = view1_feature[i * num_per_cls:(i + 1) * num_per_cls]
        view1_group_avg = view1_group.mean(0)
        view1_var = (view1_group - view1_group_avg).pow(2).sum(-1).mean()        
        view1_feature_var.append(view1_var.unsqueeze(0))
        
        view2_group = view2_feature[i * num_per_cls:(i + 1) * num_per_cls]
        view2_group_avg = view2_group.mean(0)
        view2_var = (view2_group - view2_group_avg).pow(2).sum(-1).mean()        
        view2_feature_var.append(view2_var.unsqueeze(0))
    
    view1_feature_var = torch.cat(view1_feature_var).mean()
    view2_feature_var = torch.cat(view2_feature_var).mean()
    
    loss = view1_feature_var + view2_feature_var
    
    return loss

def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, hyper_parameters):
    
    cm_tri = hyper_parameters['cm_tri']
    margin = hyper_parameters['margin']
    num_per_cls = hyper_parameters['num_per_cls']
    
    term1 = CrossModel_triplet_loss_hard(view1_feature, view2_feature, margin, num_per_cls)
    #term1 = CrossModel_quadruplet_loss_hard(view1_feature, view2_feature, margin, 20, num_per_cls)
    #term2 = CrossModel_center_loss(view1_feature, view2_feature, num_per_cls)
    
    im_loss = cm_tri * term1
    #criteria = AngleLoss()
    #criteria = nn.CrossEntropyLoss()
    #criteria = criteria.cuda()
    #term2 = criteria(view1_predict, labels_1.squeeze()) + criteria(view1_predict, labels_2.squeeze())
    #im_loss = term2
    
    return im_loss

def train_model(model, dataloader, cartoon_dataloader, portrait_dataloader, hyper_parameters, optimizer, scheduler=None, device="cpu", num_epochs=500, save_name='best.pt'):
    since = time.time()
    test_sketch_acc_history = []
    epoch_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_recall = [0.0, 0.0, 0.0]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for cartoons, cartoon_labels, portraits, portrait_labels in dataloader[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    cartoons = cartoons.to(device)
                    cartoon_labels = cartoon_labels.to(device)
                    portraits = portraits.to(device)
                    portrait_labels = portrait_labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    #print('a' * 20)

                    # Forward
                    cartoons_feature, portraits_feature, cartoons_predict, portraits_predict = model(cartoons, portraits)
                    #print('b' * 20)
                    loss = calc_loss(cartoons_feature, portraits_feature, cartoons_predict, portraits_predict, cartoon_labels, portrait_labels, hyper_parameters)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(cartoon_dataloader[phase].dataset)
            if phase == 'train':
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                continue
            
            t_cartoon_features, t_cartoon_labels, t_portrait_features, t_portrait_labels = [], [], [], []
            with torch.no_grad():
                for cartoons, cartoon_labels in cartoon_dataloader[phase]:
                    cartoons = cartoons.to(device)
                    cartoon_labels = cartoon_labels.to(device)
                    
                    cartoons_feature, _ = model(cartoons=cartoons)
                    t_cartoon_features.append(cartoons_feature.cpu().numpy())
                    t_cartoon_labels.append(cartoon_labels.cpu().squeeze(-1).numpy())
                    
                for portraits, portrait_labels in portrait_dataloader[phase]:
                    portraits = portraits.to(device)
                    portrait_labels = portrait_labels.to(device)
                            
                    portraits_feature, _ = model(portraits=portraits)
                    t_portrait_features.append(portraits_feature.cpu().numpy())
                    t_portrait_labels.append(portrait_labels.cpu().squeeze(-1).numpy())
            t_cartoon_features = np.concatenate(t_cartoon_features)
            t_cartoon_labels = np.concatenate(t_cartoon_labels)
            t_portrait_features = np.concatenate(t_portrait_features)
            t_portrait_labels = np.concatenate(t_portrait_labels)
            
            Sketch2Video_map = fx_calc_map_label(t_cartoon_features, t_cartoon_labels, t_portrait_features, t_portrait_labels)
            #Video2Sketch = fx_calc_map_label(t_videos, t_sketches, t_labels)
            Sketch2Video = fx_calc_recall(t_cartoon_features, t_cartoon_labels, t_portrait_features, t_portrait_labels)
            #Video2Sketch = fx_calc_recall(t_videos, t_sketches, t_labels)

            #print('{} Loss: {:.4f} Sketch2Video: {:.4f}  Video2Sketch: {:.4f}'.format(phase, epoch_loss, Sketch2Video, Video2Sketch))
            print('{} Loss: {:.4f} Cartoon2Real: mAP = {:.4f} R1 = {:.4f} R5 = {:.4f} R10 = {:.4f}'.format(phase, epoch_loss, Sketch2Video_map, Sketch2Video[0], Sketch2Video[1], Sketch2Video[2]))

            # deep copy the model
            #Sketch2Video_mean = np.mean(Sketch2Video)
            if phase == 'valid' and Sketch2Video_map > best_acc:
                best_acc = Sketch2Video_map
                best_recall = Sketch2Video
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                test_sketch_acc_history.append(Sketch2Video_map)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))
    print('Best recall: R1 = {:.4f} R5 = {:.4f} R10 = {:.4f}'.format(best_recall[0], best_recall[1], best_recall[2]))
    
    save_folder = 'weights'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(best_model_wts, os.path.join(save_folder, save_name))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, test_sketch_acc_history, epoch_loss_history


def train_single_model(model, dataloader, hyper_parameters, optimizer, scheduler=None, device="cpu", num_epochs=500, save_name='best.pt'):
    since = time.time()
    test_sketch_acc_history = []
    epoch_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    criteria = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for pics, labels in dataloader[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    pics = pics.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    #print('a' * 20)

                    # Forward
                    _, pred = model(pics)
                    #print('b' * 20)
                    loss = criteria(pred, labels.squeeze())
                    
                    #loss = calc_loss_test(cartoons_feature, portraits_feature, hyper_parameters)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            if phase == 'train':
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                continue
            
            acc = 0.0
            with torch.no_grad():
                t = 0
                a = 0
                for pics, labels in dataloader[phase]:
                    pics = pics.to(device)
                    labels = labels.to(device)
                    _, pred = model(pics)
                    result = pred.argmax(-1)
                    true_pred = len(torch.nonzero(result==labels.squeeze(), as_tuple=False))
                    t += true_pred
                    a += len(pics)
                    
                acc = t / a
            
            print('{} Loss: {:.4f} Acc = {:.4f}'.format(phase, epoch_loss, acc))

            # deep copy the model
            #Sketch2Video_mean = np.mean(Sketch2Video)
            if phase == 'valid' and acc > best_acc:
                best_acc = acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                test_sketch_acc_history.append(acc)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))
    
    save_folder = 'weights'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(best_model_wts, os.path.join(save_folder, save_name))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, test_sketch_acc_history, epoch_loss_history
