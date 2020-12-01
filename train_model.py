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

def CrossModel_triplet_loss(image_Ihash, text_Ihash, margin):
    #image_triplet_loss = torch.tensor(0.0).cuda()
    #text_triplet_loss = torch.tensor(0.0).cuda()
    #image_text_triplet_loss = torch.tensor(0.0).cuda()
    #text_image_triplet_loss = torch.tensor(0.0).cuda()
    loss = torch.tensor(0.0).cuda()
    loss.requires_grad = True
    #print(len(image_Ihash))
    label_indices = np.arange(len(image_Ihash))
    triplets = list(itertools.combinations(label_indices, 2))
    #print(triplets)
    
    length = len(triplets)

    if length:
        triplets = np.array(triplets)
        # print('triplet', triplets.shape)

        # cross model triplet loss
        # anchor-image    negative,positive-text
        image_text_I_ap_0 = (image_Ihash[triplets[:, 0]] - text_Ihash[triplets[:, 0]]).pow(2).sum(1)
        image_text_I_an_0 = (image_Ihash[triplets[:, 0]] - text_Ihash[triplets[:, 1]]).pow(2).sum(1)
        image_text_triplet_loss_0 = F.relu(margin + image_text_I_ap_0 - image_text_I_an_0).mean()
        
        image_text_I_ap_1 = (image_Ihash[triplets[:, 1]] - text_Ihash[triplets[:, 1]]).pow(2).sum(1)
        image_text_I_an_1 = (image_Ihash[triplets[:, 1]] - text_Ihash[triplets[:, 0]]).pow(2).sum(1)
        image_text_triplet_loss_1 = F.relu(margin + image_text_I_ap_1 - image_text_I_an_1).mean()
        
        text_image_I_ap_0 = (text_Ihash[triplets[:, 0]] - image_Ihash[triplets[:, 0]]).pow(2).sum(1)
        text_image_I_an_0 = (text_Ihash[triplets[:, 0]] - image_Ihash[triplets[:, 1]]).pow(2).sum(1)
        text_image_triplet_loss_0 = F.relu(margin + text_image_I_ap_0 - text_image_I_an_0).mean()
        
        text_image_I_ap_1 = (text_Ihash[triplets[:, 1]] - image_Ihash[triplets[:, 1]]).pow(2).sum(1)
        text_image_I_an_1 = (text_Ihash[triplets[:, 1]] - image_Ihash[triplets[:, 0]]).pow(2).sum(1)
        text_image_triplet_loss_1 = F.relu(margin + text_image_I_ap_1 - text_image_I_an_1).mean()

        loss = image_text_triplet_loss_0 + image_text_triplet_loss_1 + text_image_triplet_loss_0 + text_image_triplet_loss_1
    
    return loss

def calc_loss(cartoons_feature, cartoon_labels, portraits_feature, portrait_labels, hyper_parameters):
    
    cm_tri = hyper_parameters['cm_tri']
    margin = hyper_parameters['margin']
    
    #smooth_ap = SmoothAP(0.01, len(cartoons_feature), len(portraits_feature))
    
    #term1 = smooth_ap(cartoons_feature, cartoon_labels, portraits_feature, portrait_labels)
    
    term1 = CrossModel_triplet_loss(cartoons_feature, portraits_feature, margin)
    
    im_loss = cm_tri * term1

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
                    cartoons_feature, portraits_feature = model(cartoons, portraits)
                    #print('b' * 20)
                    loss = calc_loss(cartoons_feature, cartoon_labels, portraits_feature, portrait_labels, hyper_parameters)
                    #loss = calc_loss_test(cartoons_feature, portraits_feature, hyper_parameters)

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
                    
                    cartoons_feature = model(cartoons=cartoons)
                    t_cartoon_features.append(cartoons_feature.cpu().numpy())
                    t_cartoon_labels.append(cartoon_labels.cpu().squeeze(-1).numpy())
                    
                for portraits, portrait_labels in portrait_dataloader[phase]:
                    portraits = portraits.to(device)
                    portrait_labels = portrait_labels.to(device)
                            
                    portraits_feature = model(portraits=portraits)
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
            if phase == 'valid' and Sketch2Video[0] > best_recall[0]:
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

