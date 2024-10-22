# coding: utf-8
import os
import torch
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import glob
import os
import h5py

def extract_feature(device, model, img):
    img = input_transform(img).unsqueeze(0).to(device)
    out = model(img)
    feature = (activation['avgpool'].squeeze().data).cpu()
    return feature

activation = {}
def get_activation(name):
    def hook(model, input, output):
        # 如果你想feature的梯度能反向传播，那么去掉 detach（）
        activation[name] = output.detach()
    return hook

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

model = torchvision.models.inception_v3(pretrained=True)
model.avgpool.register_forward_hook(get_activation('avgpool'))
model = model.to(device).eval()

def process_images():
    OUTPUT_DIR = 'features'
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    c_h5py = h5py.File(os.path.join(OUTPUT_DIR, 'cartoon_inceptionV3.hdf5'), 'w')
    p_h5py = h5py.File(os.path.join(OUTPUT_DIR, 'portrait_inceptionV3.hdf5'), 'w')
    
    dataset_path = '/media/ckq/datasets/cartoon/train'
    
    label = 0
    
    cartoon_dict, portrait_dict  = {}, {}
    for (root, dirs, files) in os.walk(dataset_path):
        if len(dirs):
            continue
        
        print('processing label ' + str(label))
        
        if str(label) not in c_h5py.keys():
            c_group = c_h5py.create_group(str(label))
        if str(label) not in portrait_dict:
            p_group = p_h5py.create_group(str(label))
        
        c_features = []
        p_features = []
        for name in files:
            file_path = os.path.join(root, name)
            img = cv2.imread(file_path)
            feature = extract_feature(device, model, img).numpy()
            
            if name[0].lower() == 'c':
                c_features.append(feature)
            else:
                p_features.append(feature)
        
        c_features = np.asarray(c_features)
        p_features = np.asarray(p_features)
        
        c_group.create_dataset('features', data=c_features)
        p_group.create_dataset('features', data=p_features)
        
        label += 1
    
    c_h5py.close()
    p_h5py.close()

if __name__ == "__main__":

    with torch.no_grad():
        process_images()
