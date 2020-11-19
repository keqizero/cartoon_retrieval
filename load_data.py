from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
import h5py
import os
import scipy.sparse as sp
import itertools
import cv2
from torchvision import transforms
import copy
import random
from PIL import Image

input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

class CustomDataSet(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        image = input_transform(cv2.imread(image))
        label = self.labels[index]
        return image, label

    def __len__(self):
        count = len(self.images)
        assert len(self.images) == len(self.labels)
        return count

flatten = lambda l: [item for sublist in l for item in sublist]

class TrainDatasetsmoothap(Dataset):
    """
    This dataset class allows mini-batch formation pre-epoch, for greater speed

    """
    def __init__(self, cartoon_dict, portrait_dict, batch_size, samples_per_class):
        """
        Args:
            image_dict: two-level dict, `super_dict[super_class_id][class_id]` gives the list of
                        image paths having the same super-label and class label
        """
        self.cartoon_dict = cartoon_dict
        self.portrait_dict = portrait_dict
        
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        
        for sub in self.cartoon_dict:
            newsub = []
            for instance in self.cartoon_dict[sub]:
                newsub.append((instance, np.array([int(sub)])))
            self.cartoon_dict[sub] = newsub
        
        for sub in self.portrait_dict:
            newsub = []
            for instance in self.portrait_dict[sub]:
                newsub.append((instance, np.array([int(sub)])))
            self.portrait_dict[sub] = newsub
        
        # checks
        # provide avail_classes
        self.avail_classes = [*self.cartoon_dict]
        # Data augmentation/processing methods.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transf_list = []
        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transf_list)

        self.reshuffle()


    def ensure_3dim(self, img):
        if len(img.size) == 2:
            img = img.convert('RGB')
        return img


    def reshuffle(self):

        cartoon_dict = copy.deepcopy(self.cartoon_dict)
        portrait_dict = copy.deepcopy(self.portrait_dict)
        print('shuffling data')
        for sub in cartoon_dict:
            random.shuffle(cartoon_dict[sub])
        for sub in portrait_dict:
            random.shuffle(portrait_dict[sub])
        
        classes = [*cartoon_dict]
        random.shuffle(classes)
        total_batches = []
        batch = []
        finished = 0
        while finished == 0:
            for sub_class in classes:
                if (len(cartoon_dict[sub_class]) >=self.samples_per_class) and (len(portrait_dict[sub_class]) >=self.samples_per_class) and (len(batch) < self.batch_size/self.samples_per_class):
                    batch_fuse = []
                    for i in range(self.samples_per_class):
                        batch_fuse.append(cartoon_dict[sub_class][i] + portrait_dict[sub_class][i])
                    batch.append(batch_fuse)
                    
                    cartoon_dict[sub_class] = cartoon_dict[sub_class][self.samples_per_class:]
                    portrait_dict[sub_class] = portrait_dict[sub_class][self.samples_per_class:]

            if len(batch) == self.batch_size/self.samples_per_class:
                total_batches.append(batch)
                batch = []
            else:
                finished = 1

        
        random.shuffle(total_batches)
        self.dataset = flatten(flatten(total_batches))

    def __getitem__(self, idx):
        # we use SequentialSampler together with SuperLabelTrainDataset,
        # so idx==0 indicates the start of a new epoch
        batch_item = self.dataset[idx]
        
        cartoon_img = Image.open(batch_item[0])
        cartoon_cls = batch_item[1]
        portrait_img = Image.open(batch_item[2])
        portrait_cls = batch_item[3]
        return self.transform(self.ensure_3dim(cartoon_img)), cartoon_cls, self.transform(self.ensure_3dim(portrait_img)), portrait_cls


    def __len__(self):
        return len(self.dataset)
    
    
def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)

def get_loader(dataset_path, cartoon_batch_size, portrait_batch_size):
    
    label = 0
    
    cartoon_trainval = []
    cartoon_label_trainval = []
    portrait_trainval = []
    portrait_label_trainval = []
    
    for (root, dirs, files) in os.walk(dataset_path):
        if len(dirs):
            continue
        for name in files:
            file_path = os.path.join(root, name)
            if name[0].lower() == 'c':
                cartoon_trainval.append(file_path)
                cartoon_label_trainval.append([label])
            else:
                portrait_trainval.append(file_path)
                portrait_label_trainval.append([label])
        label += 1
    
    num_class = label
    
    train_label = int(label * 0.8)
    label = 0
    
    cartoon_train = []
    cartoon_valid = []
    cartoon_label_train = []
    cartoon_label_valid = []
    portrait_train = []
    portrait_valid = []
    portrait_label_train = []
    portrait_label_valid = []
    
    for (root, dirs, files) in os.walk(dataset_path):
        if len(dirs):
            continue
        for name in files:
            file_path = os.path.join(root, name)
            if name[0].lower() == 'c':
                if label < train_label:
                    cartoon_train.append(file_path)
                    cartoon_label_train.append([label])
                else:
                    cartoon_valid.append(file_path)
                    cartoon_label_valid.append([label])
            else:
                if label < train_label:
                    portrait_train.append(file_path)
                    portrait_label_train.append([label])
                else:
                    portrait_valid.append(file_path)
                    portrait_label_valid.append([label])
        label += 1
    
    cartoon_train = np.asarray(cartoon_train)
    cartoon_valid = np.asarray(cartoon_valid)
    cartoon_label_train = np.asarray(cartoon_label_train)
    cartoon_label_valid = np.asarray(cartoon_label_valid)
    portrait_train = np.asarray(portrait_train)
    portrait_valid = np.asarray(portrait_valid)
    portrait_label_train = np.asarray(portrait_label_train)
    portrait_label_valid = np.asarray(portrait_label_valid)

    #cartoon_label_train = ind2vec(cartoon_label_train, num_class).astype(int)
    #cartoon_label_valid = ind2vec(cartoon_label_valid, num_class).astype(int)
    #portrait_label_train = ind2vec(portrait_label_train, num_class).astype(int)
    #portrait_label_valid = ind2vec(portrait_label_valid, num_class).astype(int)

    cartoons = {'train': cartoon_train, 'valid': cartoon_valid}
    cartoon_labels = {'train': cartoon_label_train, 'valid': cartoon_label_valid}
    portraits = {'train': portrait_train, 'valid': portrait_valid}
    portrait_labels = {'train': portrait_label_train, 'valid': portrait_label_valid}
    
    cartoon_dataset = {x: CustomDataSet(cartoons[x], cartoon_labels[x]) for x in ['train', 'valid']}
    portrait_dataset = {x: CustomDataSet(portraits[x], portrait_labels[x]) for x in ['train', 'valid']}

    shuffle = {'train': True, 'valid': False}
    
    batch_num = len(cartoon_train) // cartoon_batch_size
    portrait_batch_size = len(portrait_train) // batch_num
    print('cartoon_batch_size = ' + str(cartoon_batch_size))
    print('portrait_batch_size = ' + str(portrait_batch_size))

    cartoon_dataloader = {x: DataLoader(cartoon_dataset[x], batch_size=cartoon_batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'valid']}
    portrait_dataloader = {x: DataLoader(portrait_dataset[x], batch_size=portrait_batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'valid']}

    input_data_par = {}
    input_data_par['cartoon_train'] = cartoon_train
    input_data_par['cartoon_valid'] = cartoon_valid
    input_data_par['cartoon_label_train'] = cartoon_label_train
    input_data_par['cartoon_label_valid'] = cartoon_label_valid
    input_data_par['portrait_train'] = portrait_train
    input_data_par['portrait_valid'] = portrait_valid
    input_data_par['portrait_label_train'] = portrait_label_train
    input_data_par['portrait_label_valid'] = portrait_label_valid
    input_data_par['num_class'] = num_class
    return cartoon_dataloader, portrait_dataloader, input_data_par


def get_loader_match(dataset_path, cartoon_batch_size, portrait_batch_size):
    
    label = 0
    
    cartoon_trainval = []
    cartoon_label_trainval = []
    portrait_trainval = []
    portrait_label_trainval = []
    
    cartoon_dict, portrait_dict  = {}, {}
    for (root, dirs, files) in os.walk(dataset_path):
        if len(dirs):
            continue
        if str(label) not in cartoon_dict:
            cartoon_dict[str(label)] = []
        if str(label) not in portrait_dict:
            portrait_dict[str(label)] = []
        for name in files:
            file_path = os.path.join(root, name)
            if name[0].lower() == 'c':
                cartoon_trainval.append(file_path)
                cartoon_label_trainval.append([label])
                cartoon_dict[str(label)].append(file_path)
            else:
                portrait_trainval.append(file_path)
                portrait_label_trainval.append([label])
                portrait_dict[str(label)].append(file_path)
        label += 1
    
    num_class = label
    
    train_label = int(label * 0.8)
    label = 0
    
    cartoon_train = []
    cartoon_valid = []
    cartoon_label_train = []
    cartoon_label_valid = []
    portrait_train = []
    portrait_valid = []
    portrait_label_train = []
    portrait_label_valid = []
    
    cartoon_train_dict, portrait_train_dict  = {}, {}
    cartoon_valid_dict, portrait_valid_dict  = {}, {}
    for (root, dirs, files) in os.walk(dataset_path):
        if len(dirs):
            continue
        if label < train_label:
            if str(label) not in cartoon_train_dict:
                cartoon_train_dict[str(label)] = []
            if str(label) not in portrait_train_dict:
                portrait_train_dict[str(label)] = []
        else:
            if str(label) not in cartoon_valid_dict:
                cartoon_valid_dict[str(label)] = []
            if str(label) not in portrait_valid_dict:
                portrait_valid_dict[str(label)] = []
        for name in files:
            file_path = os.path.join(root, name)
            if name[0].lower() == 'c':
                if label < train_label:
                    cartoon_train.append(file_path)
                    cartoon_label_train.append([label])
                    cartoon_train_dict[str(label)].append(file_path)
                else:
                    cartoon_valid.append(file_path)
                    cartoon_label_valid.append([label])
                    cartoon_valid_dict[str(label)].append(file_path)
            else:
                if label < train_label:
                    portrait_train.append(file_path)
                    portrait_label_train.append([label])
                    portrait_train_dict[str(label)].append(file_path)
                else:
                    portrait_valid.append(file_path)
                    portrait_label_valid.append([label])
                    portrait_valid_dict[str(label)].append(file_path)
        label += 1
    
    cartoon_train = np.asarray(cartoon_train)
    cartoon_valid = np.asarray(cartoon_valid)
    cartoon_label_train = np.asarray(cartoon_label_train)
    cartoon_label_valid = np.asarray(cartoon_label_valid)
    portrait_train = np.asarray(portrait_train)
    portrait_valid = np.asarray(portrait_valid)
    portrait_label_train = np.asarray(portrait_label_train)
    portrait_label_valid = np.asarray(portrait_label_valid)

    #cartoon_label_train = ind2vec(cartoon_label_train, num_class).astype(int)
    #cartoon_label_valid = ind2vec(cartoon_label_valid, num_class).astype(int)
    #portrait_label_train = ind2vec(portrait_label_train, num_class).astype(int)
    #portrait_label_valid = ind2vec(portrait_label_valid, num_class).astype(int)

    cartoons = {'train': cartoon_train, 'valid': cartoon_valid}
    cartoon_labels = {'train': cartoon_label_train, 'valid': cartoon_label_valid}
    portraits = {'train': portrait_train, 'valid': portrait_valid}
    portrait_labels = {'train': portrait_label_train, 'valid': portrait_label_valid}
    
    cartoon_dataset = {x: CustomDataSet(cartoons[x], cartoon_labels[x]) for x in ['train', 'valid']}
    portrait_dataset = {x: CustomDataSet(portraits[x], portrait_labels[x]) for x in ['train', 'valid']}

    shuffle = {'train': True, 'valid': False}
    
    batch_num = len(cartoon_train) // cartoon_batch_size
    portrait_batch_size = len(portrait_train) // batch_num
    print('cartoon_batch_size = ' + str(cartoon_batch_size))
    print('portrait_batch_size = ' + str(portrait_batch_size))

    cartoon_dataloader = {x: DataLoader(cartoon_dataset[x], batch_size=cartoon_batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'valid']}
    portrait_dataloader = {x: DataLoader(portrait_dataset[x], batch_size=portrait_batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'valid']}
    
    train_dataset = TrainDatasetsmoothap(cartoon_train_dict, portrait_train_dict, cartoon_batch_size, 1)
    valid_dataset = TrainDatasetsmoothap(cartoon_valid_dict, portrait_valid_dict, cartoon_batch_size, 1)
    
    dataloader = {}
    dataloader['train'] = DataLoader(train_dataset, batch_size=cartoon_batch_size, num_workers=8, sampler=SequentialSampler(train_dataset), pin_memory=True, drop_last=True)
    dataloader['valid'] = DataLoader(valid_dataset, batch_size=10, num_workers=8, sampler=SequentialSampler(valid_dataset), pin_memory=True, drop_last=True)
    
    return dataloader, cartoon_dataloader, portrait_dataloader


def get_loader_test(dataset_path, cartoon_txt, portrait_txt, batch_size):
    
    with open(cartoon_txt, 'r') as f:
        cartoon_list = f.readlines()
    cartoon_list = [item.strip() for item in cartoon_list]
    
    with open(portrait_txt, 'r') as f:
        portrait_list = f.readlines()
    portrait_list = [item.strip() for item in portrait_list]
    
    cartoon_test = []
    cartoon_name = []
    for name in cartoon_list:
        file_path = os.path.join(dataset_path, name + '.jpg')
        cartoon_test.append(file_path)
        cartoon_name.append(name)
    
    portrait_test = []
    portrait_name = []
    for name in portrait_list:
        file_path = os.path.join(dataset_path, name + '.jpg')
        portrait_test.append(file_path)
        portrait_name.append(name)
    
    cartoon_test = np.asarray(cartoon_test)
    cartoon_name = np.asarray(cartoon_name)
    portrait_test = np.asarray(portrait_test)
    portrait_name = np.asarray(portrait_name)
    
    cartoon_dataset = CustomDataSet(cartoon_test, cartoon_name)
    portrait_dataset = CustomDataSet(portrait_test, portrait_name)

    cartoon_dataloader = DataLoader(cartoon_dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    portrait_dataloader = DataLoader(portrait_dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    
    return cartoon_dataloader, portrait_dataloader