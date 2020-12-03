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

# load image and transform
class ImageDataSet(Dataset):
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

# load feature
class FeatureDataSet(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        count = len(self.images)
        assert len(self.images) == len(self.labels)
        return count
    
# load image and transform, however every class in each batch contains the same number of samples
flatten = lambda l: [item for sublist in l for item in sublist]
class ImageDataSet_Uniform(Dataset):
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
        transf_list.extend([transforms.RandomGrayscale(p=0.5), transforms.RandomHorizontalFlip(), transforms.RandomCrop((224, 224)), transforms.ToTensor(), normalize])
        #transf_list.extend([transforms.ToTensor(), normalize])
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
        # we use SequentialSampler together with SuperLabelFeatureDataSet_Uniform,
        # so idx==0 indicates the start of a new epoch
        batch_item = self.dataset[idx]
        
        cartoon_img = Image.open(batch_item[0])
        cartoon_cls = batch_item[1]
        portrait_img = Image.open(batch_item[2])
        portrait_cls = batch_item[3]
        return self.transform(self.ensure_3dim(cartoon_img)), cartoon_cls, self.transform(self.ensure_3dim(portrait_img)), portrait_cls


    def __len__(self):
        return len(self.dataset)

# load feature, however every class in each batch contains the same number of samples
class FeatureDataSet_Uniform(Dataset):
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

        self.reshuffle()

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
        # we use SequentialSampler together with SuperLabelFeatureDataSet_Uniform,
        # so idx==0 indicates the start of a new epoch
        batch_item = self.dataset[idx]
        
        cartoon_img = batch_item[0]
        cartoon_cls = batch_item[1]
        portrait_img = batch_item[2]
        portrait_cls = batch_item[3]
        return cartoon_img, cartoon_cls, portrait_img, portrait_cls


    def __len__(self):
        return len(self.dataset)    

# transform label to one-hot label
def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)

# split the dataset into trainset and validset by label, which will cause terrible overfit in validset
def get_loader_split_label(dataset_path, batch_size, num_per_cls=1):
    
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
    
    cartoon_dataset = {x: ImageDataSet(cartoons[x], cartoon_labels[x]) for x in ['train', 'valid']}
    portrait_dataset = {x: ImageDataSet(portraits[x], portrait_labels[x]) for x in ['train', 'valid']}

    shuffle = {'train': True, 'valid': False}

    cartoon_dataloader = {x: DataLoader(cartoon_dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'valid']}
    portrait_dataloader = {x: DataLoader(portrait_dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'valid']}
    
    train_dataset = ImageDataSet_Uniform(cartoon_train_dict, portrait_train_dict, batch_size, num_per_cls)
    valid_dataset = ImageDataSet_Uniform(cartoon_valid_dict, portrait_valid_dict, batch_size, num_per_cls)
    
    dataloader = {}
    dataloader['train'] = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler=SequentialSampler(train_dataset), pin_memory=True, drop_last=True)
    dataloader['valid'] = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, sampler=SequentialSampler(valid_dataset), pin_memory=True, drop_last=True)
    
    return dataloader, cartoon_dataloader, portrait_dataloader

# directly read the preexacted features as dataset
def get_loader_feature(cartoon_feature_path, portrait_feature_path, batch_size, num_per_cls=1):
    
    c_file = h5py.File(cartoon_feature_path, 'r')
    p_file = h5py.File(portrait_feature_path, 'r')
    
    all_keys = list(c_file.keys())
    
    num_class = len(all_keys)
    
    train_label = int(num_class * 0.8)
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
    
    for index, key_name in enumerate(all_keys):
        c_features = np.asarray(c_file.get(key_name + '/features'))
        p_features = np.asarray(p_file.get(key_name + '/features'))
        
        c_num = len(c_features)
        p_num = len(p_features)
        if c_num < 2 or p_num < 2:
            continue
        c_train_num = int(c_num * 0.8)
        p_train_num = int(p_num * 0.8)
        
        if str(index) not in cartoon_train_dict:
            cartoon_train_dict[str(index)] = []
        if str(index) not in portrait_train_dict:
            portrait_train_dict[str(index)] = []
        if str(index) not in cartoon_valid_dict:
            cartoon_valid_dict[str(index)] = []
        if str(index) not in portrait_valid_dict:
            portrait_valid_dict[str(index)] = []
        
        c_index = p_index = 0
        
        for c_feature in c_features:
            if c_index < c_train_num:
                cartoon_train.append(c_feature)
                cartoon_label_train.append([index])
                cartoon_train_dict[str(index)].append(c_feature)
            else:
                cartoon_valid.append(c_feature)
                cartoon_label_valid.append([index])
                cartoon_valid_dict[str(index)].append(c_feature)
            c_index += 1
            
        for p_feature in p_features:
            if p_index < p_train_num:
                portrait_train.append(p_feature)
                portrait_label_train.append([index])
                portrait_train_dict[str(index)].append(p_feature)
            else:
                portrait_valid.append(p_feature)
                portrait_label_valid.append([index])
                portrait_valid_dict[str(index)].append(p_feature)
            p_index += 1
    
    cartoon_train = np.asarray(cartoon_train)
    cartoon_valid = np.asarray(cartoon_valid)
    cartoon_label_train = np.asarray(cartoon_label_train)
    cartoon_label_valid = np.asarray(cartoon_label_valid)
    portrait_train = np.asarray(portrait_train)
    portrait_valid = np.asarray(portrait_valid)
    portrait_label_train = np.asarray(portrait_label_train)
    portrait_label_valid = np.asarray(portrait_label_valid)

    cartoon_label_train = ind2vec(cartoon_label_train, num_class).astype(int)
    cartoon_label_valid = ind2vec(cartoon_label_valid, num_class).astype(int)
    portrait_label_train = ind2vec(portrait_label_train, num_class).astype(int)
    portrait_label_valid = ind2vec(portrait_label_valid, num_class).astype(int)

    cartoons = {'train': cartoon_train, 'valid': cartoon_valid}
    cartoon_labels = {'train': cartoon_label_train, 'valid': cartoon_label_valid}
    portraits = {'train': portrait_train, 'valid': portrait_valid}
    portrait_labels = {'train': portrait_label_train, 'valid': portrait_label_valid}
    
    cartoon_dataset = {x: FeatureDataSet(cartoons[x], cartoon_labels[x]) for x in ['train', 'valid']}
    portrait_dataset = {x: FeatureDataSet(portraits[x], portrait_labels[x]) for x in ['train', 'valid']}

    shuffle = {'train': True, 'valid': False}

    cartoon_dataloader = {x: DataLoader(cartoon_dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'valid']}
    portrait_dataloader = {x: DataLoader(portrait_dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'valid']}
    
    train_dataset = FeatureDataSet_Uniform(cartoon_train_dict, portrait_train_dict, batch_size, num_per_cls)
    valid_dataset = FeatureDataSet_Uniform(cartoon_valid_dict, portrait_valid_dict, batch_size, num_per_cls)
    
    dataloader = {}
    dataloader['train'] = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler=SequentialSampler(train_dataset), pin_memory=True, drop_last=True)
    dataloader['valid'] = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, sampler=SequentialSampler(valid_dataset), pin_memory=True, drop_last=True)
    
    return dataloader, cartoon_dataloader, portrait_dataloader

# read the data for testing
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
    
    cartoon_dataset = ImageDataSet(cartoon_test, cartoon_name)
    portrait_dataset = ImageDataSet(portrait_test, portrait_name)

    cartoon_dataloader = DataLoader(cartoon_dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    portrait_dataloader = DataLoader(portrait_dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    
    return cartoon_dataloader, portrait_dataloader

# split the dataset into trainset and validset for every label
def get_loader(dataset_path, batch_size, num_per_cls=1):
    
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
            
        c_num = p_num = 0
        for name in files:
            if name[0].lower() == 'c':
                c_num += 1
            else:
                p_num += 1
        if c_num < 2 or p_num < 2:
            continue
        c_train_num = int(c_num * 0.8)
        p_train_num = int(p_num * 0.8)
        
        if str(label) not in cartoon_train_dict:
            cartoon_train_dict[str(label)] = []
        if str(label) not in portrait_train_dict:
            portrait_train_dict[str(label)] = []
        if str(label) not in cartoon_valid_dict:
            cartoon_valid_dict[str(label)] = []
        if str(label) not in portrait_valid_dict:
            portrait_valid_dict[str(label)] = []
        
        c_index = p_index = 0
        for name in files:
            file_path = os.path.join(root, name)
            if name[0].lower() == 'c':
                if c_index < c_train_num:
                    cartoon_train.append(file_path)
                    cartoon_label_train.append([label])
                    cartoon_train_dict[str(label)].append(file_path)
                else:
                    cartoon_valid.append(file_path)
                    cartoon_label_valid.append([label])
                    cartoon_valid_dict[str(label)].append(file_path)
                c_index += 1
            else:
                if p_index < p_train_num:
                    portrait_train.append(file_path)
                    portrait_label_train.append([label])
                    portrait_train_dict[str(label)].append(file_path)
                else:
                    portrait_valid.append(file_path)
                    portrait_label_valid.append([label])
                    portrait_valid_dict[str(label)].append(file_path)
                p_index += 1
        label += 1
    
    num_class = label
    
    cartoon_train = np.asarray(cartoon_train)
    cartoon_valid = np.asarray(cartoon_valid)
    cartoon_label_train = np.asarray(cartoon_label_train)
    cartoon_label_valid = np.asarray(cartoon_label_valid)
    portrait_train = np.asarray(portrait_train)
    portrait_valid = np.asarray(portrait_valid)
    portrait_label_train = np.asarray(portrait_label_train)
    portrait_label_valid = np.asarray(portrait_label_valid)
    
    cartoon_label_train = ind2vec(cartoon_label_train, num_class).astype(int)
    cartoon_label_valid = ind2vec(cartoon_label_valid, num_class).astype(int)
    portrait_label_train = ind2vec(portrait_label_train, num_class).astype(int)
    portrait_label_valid = ind2vec(portrait_label_valid, num_class).astype(int)

    cartoons = {'train': cartoon_train, 'valid': cartoon_valid}
    cartoon_labels = {'train': cartoon_label_train, 'valid': cartoon_label_valid}
    portraits = {'train': portrait_train, 'valid': portrait_valid}
    portrait_labels = {'train': portrait_label_train, 'valid': portrait_label_valid}
    
    cartoon_dataset = {x: ImageDataSet(cartoons[x], cartoon_labels[x]) for x in ['train', 'valid']}
    portrait_dataset = {x: ImageDataSet(portraits[x], portrait_labels[x]) for x in ['train', 'valid']}

    shuffle = {'train': True, 'valid': False}

    cartoon_dataloader = {x: DataLoader(cartoon_dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'valid']}
    portrait_dataloader = {x: DataLoader(portrait_dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'valid']}
    
    train_dataset = ImageDataSet_Uniform(cartoon_train_dict, portrait_train_dict, batch_size, num_per_cls)
    valid_dataset = ImageDataSet_Uniform(cartoon_valid_dict, portrait_valid_dict, batch_size, num_per_cls)
    
    dataloader = {}
    dataloader['train'] = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler=SequentialSampler(train_dataset), pin_memory=True, drop_last=True)
    dataloader['valid'] = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, sampler=SequentialSampler(valid_dataset), pin_memory=True, drop_last=True)
    
    input_data_par = {}
    input_data_par['num_class'] = num_class
    
    return dataloader, cartoon_dataloader, portrait_dataloader, input_data_par

# get a single loader
def get_single_loader(dataset_path, batch_size, pic_type='c'):
    
    label = 0
    
    train = []
    valid = []
    label_train = []
    label_valid = []
    
    train_dict, valid_dict  = {}, {}
    for (root, dirs, files) in os.walk(dataset_path):
        if len(dirs):
            continue
            
        num = 0
        for name in files:
            if name[0].lower() == pic_type:
                num += 1
        train_num = int(num * 0.8)
        
        if str(label) not in train_dict:
            train_dict[str(label)] = []
        if str(label) not in valid_dict:
            valid_dict[str(label)] = []
        
        index = 0
        for name in files:
            file_path = os.path.join(root, name)
            if name[0].lower() == pic_type:
                if index < train_num:
                    train.append(file_path)
                    label_train.append([label])
                    train_dict[str(label)].append(file_path)
                else:
                    valid.append(file_path)
                    label_valid.append([label])
                    valid_dict[str(label)].append(file_path)
                index += 1
        label += 1
    
    num_class = label
    
    train = np.asarray(train)
    valid = np.asarray(valid)
    label_train = np.asarray(label_train)
    label_valid = np.asarray(label_valid)
    
    #label_train = ind2vec(label_train, num_class).astype(int)
    #label_valid = ind2vec(label_valid, num_class).astype(int)

    pics = {'train': train, 'valid': valid}
    labels = {'train': label_train, 'valid': label_valid}
    
    dataset = {x: ImageDataSet(pics[x], labels[x]) for x in ['train', 'valid']}
    
    shuffle = {'train': True, 'valid': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'valid']}
    
    input_data_par = {}
    input_data_par['num_class'] = num_class
    
    return dataloader, input_data_par