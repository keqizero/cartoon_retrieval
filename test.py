import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

from datetime import datetime
import torch.optim as optim
from model import IDCM_NN, ResNet50
from load_data import get_loader_test
from evaluate import test_recall1
import numpy as np
######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    dataset = 'pascal'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters
    MAX_EPOCH = 10
    cartoon_batch_size = 32
    portrait_batch_size = 64
    # batch_size = 512
    lr = 1e-4
    betas = (0.5, 0.999)
    weight_decay = 0
    hyper_parameters = {'cm_tri': 10, 'margin': 100}

    print('...Data loading is beginning...')
    
    cartoon_dataloader, portrait_dataloader = get_loader_test(dataset_path='/media/ckq/datasets/cartoon/test', cartoon_txt='/media/ckq/datasets/cartoon/cartoon_id.txt', portrait_txt='/media/ckq/datasets/cartoon/real_id.txt', batch_size=cartoon_batch_size)
    #dataloader, cartoon_dataloader, portrait_dataloader, input_data_par = get_loader_smooth(dataset_path='/media/ckq/datasets/cartoon/train', cartoon_batch_size=cartoon_batch_size, portrait_batch_size=portrait_batch_size)
    
    print('...Data loading is completed...')
    
    model = IDCM_NN().to(device)
    model.load_state_dict(torch.load('weights/best.pt'))
    model.eval()

    print('...Testing is beginning...')
    # Train and evaluate
    
    t_cartoon_features, t_cartoon_names, t_portrait_features, t_portrait_names = [], [], [], []
    with torch.no_grad():
        for cartoons, cartoon_names in cartoon_dataloader:
            cartoons = cartoons.to(device)
            cartoon_names = np.asarray(cartoon_names)
            cartoons_feature = model(cartoons=cartoons)
            t_cartoon_features.append(cartoons_feature.cpu().numpy())
            t_cartoon_names.append(cartoon_names)
            
        for portraits, portrait_names in portrait_dataloader:
            portraits = portraits.to(device)
            portrait_names = np.asarray(portrait_names)
            portraits_feature = model(portraits=portraits)
            t_portrait_features.append(portraits_feature.cpu().numpy())
            t_portrait_names.append(portrait_names)
            
    t_cartoon_features = np.concatenate(t_cartoon_features)
    t_cartoon_names = np.concatenate(t_cartoon_names)
    t_portrait_features = np.concatenate(t_portrait_features)
    t_portrait_names = np.concatenate(t_portrait_names)
    
    results = test_recall1(t_portrait_features, t_portrait_names, t_cartoon_features, t_cartoon_names)
    
    results_txt = 'results.txt'
    with open(results_txt, 'w') as f:
        for result in results:
            #print(result)
            f.write(result)
            f.write('\n')
    
    print('...Testing is completed...')
