import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

from datetime import datetime
import torch.optim as optim
from model import C2R, C2R_Se
from load_data import get_loader_test, get_loader
from evaluate import test_recall1, fx_calc_recall, fx_calc_map_label
import numpy as np
######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters
    batch_size = 16
    
    test_compute = True
    valid_compute = True
    
    model = C2R(122).to(device)
    model.load_state_dict(torch.load('weights/best.pt'))
    model.eval()

    print('...Testing is beginning...')
    # Train and evaluate
    
    if test_compute:
        
        print('...Data loading is beginning...')
        
        cartoon_dataloader, portrait_dataloader = get_loader_test(dataset_path='/media/ckq/datasets/cartoon/test', cartoon_txt='/media/ckq/datasets/cartoon/cartoon_id.txt', portrait_txt='/media/ckq/datasets/cartoon/real_id.txt', batch_size=batch_size)
        
        print('...Data loading is completed...')
        
        t_cartoon_features, t_cartoon_names, t_portrait_features, t_portrait_names = [], [], [], []
        with torch.no_grad():
            for cartoons, cartoon_names in cartoon_dataloader:
                cartoons = cartoons.to(device)
                cartoon_names = np.asarray(cartoon_names)
                cartoons_feature, _ = model(cartoons=cartoons)
                t_cartoon_features.append(cartoons_feature.cpu().numpy())
                t_cartoon_names.append(cartoon_names)
            
            for portraits, portrait_names in portrait_dataloader:
                portraits = portraits.to(device)
                portrait_names = np.asarray(portrait_names)
                portraits_feature, _ = model(portraits=portraits)
                t_portrait_features.append(portraits_feature.cpu().numpy())
                t_portrait_names.append(portrait_names)
            
        t_cartoon_features = np.concatenate(t_cartoon_features)
        t_cartoon_names = np.concatenate(t_cartoon_names)
        t_portrait_features = np.concatenate(t_portrait_features)
        t_portrait_names = np.concatenate(t_portrait_names)
        
        results = test_recall1(t_cartoon_features, t_cartoon_names, t_portrait_features, t_portrait_names)
    
        results_txt = 'results.txt'
        with open(results_txt, 'w') as f:
            for result in results:
                #print(result)
                f.write(result)
                f.write('\n')
    
        print('...Testing is completed...')
    
    if valid_compute:
        
        print('...Data loading is beginning...')
        
        dataloader, cartoon_dataloader, portrait_dataloader, _ = get_loader(dataset_path='/media/ckq/datasets/cartoon/train', batch_size=batch_size, num_per_cls=1)
        
        print('...Data loading is completed...')
        
        t_cartoon_features, t_cartoon_names, t_portrait_features, t_portrait_names = [], [], [], []
        with torch.no_grad():
            for cartoons, cartoon_names in cartoon_dataloader['valid']:
                cartoons = cartoons.to(device)
                cartoon_names = cartoon_names.to(device)
                #cartoon_names = np.asarray(cartoon_names)
                cartoons_feature, _ = model(cartoons=cartoons)
                t_cartoon_features.append(cartoons_feature.cpu().numpy())
                t_cartoon_names.append(cartoon_names.cpu().squeeze(-1).numpy())
            
            for portraits, portrait_names in portrait_dataloader['valid']:
                portraits = portraits.to(device)
                portrait_names = portrait_names.to(device)
                #portrait_names = np.asarray(portrait_names)
                portraits_feature, _ = model(portraits=portraits)
                t_portrait_features.append(portraits_feature.cpu().numpy())
                t_portrait_names.append(portrait_names.cpu().squeeze(-1).numpy())
            
        t_cartoon_features = np.concatenate(t_cartoon_features)
        t_cartoon_names = np.concatenate(t_cartoon_names)
        t_portrait_features = np.concatenate(t_portrait_features)
        t_portrait_names = np.concatenate(t_portrait_names)
        
        Sketch2Video_map = fx_calc_map_label(t_cartoon_features, t_cartoon_names, t_portrait_features, t_portrait_names)
        Sketch2Video = fx_calc_recall(t_cartoon_features, t_cartoon_names, t_portrait_features, t_portrait_names)
        print('Sketch2Video: mAP = {:.4f} R1 = {:.4f} R5 = {:.4f} R10 = {:.4f}'.format(Sketch2Video_map, Sketch2Video[0], Sketch2Video[1], Sketch2Video[2]))
        
        print('...Validating is completed...')
        