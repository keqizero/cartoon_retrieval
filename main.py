import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

from datetime import datetime
import torch.optim as optim
from model import C2R, C2R_Se, IDCM_NN
from train_model import train_model
from load_data import get_loader, get_loader_feature
from evaluate import fx_calc_map_label, fx_calc_recall
######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters
    MAX_EPOCH = 100
    batch_size = 56
    lr = 1e-5
    betas = (0.5, 0.999)
    weight_decay = 0
    hyper_parameters = {'cm_tri': 1, 'margin': 50, 'num_per_cls': 7}

    print('...Data loading is beginning...')
    
    # the first dataloader is for training, while the next two are for validating
    dataloader, cartoon_dataloader, portrait_dataloader, input_data_par = get_loader(dataset_path='/media/ckq/datasets/cartoon/train', batch_size=batch_size, num_per_cls=hyper_parameters['num_per_cls'])
    #dataloader, cartoon_dataloader, portrait_dataloader = get_loader_feature(cartoon_feature_path='/home/sxfd91307/cartoon_retrieval/features/cartoon_pretrain.hdf5', portrait_feature_path='/home/sxfd91307/cartoon_retrieval/features/portrait_pretrain.hdf5', batch_size=batch_size, num_per_cls=hyper_parameters['num_per_cls'])
    
    print('...Data loading is completed...')
    
    model = C2R(input_data_par['num_class']).to(device)
    model.load_state_dict(torch.load('weights/best_6088.pt'))
    #model = IDCM_NN().to(device)
    
    params_to_update = list(model.parameters())
    
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas, weight_decay=weight_decay)
    scheduler = None
    #optimizer = optim.SGD(params_to_update, lr=0.00001, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,400], gamma=0.1)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    print('...Training is beginning...')
    # Train and evaluate
    
    model, img_acc_hist, loss_hist = train_model(model, dataloader, cartoon_dataloader, portrait_dataloader, hyper_parameters, optimizer, scheduler, device, MAX_EPOCH, 'best.pt')
    
    print('...Training is completed...')
