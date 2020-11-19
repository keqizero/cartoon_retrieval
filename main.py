import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

from datetime import datetime
import torch.optim as optim
from model import IDCM_NN, ResNet50
from train_model import train_model, train_model_match
from load_data import get_loader, get_loader_match
from evaluate import fx_calc_map_label, fx_calc_recall
######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    dataset = 'pascal'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters
    MAX_EPOCH = 50
    cartoon_batch_size = 40
    portrait_batch_size = 64
    # batch_size = 512
    lr = 1e-4
    betas = (0.5, 0.999)
    weight_decay = 0
    hyper_parameters = {'cm_tri': 1, 'margin': 50}

    print('...Data loading is beginning...')
    
    #cartoon_dataloader, portrait_dataloader, input_data_par = get_loader(dataset_path='/media/ckq/datasets/cartoon/train', cartoon_batch_size=cartoon_batch_size, portrait_batch_size=portrait_batch_size)
    dataloader, cartoon_dataloader, portrait_dataloader = get_loader_match(dataset_path='/media/ckq/datasets/cartoon/train', cartoon_batch_size=cartoon_batch_size, portrait_batch_size=portrait_batch_size)
    
    print('...Data loading is completed...')
    
    model = IDCM_NN().to(device)
    
    params_to_update = list(model.parameters())
    
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas, weight_decay=weight_decay)
    #scheduler = None
    #optimizer = optim.SGD(params_to_update, lr=0.0002, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,400], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-6)

    print('...Training is beginning...')
    # Train and evaluate
    
    #model, img_acc_hist, loss_hist = train_model(model, cartoon_dataloader, portrait_dataloader, hyper_parameters, optimizer, scheduler, device, MAX_EPOCH, 'best.pt')
    model, img_acc_hist, loss_hist = train_model_match(model, dataloader, cartoon_dataloader, portrait_dataloader, hyper_parameters, optimizer, scheduler, device, MAX_EPOCH, 'best.pt')
    
    print('...Training is completed...')