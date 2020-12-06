import torch
import numpy as np
from metrics import dice_coef_metric, accuracy, soft_dice_loss
import matplotlib.pyplot as plt
import time


# Manage the Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def stv(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, stv)

def TRAINING_MODEL(model_name, model, train_loader, val_loader, train_loss, optimizer, lr_scheduler, num_epochs):  
    
    loss_history  = []
    train_history = []
    val_history   = []

    for epoch in range(num_epochs):
        model.train() # Enter train mode
        
        losses = []
        train_iou = []
        train_accuracy = []
                
        if lr_scheduler:
            
            warmup_factor = 1.0 / 100
            warmup_iters = min(100, len(train_loader) - 1)
            lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        
        for i_step, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
                      
            outputs = model(data)
            
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0 
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0 
            
            train_dice = dice_coef_metric(out_cut, target.data.cpu().numpy())
            train_acc = accuracy(out_cut, target.data.cpu().numpy())
            loss = soft_dice_loss(outputs, target)
            #crossEntropy = weighted_cross_entropy_loss(outputs, target)
            losses.append(loss.item())
            #BCE.append(crossEntropy.item())
            train_iou.append(train_dice)
            train_accuracy.append(train_acc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if lr_scheduler:
                lr_scheduler.step()
 
        
        torch.save(model.state_dict(), f'{model_name}_{str(epoch)}_epoch.pt')
        val_mean_iou = compute_iou(model, val_loader)
        val_mean_acc = accuracy(model, val_loader)
        
        torch.save(model.state_dict(), f'{model_name}_epoch.pt')
        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_iou)
        
        print("Epoch [%d]" % (epoch))
        print("soft_dice_Loss :", np.array(losses).mean(),  
              "\n IoU on train:", np.array(train_iou).mean(), 
              "\n IoU on validation:", val_mean_iou)
        
    return loss_history, train_history, val_history


def compute_iou(model, loader, threshold=0.90):
    valloss = 0
    with torch.no_grad():

        for i_step, (data, target) in enumerate(loader):
            
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0
                        
            picloss = dice_coef_metric(out_cut, target.data.cpu().numpy())
            valloss += picloss

    return valloss / i_step


def plot_model_history(model_name, train_history, val_history, num_epochs):

    x = np.arange(num_epochs)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, train_history, label='train dice', lw=1, c="springgreen")
    plt.plot(x, val_history, label='validation dice', lw=1, c="deeppink")

    plt.title(f"{model_name}", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("DICE", fontsize=15)

    fn = str(int(time.time())) + ".png"
    return plt.show()
    