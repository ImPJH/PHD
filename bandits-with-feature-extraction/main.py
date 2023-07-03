import numpy as np
from util import *
from cfg import get_cfg
from tqdm.auto import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchinfo import summary
from models import get_model


OPTIMIZERS = {
    'sgd': torch.optim.SGD, 
    'rmsprop': torch.optim.RMSprop, 
    'adagrad': torch.optim.Adagrad, 
    'adam': torch.optim.Adam, 
    'adamax': torch.optim.Adamax,
    'adadelta': torch.optim.Adadelta,
}


def train(model, train_loader, optimizer):
    model.train()
    if cfg.tqdm:
        tqdm_bar = tqdm(train_loader)
    else:
        tqdm_bar = train_loader
    for image, label in tqdm_bar:
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        
        if not cfg.aux:
            loss = criterion(output, label)
        else:
            if cfg.name.lower() in ["googlenet", "googleresnet"]:
                final_output, aux2, aux1 = output
                loss = criterion(final_output, label) + (0.3*criterion(aux1, label)) + (0.3*criterion(aux2, label))
            elif cfg.name.lower() == "inception-v2":
                final_output, aux = output
                loss = criterion(final_output, label) + (0.3*criterion(aux, label))
            else:
                loss = criterion(output, label)
            
        loss.backward()
        optimizer.step()
        if cfg.tqdm:
            tqdm_bar.set_description(f"Epoch {epoch+1} - train loss: {loss.item():.6f}")


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        if cfg.tqdm:
            tqdm_bar = tqdm(test_loader)
        else:
            tqdm_bar = test_loader
        for image, label in tqdm_bar:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


if __name__ == "__main__":
    cfg = get_cfg()
    if cfg.cpu:
        DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    else:
        DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print(f"Using PyTorch version: {torch.__version__}, Device: {DEVICE}")
    
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb, config=cfg)
   
    ## import dataset and model
    train_dataset = CustomDataset(train=True, prob=cfg.aug_p)
    test_dataset = CustomDataset(train=False, prob=0.0)
    
    if not cfg.aux:
        model = get_model(cfg.name.lower())(cfg.google_aux)
    else:
        model = get_model(cfg.name.lower())()
    model_name = model.__class__.__name__

    if cfg.num_workers > 1:
        train_loader = DataLoader(dataset=train_dataset, 
                                batch_size=cfg.bs, 
                                shuffle=True, 
                                collate_fn=train_dataset.collate_fn,
                                num_workers=cfg.num_workers)
        test_loader = DataLoader(dataset=test_dataset, 
                                batch_size=cfg.bs, 
                                shuffle=False, 
                                collate_fn=test_dataset.collate_fn,
                                num_workers=cfg.num_workers)
        device_ids = [i for i in range(cfg.num_workers)]
        model = nn.DataParallel(model, device_ids=device_ids).to(DEVICE)
    else:
        train_loader = DataLoader(dataset=train_dataset, 
                                batch_size=cfg.bs, 
                                shuffle=True, 
                                collate_fn=train_dataset.collate_fn)
        test_loader = DataLoader(dataset=test_dataset, 
                                batch_size=cfg.bs, 
                                shuffle=False, 
                                collate_fn=test_dataset.collate_fn)
        model = model.to(DEVICE)
    
    ## define training vars
    EPOCHS = cfg.epoch
    optimizer = OPTIMIZERS[cfg.optim](model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.factor, patience=cfg.patience)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    
    ## print model info
    model_summary = summary(
            model=model, 
            verbose=1 # 0 : no output / 1 : print model summary / 2 : full detail(weight, bias layers)
    )
    print(model_summary)
    
    ## training
    accs = np.zeros(shape=EPOCHS)
    losses = np.zeros(shape=EPOCHS)
    max_acc = 0
    saved_model = None
    fname = ""
    save_epoch = 0
    
    for epoch in range(EPOCHS):
        train(model, train_loader, optimizer)
        test_loss, test_accuracy = evaluate(model, test_loader)
        print(f"\n[EPOCH: {epoch+1}], \tModel: {model_name}, \tTest Loss: {test_loss:6f}, \tTest Accuracy: {test_accuracy * 100:.3f}%, ", end='')
        accs[epoch] = test_accuracy
        losses[epoch] = test_loss
        
        ## save point
        if cfg.save_model:
            if max_acc < accs[epoch]:
                max_acc = accs[epoch]
                saved_model = model.state_dict()
                fname = f"./saved_models/{cfg.dataset}_{model_name}_{cfg.bs}.pt"
                save_epoch = epoch + 1
            else:
                torch.save(saved_model, fname)
                print(f"Model save at epoch {save_epoch}")
        
        # scheduler.step(test_loss)
        scheduler.step(test_accuracy)
        print(f"\tLast LR: {scheduler.state_dict()['_last_lr'][-1]} \n")
        
        if cfg.use_wandb:
            wandb.log({"acc": test_accuracy, "loss": test_loss, "learning rate": scheduler.state_dict()['_last_lr'][-1]})
        
    ## plot save
    if cfg.save_plot:
        plt.figure(figsize=(15, 6))
        
        plt.subplot(121)
        plt.plot(losses)
        plt.title("Test loss")
        plt.grid(True)

        plt.subplot(122)
        plt.plot(accs)
        plt.title("Test accuracy")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"./plots/{cfg.dataset}_{model_name}_{cfg.bs}.png")
