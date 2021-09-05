import time
from torch import nn
from models import load_model
from hp import load_hps
from datasets.dataset_torch import Dataset
from plotting import plot
import torch
from callback import Model_checkpoint

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train(ds_train, model, criterion, optimizer, device):
    model.train()
    loss_ = 0
    train_acc = 0
    num_image = 0
    for x, y_true in ds_train:
        optimizer.zero_grad()
        X = x.to(device)
        Y = y_true.to(device)
        logit, probe = model(X)
        loss = criterion(logit.squeeze(1), Y.long())
        loss_ += loss.item() * x.size(0)
        Max, num = torch.max(logit, 1)
        train_acc += torch.sum(num == Y)
        num_image += x.size(0)
        loss.backward()
        optimizer.step()
    total_loss_train = loss_ / num_image
    total_acc_train = train_acc / num_image

    return model, total_loss_train, total_acc_train.item()


def valid(ds_valid, model, criterion, device):
    model.eval()
    loss_ = 0
    valid_acc = 0
    num_image = 0
    for x, y_true in ds_valid:
        X = x.to(device)
        Y = y_true.to(device)
        logit = model(X)
        loss = criterion(logit.squeeze(1), Y.long())
        loss_ += loss.item() * x.size(0)
        Max, num = torch.max(logit, 1)
        valid_acc += torch.sum(num == Y)
        num_image += x.size(0)
    total_loss_valid = loss_ / num_image
    total_acc_valid = valid_acc / num_image
    return model, total_loss_valid, total_acc_valid.item()


def training(model, ds_train, ds_valid, criterion, optimizer, scheduler, device, epochs):
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    history = {}
    for epoch in range(epochs):
        since = time.time()
        print('\nEpoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
        model, total_loss_train, total_acc_train = train(ds_train, model, criterion, optimizer, device)
        train_losses.append(total_loss_train)
        train_accs.append(total_acc_train)
        with torch.no_grad():
            model, total_loss_valid, total_acc_valid = valid(ds_valid, model, criterion, device)
            valid_losses.append(total_loss_valid)
            valid_accs.append(total_acc_valid)

        scheduler.step(total_loss_valid)
        metrics = {'train_loss': train_losses, 'train_acc': train_accs, 'val_loss': valid_losses,
                   'val_acc': valid_accs}
        Model_checkpoint(path='./', metrics=metrics, model=model,
                         monitor='val_acc', verbose=True,
                         file_name="best.pth")

        history = {'epoch': epochs, 'accuracy': train_accs, 'loss': train_losses, 'val_accuracy': valid_accs,
                   'val_loss': valid_losses, 'LR': optimizer.param_groups[0]['lr']}
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("Epoch:", epoch + 1, "- Train Loss:", total_loss_train, "- Train Accuracy:", total_acc_train,
              "- Validation Loss:", total_loss_valid, "- Validation Accuracy:", total_acc_valid)
    return model, history


def train_torch():
    hps = load_hps(dataset_dir="./covid-19/", model_name='inception_resnetv2_pytorch', n_epochs=50, batch_size=8,
                   learning_rate=0.001,
                   lr_reducer_factor=0.2,
                   lr_reducer_patience=8, img_size=299, framework='pytorch')
    model = load_model(model_name=hps['model_name'])

    if hps['framework'] == 'pytorch':
        train_loader, val_loader, test_loader = Dataset.pytorch_preprocess(dataset_dir=hps['dataset_dir'],
                                                                           img_size=hps['img_size'],
                                                                           batch_size=hps['batch_size'],
                                                                           split_size=0.3, augment=True)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hps['learning_rate'])
        reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hps['lr_reducer_factor'],
                                                                       patience=hps['lr_reducer_patience'],
                                                                       verbose=True)
        model, history = training(model, train_loader, val_loader, criterion, optimizer,
                                  reduce_on_plateau, device, hps['n_epochs'])
        plot(history)


if __name__ == '__main__':
    train_torch()
