import csv
import os.path
import torch


class EarlyStopping:
    no_improve = 0

    def Early_Stopping(self, metrics, patience, monitor='val_loss', verbose=False):
        early_stop = False
        metric = metrics[monitor]
        if monitor == 'val_loss' or 'train_loss':
            min_loss = min(metric)
            early_stop = False
            if metric[-1] <= min_loss:
                min_loss = metric
                self.no_improve = 0
            else:
                self.no_improve += 1

        if monitor == 'train_acc' or 'val_acc':
            max_acc = max(metric)
            early_stop = False
            if metric[-1] >= max_acc:
                max_acc = metric
                self.no_improve = 0
            else:
                self.no_improve += 1
        if self.no_improve == patience + 1:
            early_stop = True
            if verbose:
                print("The Training Stopped By Early Stopping")
                print(f"Based on {monitor}")
        return early_stop


def Model_checkpoint(path, metrics, model, monitor='val_loss', verbose=False, file_name="best.pth"):
    name = os.path.join(path, file_name)
    metric = metrics[monitor]
    if monitor == 'val_loss' or 'train_loss':
        if metric[-1] <= min(metric):
            torch.save(model.state_dict(), name)
            if verbose:
                print(f"{monitor} of Model Improved in This Epoch From {min(metric)}")
        elif verbose:
            print(f"{monitor} Did Not Improved in This Epoch")
    if monitor == 'val_acc' or 'train_acc':
        if metric[-1] >= max(metric):
            torch.save(model.state_dict(), name)
            if verbose:
                print(f"{monitor} of Model Improved in This Epoch From {max(metric)}")
        elif verbose:
            print(f"{monitor} Did Not Improved in This Epoch")


def CSV_log(path, score, filename='csv_log'):
    header = ['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss', 'LR']
    scores = [score['epoch'] + 1, score['acc'], score['loss'], score['val_acc'], score['val_loss'], score['LR']]
    file_exists = os.path.exists(os.path.join(path, filename + '.csv'))
    with open(os.path.join(path, filename + ".csv"), 'a') as csv_file:
        write = csv.writer(csv_file)
        if not file_exists:
            write.writerow(header)
            write.writerow(scores)
        else:
            write.writerow(scores)
