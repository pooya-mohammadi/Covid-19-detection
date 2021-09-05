import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from hp import load_hps
from models import load_model
from plotting import plot
from queries import query_the_oracle
from datasets.dataset_torch import CustomDataset
import os

# np.random.seed(42)
# random.seed(10)
# torch.manual_seed(999)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train(model, device, train_loader, optimizer, criterion, display=False):
    print("Training Started...")
    model.train()
    epoch_loss = 0
    train_acc = 0
    m_train = 0
    for batch in train_loader:
        data, target, _ = batch
        m_train += data.size(0)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, probe = model(data)
        loss = criterion(output.squeeze(1), target.long())
        Max, num = torch.max(output, 1)
        train_acc += torch.sum(num == target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    total_loss_train = epoch_loss / m_train
    total_acc_train = train_acc / m_train
    if display:
        print('Total loss on the train set: ', total_loss_train)
        print('Total acc on the train set: ', total_acc_train.item())
    return total_loss_train, total_acc_train.item()


def test(model, device, test_loader, criterion, display=False):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target, _ = batch
            data, target = data.to(device), target.to(device)
            output = model(data)
            Max, num = torch.max(output, 1)
            test_acc += torch.sum(num == target)
            test_loss += criterion(output.squeeze(1), target.long()).item()
            data_size = len(test_loader.dataset)
    total_test_loss = test_loss / data_size
    total_test_acc = test_acc / data_size
    if display:
        print('Accuracy on the test set: ', total_test_acc.item())
        print("Loss on the test set: ", total_test_loss)
    return total_test_loss, total_test_acc.item()


def training(classifier, train_set, test_set, optimizer, criterion, batch_size, query_size, pool_size, num_queries=10,
             query_strategy='least_confidence', interactive=True):
    train_loss = 0
    train_acc = 0
    total_train_acc = []
    total_train_loss = []
    total_test_acc = []
    total_test_loss = []
    queries = []
    for query in range(num_queries):
        print('\nQuery {}/{}'.format(query + 1, num_queries))
        print('-' * 10)

        # Query the oracle for more labels
        query_the_oracle(classifier, device, train_set, query_size=query_size, query_strategy=query_strategy,
                         interactive=interactive,
                         pool_size=pool_size)

        # Train the model on the data that has been labeled so far:
        labeled_idx = np.where(train_set.unlabeled_mask == 0)[0]
        labeled_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0,
                                    sampler=SubsetRandomSampler(labeled_idx))

        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

        previous_test_acc = 0
        current_test_acc = 1
        while current_test_acc > previous_test_acc:
            previous_test_acc = current_test_acc
            train_loss, train_acc = train(classifier, device, labeled_loader, optimizer, criterion)
            current_test_loss, current_test_acc = test(classifier, device, test_loader, criterion)

        print("Query:", query + 1, "- Train Loss:", train_loss, "- Train Accuracy:", train_acc, "- Test Accuracy:",
              current_test_acc, "- Test Loss:", current_test_loss)
        total_train_acc.append(train_acc)
        total_train_loss.append(train_loss)
        total_test_acc.append(current_test_acc)
        total_test_loss.append(current_test_loss)
        queries.append(query + 1)

        # scheduler.step(total_loss_valid)
        # metrics = {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': current_test_loss,
        #            'val_acc': current_test_acc}
        # Model_checkpoint(path='./', metrics=metrics, model=classifier,
        #                  monitor='val_acc', verbose=True,
        #                  file_name="best.pth")

        # Test the model:
        print("\nTesting The Model...")
        test(classifier, device, test_loader, criterion, display=True)

    history = {'accuracy': total_train_acc, 'loss': total_train_loss, 'val_accuracy': total_test_acc,
               'val_loss': total_test_loss, 'LR': optimizer.param_groups[0]['lr']}
    return history


def train_torch():
    hps = load_hps(dataset_dir="./covid-19", model_name='inception_resnetv2_pytorch', n_epochs=3, batch_size=16,
                   learning_rate=0.001,
                   lr_reducer_factor=0.2,
                   lr_reducer_patience=8, img_size=299, framework='pytorch', query_strategy='random',
                   query_size=5, pool_size=10)
    model = load_model(model_name=hps['model_name'])

    if hps['framework'] == 'pytorch':
        train_set, test_set = CustomDataset.preparing_datasets(hps)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hps['learning_rate'])
        reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hps['lr_reducer_factor'],
                                                                       patience=hps['lr_reducer_patience'],
                                                                       verbose=True)

        # Label the initial subset
        # query_the_oracle(model, device, train_set, query_size=20, interactive=False, query_strategy='random',
        #                  pool_size=5)
        history = training(classifier=model, train_set=train_set, test_set=test_set, optimizer=optimizer,
                           criterion=criterion,
                           batch_size=hps['batch_size'], num_queries=hps['n_epochs'],
                           query_strategy=hps['query_strategy'], query_size=hps['query_size'],
                           pool_size=hps['pool_size'],
                           interactive=False)
        plot(history)


if __name__ == 'main':
    train_torch()
