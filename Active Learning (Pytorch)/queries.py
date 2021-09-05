import torch
import numpy as np
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F


def query_the_oracle(model, device, dataset, query_size=10, query_strategy='random',
                     interactive=True, pool_size=0, batch_size=32, num_workers=0):
    unlabeled_idx = np.nonzero(dataset.unlabeled_mask)[0]

    # Select a pool of samples to query from
    if pool_size > 0:
        pool_idx = random.sample(range(1, len(unlabeled_idx)), pool_size)
        pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                 sampler=SubsetRandomSampler(unlabeled_idx[pool_idx]))
    else:
        pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                 sampler=SubsetRandomSampler(unlabeled_idx))

    if query_strategy == 'margin':
        sample_idx = margin_query(model, device, pool_loader, query_size)
    elif query_strategy == 'least_confidence':
        sample_idx = least_confidence_query(model, device, pool_loader, query_size)
    else:
        sample_idx = random_query(pool_loader, query_size)

    # Query the samples, one at a time
    for sample in sample_idx:

        if interactive:
            dataset.display(sample)
            print("What is the class of this image?")
            new_label = int(input())
            dataset.update_label(sample, new_label)

        else:
            dataset.label_from_filename(sample)


def random_query(data_loader, query_size=10):
    sample_idx = []

    # Because the data has already been shuffled inside the data loader,
    # we can simply return the `query_size` first samples from it
    for batch in data_loader:

        _, _, idx = batch
        sample_idx.extend(idx.tolist())

        if len(sample_idx) >= query_size:
            break

    return sample_idx[0:query_size]


def least_confidence_query(model, device, data_loader, query_size=10):
    confidences = []
    indices = []

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            data, _, idx = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)

            # Keep only the top class confidence for each sample
            most_probable = torch.max(probabilities, dim=1)[0]
            confidences.extend(most_probable.cpu().tolist())
            indices.extend(idx.tolist())

    conf = np.asarray(confidences)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(conf)
    # Return the indices corresponding to the lowest `query_size` confidences
    return ind[sorted_pool][0:query_size]


def margin_query(model, device, data_loader, query_size=10):
    margins = []
    indices = []
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            data, _, idx = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)

            # Select the top two class confidences for each sample
            toptwo = torch.topk(probabilities, 2, dim=1)[0]

            # Compute the margins = differences between the two top confidences
            differences = toptwo[:, 0] - toptwo[:, 1]
            margins.extend(torch.abs(differences).cpu().tolist())
            indices.extend(idx.tolist())

    margin = np.asarray(margins)
    index = np.asarray(indices)
    sorted_pool = np.argsort(margin)
    # Return the indices corresponding to the lowest `query_size` margins
    return index[sorted_pool][0:query_size]
