import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, AttributedGraphDataset, WikipediaNetwork
from torch_geometric.datasets import Actor, WebKB, CitationFull, CoraFull, Flickr, FacebookPagePage
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, remove_self_loops
import os
import numpy as np
import random
import torch.nn.functional as F
import warnings

# CSR warning ignore
warnings.filterwarnings(action='ignore')

# set the seed
def set_seed(seed, cuda):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# train process
def train(epochs, model, optimizer, features, adj, labels, idx_train, idx_val):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'model')
    maxacc = 0.
    
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        # validate the model
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            
        # print train accuracy and validation accuracy
        if ep % 100 == 0:
            print(f"acc_val: {100 * acc_val:.2f}%, loss_val: {loss_val:.4f}")

        # find the model with highest val accuracy
        if acc_val > maxacc:
            saved_model = model
            maxacc = acc_val

    # save the best model
    torch.save(saved_model, path)
    
    # computes the DGR value
    dis_ratio = dis_cluster(saved_model, features, adj, labels)
    
    return maxacc, dis_ratio
    
# if you have a test set
def test(model, features, adj, idx_test, labels):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test, loss_test

# Distance Group Ratio calculation
def dis_cluster(model, features, adj, labels):
    model.eval()
    with torch.no_grad():
        X = model(features, adj)
    X_labels = []
    for i in range(labels.max().item() + 1):
        X_label = X[labels == i].data.cpu().numpy()
        h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
        h_norm[h_norm == 0.] = 1e-3
        X_label = X_label / np.sqrt(h_norm)
        X_labels.append(X_label)

    # calculate intra mean distance
    dis_intra = []
    for i in range(labels.max().item() + 1):
        x2 = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
        dists = x2 + x2.T - 2 * np.matmul(X_labels[i], X_labels[i].T)
        dis_intra.append(np.mean(dists))
    mean_dis_intra = np.mean(dis_intra)
    
    # calculate inter mean distance
    dis_inter = []
    for i in range(labels.max().item()):
        for j in range(i+1, labels.max().item() + 1):
            x2_i = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
            x2_j = np.sum(np.square(X_labels[j]), axis=1, keepdims=True)
            dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
            dis_inter.append(np.mean(dists))
    mean_dis_inter = np.mean(dis_inter)
    
    # calculate ratio
    dis_ratio = mean_dis_intra / mean_dis_inter
    dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
    
    return dis_ratio

def encode_onehot(labels):
    labels = labels.detach().cpu().numpy()
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

'''
Below number of training set and validation set are following ratio of 6:4

Cora - total_data: 2708 (trainset: 2166 val: 542), feature: 1433, edge: 10556, class: 7
Citeseer - total_data: 3327 (trainset: 2662 val: 665), feature: 3703, edge: 9104, class: 6
Pubmed - total_data: 19717 (trainset: 15774 val: 3943), feature: 500, edge: 88648, class: 3
CS - total_data: 18333 (trainset: 14666 val: 3667), feature: 6805, edge: 163788, class: 15
Physics - total_data: 34493 (trainset: 27594 val: 6899), feature: 8415, edge: 495924, class: 5
Computers - total_data: 13752 (trainset: 11002 val: 2750), feature: 767, edge: 491722, class: 10
Photo - total_data: 7650 (trainset: 6120 val: 1530), feature: 745, edge: 238162, class: 8
Wiki - total_data: 2405 (trainset: 1924 val: 481), feature: 4973, edge: 17981, class: 17
Cornell - total_data: 183 (trainset: 146 val: 37), feature: 1703, edge: 298, class: 5
Texas - total_data: 183 (trainset: 146 val: 37), feature: 1703, edge: 325, class: 5
Wisconsin - total_data: 251 (trainset: 201 val: 50), feature: 1703, edge: 515, class: 5
Chameleon - total_data: 2277 (trainset: 1822 val: 455), feature: 2325, edge: 36101, class: 5
Squirrel - total_data: 5201 (trainset: 4161 val: 1040), feature: 2089, edge: 217073, class: 5
Actor - total_data: 7600 (trainset: 6080 val: 1520), feature: 932, edge: 30019, class: 5
DBLP - total_data: 17716 (trainset: 10630 val: 7086), feature: 1639 edge: 105734, label: 4
CoraFull - total_data: 19793 (trainset: 11876 val: 7917), feature: 8710 edge: 126842, label: 70
Flickr - total_data: 89250 (trainset: 53550 val: 35700), feature: 500 edge: 899756, label: 7
Facebook - total_data: 22470 (trainset: 13482 val: 8988), feature: 128 edge: 342004, label: 4
'''
def load_data(dataset):
    
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')

    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = T.NormalizeFeatures()(Planetoid(path, dataset)[0])
    
    elif dataset in ["CS", "Physics"]:
        data = T.NormalizeFeatures()(Coauthor(path, dataset)[0])
        
    elif dataset in ["Computers", "Photo"]:
        data = T.NormalizeFeatures()(Amazon(path, dataset)[0])
        
    elif dataset in ["Wiki"]:
        data = T.NormalizeFeatures()(AttributedGraphDataset(path, dataset)[0])

    elif dataset in ['Cornell', 'Texas', 'Wisconsin']:
        data = T.NormalizeFeatures()(WebKB(path, dataset)[0])

    elif dataset in ["Chameleon", "Squirrel"]:
        data = T.NormalizeFeatures()(WikipediaNetwork(path, dataset)[0])
    
    elif dataset in ['Actor']:
        data = T.NormalizeFeatures()(Actor(path)[0])
        
    elif dataset in ['DBLP']:
        data = T.NormalizeFeatures()(CitationFull(path, dataset)[0])
        
    elif dataset in ['CoraFull']:
        data = T.NormalizeFeatures()(CoraFull(path)[0])

    elif dataset in ['Flickr']:
        data = T.NormalizeFeatures()(Flickr(path)[0])

    elif dataset in ['Facebook']:
        data = T.NormalizeFeatures()(FacebookPagePage(path)[0])

    else:
        raise Exception(f'the dataset of {dataset} has not been implemented')

    # Train and validation split
    split = T.RandomNodeSplit(num_val=0.4, num_test=0.)
    data = split(data)
    
    # Data preprosessing
    labels = encode_onehot(data.y)
    features = data.x
    edges, _ = remove_self_loops(data.edge_index)
    edges = add_self_loops(edges)[0].transpose(0,1)
    adj = normalize(sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), 
                    shape=(labels.shape[0], labels.shape[0]), dtype=np.float32))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    labels = torch.LongTensor(np.where(labels)[1])
    
    idx_train = data.train_mask
    idx_val = data.val_mask
    # idx_test = data.test_mask
    
    # present dataset information
    print(f"{dataset} - total_data: {len(features)} (trainset: {sum(data.train_mask)} val: {sum(data.val_mask)} feature: {data.x.shape[1]} edge: {data.edge_index.shape[1]}, label: {max(data.y)+1})")
    
    # If you want to use test set, add idx_test next to the idx_val
    return adj, features, labels, idx_train, idx_val
