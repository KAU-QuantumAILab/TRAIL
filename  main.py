import os
import sys
sys.path.append(os.getcwd())
import time
import torch
import torch.optim as optim
from utils import load_data, set_seed, train, test
from models import AE_GNN
from setting import Setting


def main(args):
    # load data
    adj, features, labels, idx_train, idx_val = load_data(args.dataset)
    
    # model
    model = AE_GNN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1,
                    dropout=args.dropout, num_layer=args.layer, alpha = args.alpha, 
                    model=args.model, leaky_slop = args.leaky_slop, momentum=args.momentum)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    # idx_test = idx_test.to(device)

    t_total = time.time()
    # Train model
    maxacc, dis_ratio = train(args.epochs, model, optimizer, features, 
                              adj, labels, idx_train, idx_val, )
    take_time = time.time() - t_total

    print(f"Time: {take_time:.2f}s, Max validation acc: {100 * maxacc:.2f}%, DGR: {dis_ratio:.4f}")

    # acc_test, loss_test = test(model, features, adj, idx_test, labels)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    
    args = Setting().init_state()
    set_seed(args.seed, args.cuda)
    main(args)