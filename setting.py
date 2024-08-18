import argparse

class Setting():
    
    def __init__(self):
        pass
    
    def init_state(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=int, default = 0,
                            help= 'cuda number')        
        parser.add_argument('--epochs', type=int, default=1200,
                            help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=5e-4,
                            help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--dropout', type=float, default=0.6,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--layer', type=int, default = 128,
                            help= 'Number of hidden layer')
        parser.add_argument('--hidden', type=int, default=128,
                            help='Number of hidden layer dimension.')
        parser.add_argument('--leaky_slop', type=float, default = 0.3,
                            help= 'Leaky ReLU negative slope')        
        parser.add_argument('--alpha', type=float, default = 0.35,
                            help= 'proportion of vector')
        parser.add_argument('--momentum', type=float, default = 0.05,
                            help= 'batch norm momentum')        
        parser.add_argument('--seed', type=float, default = 42,
                            help= 'set the seed')
        parser.add_argument('--model', type=str, default = 'GCN',
                            help= 'graph model GCN/GraphSage/SGC')
        parser.add_argument('--dataset', type=str, default = 'Cora',
                            help= 'set the dataset')

        args = parser.parse_args()
        
        return args
    
'''
now available datasets
Cora, Citeseer, Pubmed, CS, Physics, Computers, Photo, Wiki, Cornell, Texas, 
Wisconsin, Chameleon, Squirrel, Actor, DBLP, CoraFull, Flickr, Facebook
'''
'''
seed
42, 200 ~ 1500
'''