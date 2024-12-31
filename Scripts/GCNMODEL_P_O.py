# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:33:00 2024

@author: Li
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import  argparse
import  scipy.sparse as sp
from    torch import optim
from road_dataloader_P_O import CustomDataset
from torch.utils.data import Dataset, DataLoader
import ast

def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, inputs):
        # print('inputs:', inputs)
        
        x, support = inputs

        #if self.training:
        #    x = F.dropout(x, self.dropout)
        x = torch.mm(x,support)
        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                xw = torch.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight
        #print(xw.shape)
        #out = torch.mm(xw,support)
        out=xw
        if self.bias is not None:
            out += self.bias

        return self.activation(out), support
class SimpleFullyConnected(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(SimpleFullyConnected, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim*2)
        self.bn1=nn.BatchNorm1d(hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.bn2=nn.BatchNorm1d(hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*2) 
        #self.bn3=nn.BatchNorm1d(hidden_dim*2)
        self.dropout = nn.Dropout(dropout)  
        self.fc4 = nn.Linear(hidden_dim, output_dim) 
    def forward(self, x):
        x=x.float()
        x = F.relu(self.bn1(self.fc1(x)))  
        x = self.dropout(x)      
        x = F.relu(self.bn2((self.fc2(x))) )  
        x = self.dropout(x)      
        #x = F.tanh(self.bn3(self.fc3(x)))   
        x = torch.tanh(self.fc4(x))  
        return x
    
class GCN(nn.Module):


    def __init__(self, input_dim, output_dim, num_features_nonzero):
        super(GCN, self).__init__()

        self.input_dim = input_dim 
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)


        self.layers = nn.Sequential(GraphConvolution(self.input_dim, args.hidden, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False),

                                    GraphConvolution(args.hidden, output_dim, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False),

                                    )

    def forward(self, inputs):
        #x=x.float()
        x, support = inputs
        x=x.float()
        x = self.layers((x, support))

        return x[0]

    def l2_loss(self):

        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss
class FCGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, output_dim):
        super(FCGCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1=nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim1*2)
        self.bn2=nn.BatchNorm1d(hidden_dim1*2)
        self.fc3 = nn.Linear(hidden_dim1*2, args.hidden) 
        self.dropout = nn.Dropout(args.dropout)  
        self.gcnlayers = nn.Sequential(GraphConvolution(args.hidden, args.hidden,num_features_nonzero=0,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False),

                                    GraphConvolution(args.hidden, output_dim,num_features_nonzero=0,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False),

                                    )
    def forward(self, x):
        #x, support = inputs
        x=x.float()
        x = F.relu(self.bn1(self.fc1(x)))  
        #x = self.dropout(x)      
        x = F.relu(self.bn2((self.fc2(x))) )  
        #x = self.dropout(x)     
        x = F.relu(self.fc3(x))  
        adj_x = build_fcgcn_adjacency_matrix(x)
        supports = preprocess_adj(adj_x)
        #print(supports)
        i = torch.from_numpy(supports[0]).long().to(device)
        v = torch.from_numpy(supports[1]).to(device)
        support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)
        x = self.gcnlayers((x, support.to_dense()))  
        return x[0]

class GCNFC(nn.Module):
    def __init__(self, input_dim, hidden_dim1, output_dim):
        super(GCNFC, self).__init__()
        self.fc1 = nn.Linear(args.hidden, hidden_dim1)
        self.bn1=nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim1*2)
        self.bn2=nn.BatchNorm1d(hidden_dim1*2)
        self.fc3 = nn.Linear(hidden_dim1*2,output_dim) 
        self.dropout = nn.Dropout(args.dropout)  
        self.gcnlayers = nn.Sequential(GraphConvolution(args.hidden, args.hidden,num_features_nonzero=0,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False),

                                    #GraphConvolution(args.hidden, args.hidden,num_features_nonzero=0,
                                                     #activation=F.relu,
                                                     #dropout=args.dropout,
                                                     #is_sparse_inputs=False),

                                    )
    def forward(self, x):
        #x, support = inputs
        x=x.float()
        adj_x = build_fcgcn_adjacency_matrix(x)
        supports = preprocess_adj(adj_x)
        #print(supports)
        i = torch.from_numpy(supports[0]).long().to(device)
        v = torch.from_numpy(supports[1]).to(device)
        support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)
        x = self.gcnlayers((x, support.to_dense())) 
        x=x[0]
        x = F.relu(self.bn1(self.fc1(x)))  
        x = self.dropout(x)      
        x = F.relu(self.bn2((self.fc2(x))) )  
        x = self.dropout(x)     
        x = F.tanh(self.fc3(x)) 
        return x



def generate_data(num_samples, input_dim, num_classes):
    X_train = np.random.randint(2, size=(num_samples, input_dim))
    y_train = np.random.randint(num_classes, size=num_samples)
    return X_train, y_train

def build_adjacency_matrix(X):
    num_nodes = args.hidden
    adj_matrices = []
    adj = np.eye(num_nodes)  
    for i in range(num_nodes):
        for j in range(num_nodes):
            
            adj[i,j] = np.dot(X[:, i].cpu().numpy(), X[:, j].cpu().numpy()) / (np.linalg.norm(X[:, i].cpu().numpy()) * np.linalg.norm(X[:, j].cpu().numpy())+1e-10)

    return np.array(adj)
def build_fcgcn_adjacency_matrix(X):
    num_nodes = args.hidden
    adj_matrices = []
    adj = np.eye(num_nodes)  
    for i in range(num_nodes):
        for j in range(num_nodes):
            
            adj[i,j] = (np.dot(X[:, i].cpu().detach().numpy(), X[:, j].cpu().detach().numpy())) / (np.linalg.norm(X[:, i].cpu().detach().numpy()) * np.linalg.norm(X[:, j].cpu().detach().numpy())+1e-10)

    return np.array(adj)

if __name__ == "__main__": 
    args = argparse.ArgumentParser()
    args.add_argument('--model', default='FC')
    args.add_argument('--learning_rate', type=float, default=0.0003)
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--hidden', type=int, default=322)
    args.add_argument('--dropout', type=float, default=0.3)
    
    args = args.parse_args()
    print(args)
    batch_size=256
    structure=args.model
    dataset = CustomDataset('./GCN_Berlin_Data.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Split dataset into train and test
    train_data, test_data = train_test_split(dataset, test_size=0.4, random_state=42)

    # Create dataloaders for train and test sets
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    train_loader_length = len(train_dataloader)
    test_loader_length = len(test_dataloader)

    #num_samples = 3000
    input_dim = args.hidden
    num_classes = 28
    num_features_nonzero = 0
    if structure=='GCN':
        net = GCN(input_dim, num_classes, num_features_nonzero)
    elif structure=='FC':
        net = SimpleFullyConnected(input_dim,2048,num_classes)
    elif structure=='FCGCN':
        net = FCGCN(input_dim,1024, num_classes)  
    elif structure=='GCNFC':
        net = GCNFC(input_dim,1024, num_classes)  
    net.to(device)
    best_acc=0
    for epoch in range(args.epochs):
        net.train()
        #net.to(device)
        train_dataloader_iter=iter(train_dataloader)
        for train_i in range(train_loader_length):
            #print(train_i)
            batch = train_dataloader_iter.next()
            X_train,y_train=batch
            X_train=X_train.to(device)
            y_train=y_train.to(device)
            #X_train, y_train = generate_data(num_samples, input_dim, num_classes)
        
            #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
            
            optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
            optimizer.zero_grad()
            if structure=='GCN':
                adj_train = build_adjacency_matrix(X_train)
                supports = preprocess_adj(adj_train)
                i = torch.from_numpy(supports[0]).long().to(device)
                v = torch.from_numpy(supports[1]).to(device)
                support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)
                output = net((X_train, support.to_dense()))
            elif structure=='FC':
                output = net(X_train)
            elif structure=='FCGCN':
                output = net(X_train)
            elif structure=='GCNFC':
                output = net(X_train)
            #print(output)
            #loss = F.nll_loss(output[0], y_train)
            cir=torch.nn.CrossEntropyLoss()
            loss=cir(output, y_train)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item()}')

            test_dataloader_iter=iter(test_dataloader)
            net.eval()
            accuracy=[]
            with torch.no_grad():
                for test_i in range(test_loader_length):   
                    batch = test_dataloader_iter.next()
                    X_test,y_test=batch
                    X_test = X_test.to(device)
                    y_test = y_test.to(device)
                    if structure=='GCN':
                        adj_test = build_adjacency_matrix(X_test)
                        supports = preprocess_adj(adj_test)
                        i = torch.from_numpy(supports[0]).long().to(device)
                        v = torch.from_numpy(supports[1]).to(device)
                        support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)
                        output = net((X_test, support.to_dense()))
                    elif structure=='FC':
                        output = net(X_test)
                    elif structure=='FCGCN':
                        output = net(X_test)
                    elif structure=='GCNFC':
                        output = net(X_test)
                    predicted = output.argmax(dim=1)
                    
                    accuracy.append( (predicted == y_test).sum().item() / len(y_test))
            print(f'Test Accuracy: {np.mean(accuracy)}')
            if np.mean(accuracy)>best_acc:
                best_acc=np.mean(accuracy)
                torch.save(net.state_dict(), './best.pth')
                
            print(f'Best Accuracy: {np.mean(best_acc)}')
        
