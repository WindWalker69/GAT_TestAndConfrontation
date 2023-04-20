import torch
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data

from InfoGraphUtilities import GATEncoder, GINEncoder, FF, local_global_loss_, logistic_classify, svc_classify

# Our Dataset
dataset = TUDataset(root='/tmp/TUDataset', name='MUTAG')
# dataset = TUDataset(root='/tmp/TUDataset', name='IMDB-BINARY')
# dataset = TUDataset(root='/tmp/TUDataset', name='IMDB-MULTI')
# dataset = TUDataset(root='/tmp/TUDataset', name='REDDIT-BINARY')
# dataset = TUDataset(root='/tmp/TUDataset', name='REDDIT-MULTI-5K')
# dataset = TUDataset(root='/tmp/TUDataset', name='ENZYMES')
# dataset = TUDataset(root='/tmp/TUDataset', name='PTC_MR')

print(f'{dataset.data}')

# Dataset Info
print('\n')
print(f'Dataset: {dataset}')
print('==========================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get first graph object of the dataset to analyze

# Data object info
print(f'\n{data}')
print('=================================================')

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirect: {data.is_undirected()}\n')


# def check_balance(dataset):
#     total_classes = dataset.num_classes
#     total_per_class = [0] * total_classes
#
#     for _, label in enumerate(dataset.data.y):
#         total_per_class[label] += 1
#
#     for label, class_sum in enumerate(total_per_class):
#         print(f'{class_sum} of class: {label} - {class_sum/sum(total_per_class) * 100:0.1f}%')
#     print('\n')
#
#
# check_balance(dataset)


# Split dataset into train and test set (after shuffle it)
torch.manual_seed(12345)
dataset = dataset.shuffle()

# train_dataset = dataset[:150]
# test_dataset = dataset[150:]
#
# print(f'\nNumber of training graphs: {len(train_dataset)}')
# print(f'Number of test graphs: {len(test_dataset)}\n')


# =========================================== GAT Version ====================================================
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_gat_layers, heads, dropout):
        super(GAT, self).__init__()

        self.embedding_dim = hidden_channels * heads * (num_gat_layers - 1) + dataset.num_classes  # hidden_channels * num_gat_layers
        self.encoder = GATEncoder(dataset_num_features,
                                  dataset.num_classes,
                                  hidden_channels,
                                  num_gat_layers,
                                  heads,
                                  dropout)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        measure = 'GAN'
        local_global_loss = local_global_loss_(l_enc, g_enc, batch, measure)

        return local_global_loss


# Settings
epochs = 100
batch_size = 128
lr = 0.01
weight_decay = 0  # 5e-4

# Setup utilities
dataset_num_features = max(dataset.num_features, 1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(hidden_channels=64, num_gat_layers=3, heads=8, dropout=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Mini-batching graphs of the dataset
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
data_loader = DataLoader(dataset, batch_size=batch_size)

for step, data in enumerate(data_loader):  # to see the mini-batching we makes
    print(f'Step: {step + 1}')
    print('=========')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(f'{data}\n')

print(model)

print('\n=============== Before Training ==================')
emb, y = model.encoder.get_embeddings(data_loader)
log_reg_acc = logistic_classify(emb, y)
svc_acc = svc_classify(emb, y)
print(f'LogReg Acc: {log_reg_acc:.4f}')
print(f'SVC Acc: {svc_acc:.4f}\n')


# Training function
def train():
    loss_all = 0

    model.train()
    for data in data_loader:  # Iterate in batches over the training dataset
        data = data.to(device)
        if data.x is None:
            data_list = []
            for i in range(len(data)):
                data_list.append(Data(x=torch.ones(data[i].num_nodes, dataset_num_features).to(device), edge_index=data[i].edge_index))
            batch = Batch.from_data_list(data_list)
            loss = model(batch.x, batch.edge_index, data.batch)  # Compute the loss

        else:
            loss = model(data.x, data.edge_index, data.batch)  # Compute the loss

        loss_all += loss.item() * data.num_graphs
        loss.backward()  # Derive gradients
        optimizer.step()  # Update parameters based on gradients
        optimizer.zero_grad()  # Clear gradients
    return loss_all


def test(loader):
    model.eval()

    emb, y = model.encoder.get_embeddings(loader)
    svc_acc = svc_classify(emb, y)
    log_reg_acc = logistic_classify(emb, y)
    return log_reg_acc, svc_acc


accuracies = []
for epoch in range(1, epochs + 1):
    tot_loss = train()
    # train_acc = test(train_loader)
    # test_acc = test(test_loader)
    log_reg_acc, svc_acc = test(data_loader)
    accuracies.append(log_reg_acc)
    print(f'Epoch: {epoch:03d}, '
          f'Loss: {tot_loss / len(data_loader):.3f}, '
          f'LogReg Acc: {log_reg_acc:.4f}, '
          f'SVC Acc: {svc_acc:.4f}')

    if epoch % epochs == 0:
        print(f'\nMax Acc = {max(accuracies):.4f}')
        print(f'Min Acc = {min(accuracies):.4f}')
        print(f'Mean Acc = {sum(accuracies) / len(accuracies):.4f}')


# =========================================== GIN Version ==========================================================
# class GIN(torch.nn.Module):
#     def __init__(self, hidden_channels, num_gin_layers):
#         super(GIN, self).__init__()
#
#         self.embedding_dim = hidden_channels * num_gin_layers
#         self.encoder = GINEncoder(dataset_num_features, hidden_channels, num_gin_layers)
#
#         self.local_d = FF(self.embedding_dim)
#         self.global_d = FF(self.embedding_dim)
#
#         self.init_emb()
#
#     def init_emb(self):
#         for m in self.modules():
#             if isinstance(m, Linear):
#                 torch.nn.init.xavier_uniform_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.fill_(0.0)
#
#     def forward(self, x, edge_index, batch):
#         if x is None:
#             x = torch.ones(batch.shape[0]).to(device)
#
#         y, M = self.encoder(x, edge_index, batch)
#
#         g_enc = self.global_d(y)
#         l_enc = self.local_d(M)
#
#         measure = 'JSD'
#         local_global_loss = local_global_loss_(l_enc, g_enc, batch, measure)
#
#         return local_global_loss
#
#
# # Settings
# epochs = 100
# batch_size = 128
# lr = 0.01
# weight_decay = 0
#
# # Setup utilities
# dataset_num_features = max(dataset.num_features, 1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GIN(hidden_channels=32, num_gin_layers=5).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#
# # Mini-batching graphs of the dataset
# data_loader = DataLoader(dataset, batch_size=batch_size)
#
# for step, data in enumerate(data_loader):  # to see the mini-batching we makes
#     print(f'Step: {step + 1}')
#     print('=========')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(f'{data}\n')
#
# print(model)
#
# print('\n=============== Before Training ==================')
# emb, y = model.encoder.get_embeddings(data_loader)
# log_reg_acc = logistic_classify(emb, y)
# svc_acc = svc_classify(emb, y)
# print(f'LogReg Acc: {log_reg_acc:.4f}')
# print(f'SVC Acc: {svc_acc:.4f}\n')
#
#
# # Training function
# def train():
#     loss_all = 0
#
#     model.train()
#     for data in data_loader:  # Iterate in batches over the training dataset
#         data = data.to(device)
#         loss = model(data.x, data.edge_index, data.batch)  # Compute the loss
#         loss_all += loss.item() * data.num_graphs
#         loss.backward()  # Derive gradients
#         optimizer.step()  # Update parameters based on gradients
#         optimizer.zero_grad()  # Clear gradients
#     return loss_all
#
#
# def test(loader):
#     model.eval()
#
#     emb, y = model.encoder.get_embeddings(loader)
#     svc_acc = svc_classify(emb, y)
#     log_reg_acc = logistic_classify(emb, y)
#     return log_reg_acc, svc_acc
#
#
# accuracies = []
# for epoch in range(1, epochs + 1):
#     tot_loss = train()
#     log_reg_acc, svc_acc = test(data_loader)
#     accuracies.append(log_reg_acc)
#     print(f'Epoch: {epoch:03d}, '
#           f'Loss: {tot_loss / len(data_loader):.3f}, '
#           f'LogReg Acc: {log_reg_acc:.4f}, '
#           f'SVC Acc: {svc_acc:.4f}')
#
#     if epoch % epochs == 0:
#         print(f'\nMax Acc = {max(accuracies):.4f}')
#         print(f'Min Acc = {min(accuracies):.4f}')
#         print(f'Mean Acc = {sum(accuracies) / len(accuracies):.4f}')
