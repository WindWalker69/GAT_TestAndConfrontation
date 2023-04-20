import math
import torch
import numpy as np
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from torch.nn import Sequential, Linear, ReLU, LeakyReLU
from torch_geometric.nn import GINConv, GATConv, GATv2Conv, BatchNorm, global_add_pool, global_mean_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GATEncoder(torch.nn.Module):
    def __init__(self, num_features, num_classes, hid_dim, num_gat_layers, heads, dropout):
        super(GATEncoder, self).__init__()

        self.num_gat_layers = num_gat_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()  # List of convolutions
        self.bns = torch.nn.ModuleList()  # List of batch-normalizations

        # ==================== GATConv ==========================

        self.convs.append(GATConv(in_channels=num_features,
                                  out_channels=hid_dim,
                                  heads=heads,
                                  dropout=dropout))
        self.bns.append(torch.nn.BatchNorm1d(hid_dim * heads))
        for _ in range(num_gat_layers - 2):
            self.convs.append(GATConv(in_channels=hid_dim * heads,
                                      out_channels=hid_dim,
                                      heads=heads,
                                      dropout=dropout))
            self.bns.append(torch.nn.BatchNorm1d(hid_dim * heads))
        self.convs.append(GATConv(in_channels=hid_dim * heads,
                                  out_channels=num_classes,
                                  concat=False,
                                  dropout=dropout))
        self.bns.append(torch.nn.BatchNorm1d(num_classes))

        # ==================== GATv2Conv ==========================

        # self.convs.append(GATv2Conv(in_channels=num_features,
        #                             out_channels=hid_dim,
        #                             heads=heads,
        #                             dropout=0.5))
        # self.bns.append(BatchNorm(hid_dim * heads))
        # for _ in range(num_gat_layers - 2):
        #     self.convs.append(GATv2Conv(in_channels=hid_dim * heads,
        #                                 out_channels=hid_dim,
        #                                 heads=heads,
        #                                 dropout=0.5))
        #     self.bns.append(BatchNorm(hid_dim * heads))
        # self.convs.append(GATv2Conv(in_channels=hid_dim * heads,
        #                             out_channels=num_classes,
        #                             concat=False,
        #                             dropout=0.5))
        # self.bns.append(BatchNorm(num_classes))

        # for i in range(num_gat_layers):
        #     if i:
        #         if i == num_gat_layers - 1:
        #             conv = GATConv(in_channels=hid_dim * heads,
        #                            out_channels=num_classes,
        #                            heads=heads,
        #                            concat=False,
        #                            dropout=0.5)
        #         else:
        #             conv = GATConv(in_channels=hid_dim * heads,
        #                            out_channels=hid_dim,
        #                            heads=heads,
        #                            dropout=0.5)
        #     else:
        #         conv = GATConv(in_channels=num_features,
        #                        out_channels=hid_dim,
        #                        heads=heads,
        #                        dropout=0.5)
        #
        #     # bn = torch.nn.BatchNorm1d(hid_dim)
        #
        #     self.convs.append(conv)
        #     # self.bns.append(bn)

        # self.convs.append(Linear(in_features=hid_dim * heads, out_features=num_classes))

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gat_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        x_pool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(x_pool, 1)
        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):

        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class GINEncoder(torch.nn.Module):
    def __init__(self, num_features, hid_dim, num_gin_layers):
        super(GINEncoder, self).__init__()

        self.num_gin_layers = num_gin_layers

        self.convs = torch.nn.ModuleList()  # List of convolutions
        self.bns = torch.nn.ModuleList()  # List of batch-normalizations

        # self.convs.append(GINConv(Sequential(Linear(num_features, hid_dim), ReLU(), Linear(hid_dim, hid_dim))))
        # self.bns.append(torch.nn.BatchNorm1d(hid_dim))
        # for _ in range(num_gat_layers - 2):
        #     self.convs.append(GINConv(Sequential(Linear(hid_dim, hid_dim), ReLU(), Linear(hid_dim, hid_dim))))
        #     self.bns.append(torch.nn.BatchNorm1d(hid_dim))
        # self.convs.append(GINConv(Sequential(Linear(hid_dim, hid_dim), ReLU(), Linear(hid_dim, num_classes))))
        # self.bns.append(torch.nn.BatchNorm1d(num_classes))

        for i in range(num_gin_layers):

            if i:
                nn = Sequential(Linear(hid_dim, hid_dim), ReLU(), Linear(hid_dim, hid_dim))
            else:
                nn = Sequential(Linear(num_features, hid_dim), ReLU(), Linear(hid_dim, hid_dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(hid_dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gin_layers):
            x = F.elu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        x_pool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(x_pool, 1)
        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):

        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class LogReg(torch.nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def logistic_classify(x, y):
    y = preprocessing.LabelEncoder().fit_transform(y)
    x, y = np.array(x), np.array(y)

    nb_classes = np.unique(y).shape[0]
    xent = torch.nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls = y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls = torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()

        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())
    return np.mean(accs)


def svc_classify(x, y, search=True):
    y = preprocessing.LabelEncoder().fit_transform(y)
    x, y = np.array(x), np.array(y)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(balanced_accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)


class FF(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = Sequential(
            Linear(input_dim, input_dim),
            ReLU(),
            Linear(input_dim, input_dim),
            ReLU(),
            Linear(input_dim, input_dim),
            ReLU()
        )
        self.linear_shortcut = Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


def local_global_loss_(l_enc, g_enc, batch, measure):
    num_graphs = g_enc.shape[0]  # numero di grafi nei vari batch
    num_nodes = l_enc.shape[0]  # numero di nodi totali di tutti i grafi nei vari batch

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    for node_idx, graph_idx in enumerate(batch):
        pos_mask[node_idx][graph_idx] = 1.
        neg_mask[node_idx][graph_idx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2

    if average:
        return Eq.mean()
    else:
        return Eq
