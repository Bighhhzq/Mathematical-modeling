import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as Fun


# defeine BP neural network
class Net1(torch.nn.Module):
    def __init__(self, n_feature, n_output=2):
        super(Net1, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 50)
        self.out = torch.nn.Linear(50, n_output)

    def forward(self, x):
        x = Fun.relu(self.hidden(x))
        x = self.out(x)
        return x


class Net2(torch.nn.Module):
    def __init__(self, n_feature=729, n_output=2):
        super(Net2, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, 1000)
        self.hidden2 = torch.nn.Linear(1000, 200)
        self.out = torch.nn.Linear(200, n_output)

    def forward(self, x):
        x = Fun.relu(self.hidden1(x))
        x = Fun.relu(self.hidden2(x))
        x = self.out(x)
        return x
def printreport(exp, pred):
    print(classification_report(exp, pred))
    print("recall score")
    print(recall_score(exp, pred, average='macro'))


gr = pd.read_csv('./clean451.csv', index_col=0, encoding='gb18030')

feature = ['ATSm2', 'ATSm3', 'BCUTc-1h', 'SCH-6', 'VC-5', 'SP-1', 'ECCEN', 'SHBd',
       'SsCH3', 'SaaO', 'minHBa', 'minaaO', 'maxaaO', 'hmin',
       'LipoaffinityIndex', 'ETA_Beta', 'ETA_Beta_s', 'ETA_Eta_R', 'ETA_Eta_F',
       'ETA_Eta_R_L', 'FMF', 'MDEC-12', 'MDEC-23', 'MLFER_S', 'MLFER_E',
       'MLFER_L', 'TopoPSA', 'MW', 'WTPT-1', 'WPATH']

feature_df = gr[feature]

x = feature_df.values


print(x)
y_var = ['Caco-2', 'CYP3A4', 'hERG', 'hERG', 'MN']

for v in y_var:
    y = gr[v]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.7)
    print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    scaler = StandardScaler()
    x = scaler.fit_transform(x_train)

    input = torch.FloatTensor(x)
    label = torch.LongTensor(y_train)

    net = Net1(n_feature=30, n_output=2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    # SGD: random gradient decend
    loss_func = torch.nn.CrossEntropyLoss()
    # define loss function

    for i in range(100):
        out = net(input)

        loss = loss_func(out, label)
        optimizer.zero_grad()
        # initialize
        loss.backward()
        optimizer.step()

    x = scaler.fit_transform(x_test)

    input = torch.FloatTensor(x)
    label = torch.Tensor(y_test.to_numpy())

    out = net(input)

    prediction = torch.max(out, 1)[1]
    pred_y = prediction.numpy()
    target_y = label.data.numpy()

    s = accuracy_score(target_y, pred_y)
    print('accury')
    print(s)

    cm = confusion_matrix(target_y, pred_y)
    printreport(target_y, pred_y)

    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.show()
