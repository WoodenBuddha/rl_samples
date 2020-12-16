import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self):
        print('Model: {}'.format(self.__class__.__name__))
        super(MLP, self).__init__()


class DenseNet(MLP):
    def __init__(self, input, hidden, outputs, weights_init='xavier'):
        super(DenseNet, self).__init__()

        self.dense1 = nn.Linear(input, hidden)
        self.dense2 = nn.Linear(hidden, outputs)

        init.xavier_uniform_(self.dense1.weight)
        init.xavier_uniform_(self.dense2.weight)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)

        x = self.dense2(x)
        x = F.relu(x)

        return x
