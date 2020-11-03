import torch.nn as nn
import torch.nn.functional as F



class FLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(79, 256),
                                 nn.Dropout(),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.Dropout(),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.Dropout(),
                                 nn.ReLU(),
                                 # nn.Linear(256, 256),
                                 # nn.Dropout(),
                                 # nn.ReLU(),
                                 nn.Linear(256, 14))  # 5 layers

    def forward(self, x):
        x = self.net(x)
        output = F.log_softmax(x, dim=1)  # use NLLLoss(), which accepts a log probability

        return output
