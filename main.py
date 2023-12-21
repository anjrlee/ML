# load the model from disk
import torch  
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(#28*28
                nn.Conv2d(in_channels =1, out_channels = 16, kernel_size = 5,
                          stride = 1, padding = 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2)#16張14*14
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5,
                          stride = 1, padding = 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2)#32張7*7
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5,
                          stride = 1, padding = 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2)
                )
        self.fc1 = nn.Linear(in_features = 64*3*3, out_features = 256)
        self.fc2 = nn.Linear(in_features = 256, out_features = 62)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)#flatten
        x = self.fc1(x)
        output = self.fc2(x)
        return output
cnn = torch.load('pytorch_emnist_cnn')
BATCH_SIZE = 32

test_data = pd.read_csv('C:/Users/steph/OneDrive/桌面/code/ml/ocr/ubyteFile/emnist-letters-test.csv')
x_test_data = test_data.values[0:,1:]
y_test_label = test_data.values[0:,0]

torch_X_test = torch.from_numpy(x_test_data).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test_label).type(torch.LongTensor) # data type is long

torch_X_test = torch_X_test.view(-1,1,28,28).float()
test = torch.utils.data.TensorDataset(torch_X_test, torch_y_test)

test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

it = iter(test_loader)
X_batch, y_batch = next(it)
test_out = cnn(X_batch)
predict = torch.max(test_out, 1)
