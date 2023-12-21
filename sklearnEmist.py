import torch  
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
train_data = pd.read_csv('C:/Users/steph/OneDrive/桌面/code/ml/ocr/ubyteFile/emnist-letters-train.csv')
#print(train_data.shape) (63999, 785)
#print(test_data.shape)  (35999, 785)

x_train_data = train_data.values[0:,1:]
y_train_label = train_data.values[0:,0]

X_train, X_test, y_train, y_test = train_test_split(x_train_data, y_train_label,test_size = 0.2)

BATCH_SIZE = 32

torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor) # data type is long

torch_X_train = torch_X_train.view(-1, 1,28,28).float()
torch_X_test = torch_X_test.view(-1,1,28,28).float()
# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

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
cnn = CNN()

def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = 5
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            #print(correct)
            if batch_idx % 50 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.item(), float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))

fit(cnn,train_loader)
def evaluate(model):
#model = CNN
    correct = 0 
    for test_imgs, test_labels in test_loader:
        #print(test_imgs.shape)
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*BATCH_SIZE)))
evaluate(cnn)
torch.save(cnn, 'pytorch_emnist_cnn')
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
print('predict_result :', predict[1])
print('y_label        :',y_batch)
#pickle.dump(cnn, open('finalized_model.sav', 'wb'))#save