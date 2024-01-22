#%%
import torch
import torch.nn as nn
import torch.quantization as quantization
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from torchviz import make_dot
from torchinfo import summary
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import copy

from scipy.stats import norm
from scipy.stats import rankdata
import time
import decimal
import ctypes
from ctypes import c_void_p, c_int, cdll, POINTER
#%%
from myutils import eval_loss, fit, evaluate_history, show_images_labels, torch_seed
#%%
# GPU 디바이스 할당

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#%%
# 분류 클래스 명칭 리스트
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 분류 클래스 수,　10
n_output = len(list(set(classes)))


#%%

"""## 데이터 준비"""

# Transforms의 정의

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(0.5, 0.5)
])

# 데이터 취득용 함수 dataset

data_root = './data'

train_set = datasets.CIFAR10(
    root = data_root, train = True,
    download = True, transform = transform)

# 검증 데이터셋
test_set = datasets.CIFAR10(
    root = data_root, train = False,
    download = True, transform = transform)

# 미니 배치 사이즈 지정
batch_size = 100

# 훈련용 데이터로더
# 훈련용이므로 셔플을 True로 설정
train_loader = DataLoader(train_set,
    batch_size = batch_size, shuffle = True)

# 검증용 데이터로더
# 검증용이므로 셔플하지 않음
test_loader = DataLoader(test_set,
    batch_size = batch_size, shuffle = False)

# 처음 50개 이미지 출력
show_images_labels(test_loader, classes, None, None)
#%%


class CNN_v4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1,1),bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1,1),bias=False)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1,1),bias=False)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1,1),bias=False)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=(1,1),bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1,1),bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d((2,2))
        self.l1 = nn.Linear(4*4*128, 128)
        self.l2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
            self.relu,
            self.maxpool,
            self.dropout1,
            self.conv3,
            self.bn3,
            self.relu,
            self.conv4,
            self.bn4,
            self.relu,
            self.maxpool,
            self.dropout2,
            self.conv5,
            self.bn5,
            self.relu,
            self.conv6,
            self.bn6,
            self.relu,
            self.maxpool,
            self.dropout3,
            )

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.dropout3,
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3
#%%
# 난수 고정
torch_seed()

# 모델 인스턴스 생성
net = CNN_v4(n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
history = np.zeros((0, 5))
#%%
# 학습

num_epochs = 30
history = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)
#%%
evaluate_history(history)
#%%

model = net  # Your trained model
torch.save(model, 'model_trained.pt')

#%%

summary(net,(100,3,32,32))
# %%
for images, labels in test_loader:
    break


images.shape
# %%
#%%print(model)

# %%
model_load=torch.load( 'model_trained.pt')
F_layer=nn.Flatten()

model_load.eval()
train_loss = 0
train_acc = 0
val_loss = 0
val_acc = 0
count = 0

for inputs, labels in test_loader:
    count += len(labels)
    inputs = inputs.to(device)
    labels = labels.to(device)

    # 예측 계산
            
    x=model_load.conv1(inputs)
    x=model_load.bn1(x)
    x=model_load.relu(x)

    x2=x.clone()
    x=model_load.conv2(x)
    y2=x.clone()
    x=model_load.bn2(x)
    x=model_load.relu(x)

    x=model_load.maxpool(x)
    
    x=model_load.conv3(x)
    x=model_load.bn3(x)
    x=model_load.relu(x)

    x=model_load.conv4(x)
    x=model_load.bn4(x)
    x=model_load.relu(x)

    x=model_load.maxpool(x)
    
    x6=x.clone()
    x=model_load.conv5(x)
    y6=x.clone()

    x=model_load.bn5(x)
    x=model_load.relu(x)

    
    x=model_load.conv6(x)
    '''
    #####################################
    o_6=[]
    fw6=F_layer(model_load.conv6.weight)
    np_fw6=fw6.cpu().detach().numpy()
    
    i6 = F.pad(x6, (1, 1, 1, 1), "constant", 0)
    np_i6=i6.cpu().detach().numpy()
    for i_ch_n in range (np_i6.shape[0]):
        print(i_ch_n)
        o_6a=[]
        for o_ch_n in range (np_fw6.shape[0]):
            for y_n in range(x6.shape[3]):
                for x_n in range(x6.shape[3]):
                    out=np.sum(np_i6[i_ch_n][:, y_n:y_n+3, x_n:x_n+3].reshape(-1)*np_fw6[o_ch_n])
                    o_6a=np.append(o_6a,out)
        o_6=np.append(o_6,o_6a)
    
    x=torch.tensor(o_6.reshape((100,128,8,8)),dtype=torch.float32).to(device)
    #####################################
    '''
    
    x=model_load.bn6(x)
    x=model_load.relu(x)

    x=model_load.maxpool(x)
    
    x=model_load.flatten(x)

    x=model_load.l1(x)
    x=model_load.relu(x)
    
    outputs=model_load.l2(x)





    # 손실 계산
    loss = criterion(outputs, labels)
    val_loss += loss.item()

    # 예측 라벨 산출
    predicted = torch.max(outputs, 1)[1]

    # 정답 건수 산출
    val_acc += (predicted == labels).sum().item()

    # 손실과 정확도 계산
    avg_val_loss = val_loss / count
    avg_val_acc = val_acc / count
    break
    

print(avg_val_acc)

# %%
x6=x6*0+0.1

# %%
x6[0][0].shape
# %%
F_layer=nn.Flatten()
model_load.conv5.weight=torch.nn.Parameter (model_load.conv5.weight*0+0.1)
fw6=F_layer(model_load.conv5.weight)
np_fw6=fw6.cpu().detach().numpy()

# %%

i6 = F.pad(x6, (1, 1, 1, 1), "constant", 0)
i6[0].shape
np_i6=i6.cpu().detach().numpy()
np_i6
# %%
np_i6[0][:, 0:3, 0:3].reshape(-1)*np_fw6[0]
#%%
np.sum(np_i6[0][:, 1:4, 1:4].reshape(-1)*np_fw6[0]).dtype
# %%
F_layer(i6[0][:, :3, :3]).shape
# %%
#model_load.conv6.weight=torch.nn.Parameter (model_load.conv6.weight*0+1)
yy6=model_load.conv5(x6)
yy6

# %%
o_6=[]
for i_ch_n in range (np_i6.shape[0]):
    print(i_ch_n)
    o_6a=[]
    for o_ch_n in range (np_fw6.shape[0]):
        for y_n in range(x6.shape[3]):
            for x_n in range(x6.shape[3]):
                out=np.sum(np_i6[i_ch_n][:, y_n:y_n+3, x_n:x_n+3].reshape(-1)*np_fw6[o_ch_n])
                o_6a=np.append(o_6a,out)
    o_6=np.append(o_6,o_6a)
o_6
# %%
o_6
# %%
np.mean(np.abs(1-yy6.cpu().detach().numpy()/o_6.reshape((100,128,8,8))))
# %%
np.sum(np_i6[0][:, 0:0+3, 0:0+3].reshape(-1)*np_fw6[0])

# %%
yy6[0][0]
# %%
n=10000
conv_test = nn.Conv2d(n, 128, 3, padding=(1,1),bias=False,dtype=torch.float64)
conv_test = nn.Conv2d(n, 128, 3, padding=(1,1),bias=False)
conv_test.weight=torch.nn.Parameter (conv_test.weight*0+0.1)
conv_test.to(torch.device("cpu"))
conv_test.to(torch.device("cuda:0"))
conv_test.weight.dtype

dummy=torch.randn(10,n,8,8).to(torch.float64)
dummy=torch.randn(10,n,8,8)
dummy=dummy*0+0.1
dummy=dummy.to(torch.device("cpu"))
dummy=dummy.to(torch.device("cuda:0"))
dummy.dtype


conv_test(dummy)
# %%
with torch.no_grad():
    aa=conv_test(dummy)
aa
# %%
x6.shape
# %%
dummy=torch.randn(10,64,8,8)
# %%
dummy.dtype
# %%
conv_test.weight.dtype
# %%
