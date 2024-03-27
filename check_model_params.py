import torch
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的设备:", device)
model = CNN2()
model.load_state_dict(torch.load("model/20240327-19-cnn_bn/cnn.pth"))
for name, param in model.named_parameters():  
    print(name, param.size())