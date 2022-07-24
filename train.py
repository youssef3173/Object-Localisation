import time
import os

import cv2
import tqdm  

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import torch.optim as optim


#############################################################################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#############################################################################################################################

imgSize = 256
data_transforms = { 
    'TRAIN': transforms.Compose([          # 2886 imgs
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'IMGs/'
image_datasets = { 'TRAIN': datasets.ImageFolder(os.path.join(data_dir, 'TRAIN'), data_transforms['TRAIN']) }
dataloaders = { 'TRAIN': torch.utils.data.DataLoader(image_datasets['TRAIN'], batch_size=13, shuffle=True, num_workers=0) }
dataset_sizes = { 'TRAIN': len(image_datasets['TRAIN']) }
classes = image_datasets['TRAIN'].classes



#############################################################################################################################


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)        # convolution layer : input is 1 image (3 channels: RGB), 5x5 kernel / window 
        self.pool = nn.MaxPool2d(2, 2)         # 2*2 pxl --max--> 1 pxl
        self.conv2 = nn.Conv2d(6, 16, 5)       # Convolution
        
        #self.classifier = nn.Sequential(     # Or fully connected
        self.fc1 =  nn.Linear(16 * 50 * 50, 1024)
        self.fc2 =  nn.Linear(1024, 256)
        self.fc3 =  nn.Linear(256, 64)
        self.fc4 =  nn.Linear(64, 2)
        # )
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # F.relu or nn.Sigmoid : activation function 
                                               # IMG = 3*52*52 --conv1--> IMG = 6*48*48 --pool-->  IMG = 6*24*24                                     
        x = self.pool(F.relu(self.conv2(x)))   # IMG = 6*24*24 --conv2--> IMG = 16*20*20 --pool-->  IMG = 16*10*10
        x = self.pool(x)                       # IMG = 16*10*10 --pool-->  IMG = 16*5*5
        
        x = x.view(-1, 16 * 50 * 50)           # flattind the tensor   
        
        x = F.relu(self.fc1(x) )
        x = F.relu(self.fc2(x) )
        x = F.relu(self.fc3(x) )
        x = self.fc4(x) 
        return x

model = CNN().to(device)
print(model)



#############################################################################################################################


PATH = 'checkpoints/FullNet.pth'
model.load_state_dict(torch.load(PATH))
model.eval()


#############################################################################################################################

epochs = 100           
learning_rate = 0.00002

loss_values = np.zeros((10, 5))

Loss = nn.CrossEntropyLoss()       
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)    

n_total_steps = len(dataloaders['TRAIN'])


#############################################################################################################################

start = time.time()   

for epoch in range(epochs):
    for i, data in enumerate(dataloaders['TRAIN'], 0):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()        
        outputs = model(images)       
        loss = Loss(outputs, labels)   

        loss.backward()              
        optimizer.step()              

        # loss_value[epoch] = loss.item()
        if((i+1) % 40 == 0):    
            print (f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            loss_values[epoch][(i+1)//40 - 1] = loss.item()

end = time.time()

print("\n Finished Training in : {:>20f} s".format(end - start))


PATH = 'checkpoints/FullNet.pth'     
torch.save(model.state_dict(), PATH)



print('Labels in a batch :\n \t', labels),
print('check size and data type of imgs : '),
print('\t', images.size())
print('\t', images.dtype, '\n')


loss_flatten = [i for sublist in loss_values for i  in sublist]

ax = plt.subplot(111)
ax.plot( loss_flatten, c = 'b')  

ax.legend()
plt.show()




#############################################################################################################################



data_transforms = {
    'TEST': transforms.Compose([                   # 720 imgs 
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'IMGs/'
image_datasets = { 'TEST': datasets.ImageFolder(os.path.join(data_dir, 'TEST'), data_transforms['TEST']) }
dataloaders = { 'TEST': torch.utils.data.DataLoader(image_datasets['TEST'], batch_size=30, shuffle=False, num_workers=1) }
dataset_sizes = { 'TEST': len(image_datasets['TEST']) }

batch_size = 30


#############################################################################################################################


start = time.time()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(2)]     
    n_class_samples = [0 for i in range(2)]
    for images, labels in dataloaders['TEST']:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        value, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network : {acc} %')

    for j in range(2):
        acc = 100.0 * n_class_correct[j] / n_class_samples[j]
        print(f'Accuracy of {classes[j]} \t : {acc} %')
               
end = time.time()
print("Finished Testing in : {:f} s".format(end - start))


#############################################################################################################################

# Accuracy of the network : 91.94444444444444 %
# Accuracy of A 	 : 94.78260869565217 %
# Accuracy of P 	 : 86.92307692307692 %
# Finished Testing in : 430.700802 s

#############################################################################################################################

