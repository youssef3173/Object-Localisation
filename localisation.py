 
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize
import torch.nn.functional as F
import torch.optim as optim



from PIL import Image
from torch.autograd import Variable
from torch import topk
import skimage.transform
from matplotlib.patches import Circle


from train import CNN



########################################################################################################
# Load Model:
########################################################################################################


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device : ', device)


image = Image.open('IMGs/TEST/P/84.jpg')



normalize = transforms.Normalize(
   mean=[0.5, 0.5, 0.5],
   std=[0.25, 0.25, 0.25]
)

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

display_transform = transforms.Compose([
        transforms.Resize((224,224))])

tensor = preprocess(image)
prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)

# Using ResNET18 :
model = CNN().to(device)

PATH = 'checkpoints/FullNet.pth'
model.load_state_dict(torch.load(PATH))
model.cuda()
model.eval()


########################################################################################################
# Class Activation Map (CAM):
########################################################################################################

class SaveFeatures():
    features = None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

final_layer = model._modules.get('layer4')
activated_features = SaveFeatures(final_layer)

prediction = model(prediction_var) 
pred_probabilities = F.softmax(prediction).data.squeeze() 
activated_features.remove()

topk(pred_probabilities, 1)
def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

weight_softmax_params = list(model._modules.get('fc').parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
weight_softmax_params
class_idx = topk(pred_probabilities, 1)[1].int() 
overlay = getCAM(activated_features.features, weight_softmax, class_idx )



t = torch.tensor([ [[1.2,2.3],[3.4,4.5],[5.6,6.7]],[[1.2,2.3],[3.4,4.5],[5.6,6.7]] ])
u = torch.tensor( [[1.2, 2.3, 3.4, 4.5, 5.6, 6.7]] )
# a =  t.unsqueeze(0)
# print(a)
# pred = Variable( a, requires_grad=True)
# print(pred)
print(activated_features)
topk( u, 1)


plt.figure(figsize=(20,20))
plt.subplot(141), plt.imshow(image.resize((224, 224)))
plt.subplot(142), plt.imshow(display_transform(image)), plt.imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.7, cmap='plasma')
heat_map = skimage.transform.resize(overlay[0], tensor.shape[1:3])
plt.subplot(143), plt.imshow(heat_map)

# Position :
arr = np.array(heat_map)
max = np.where(arr == np.amax(arr))

x, y = max[0][0], max[1][0]
patch = [ Circle((y, x), radius=7, color='lime') ]
plt.subplot(144), plt.imshow(image.resize((224, 224))), plt.gca().add_patch(patch[0])




########################################################################################################
# Gradient basd method:
########################################################################################################

img = read_image('IMGs/TEST/P/1197.jpg')
 
tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(device)
tensor.requires_grad = True

out = model(tensor)
val = out.argmax(dim = 1)

gradients = torch.autograd.grad(outputs = out.squeeze(0)[val], inputs=tensor, retain_graph=True)[0]
heatmap = gradients[0][0] + gradients[0][1] + gradients[0][2]
print(heatmap.shape)

plt.figure(figsize = (20, 20))
plt.subplot(131) , plt.imshow( resize( img, (224, 224) ).permute(1,2,0) )
plt.subplot(132) , plt.imshow( skimage.transform.resize( heatmap.cpu() , (24, 24)) , alpha=0.9, cmap = 'RdYlBu')

arr = np.array(skimage.transform.resize(heatmap.cpu(), (224, 224)))
max = np.where(arr == np.amax(arr))

x, y = max[0][0], max[1][0]
patch = [ Circle((y, x), radius=7, color='yellow') ]
plt.subplot(133), plt.imshow( skimage.transform.resize( img.permute(1, 2, 0), (224, 224)) ), plt.gca().add_patch(patch[0])

plt.show()



"""
##################################################################################################################
# CAM: (another method)
##################################################################################################################


# img = read_image('TEST/P/5844.jpg')

# IMG to TENSOR :
tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(device)
out = model(tensor)

fm0 = model.maxpool( model.relu( model.bn1( model.conv1(tensor) ) ) )
fm4 = model.layer4( model.layer3( model.layer2( model.layer1( fm0 ) ) ) )

fmfc =  fm4.reshape(512,7*7)

fn = model.fc(fmfc.permute(1,0)).permute(1,0).cpu().detach().numpy() 
hm = fn.reshape(2,7,7)

plt.figure(figsize = (20, 20) )
plt.subplot(131), plt.imshow( skimage.transform.resize(img.permute(1,2,0), (224, 224)) )
plt.subplot(132), plt.imshow( skimage.transform.resize(hm[1], (224, 224)) , alpha=0.9, cmap = 'jet')

# plt.figure(figsize = (15, 15) )
# plt.subplot(121), plt.imshow( skimage.transform.resize( img.permute(1, 2, 0), (224, 224)) ), 
# plt.imshow( skimage.transform.resize(hm[1], (224, 224)) , alpha=0.5, cmap='jet')

from skimage import io
from matplotlib.patches import Circle

arr = np.array(skimage.transform.resize(hm[1], (224, 224)))
max = np.where(arr == np.amax(arr))

x, y = max[0][0], max[1][0]
patch = [ Circle((y, x), radius=7, color='yellow') ]
plt.subplot(133), plt.imshow( skimage.transform.resize( img.permute(1, 2, 0), (224, 224)) ), plt.gca().add_patch(patch[0])

"""




