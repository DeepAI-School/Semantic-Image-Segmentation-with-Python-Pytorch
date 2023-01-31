#This code uses the DeepLabV3 decoder and resnet101 encoder from torchvision library to perform semantic segmentation on an input image. 
#The models have been trained on COCO dataset with total of 21 classes including background.
#Model trained on the following Classes: background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse,
#motorbike, person, potted plant, sheep, sofa, train, and tv/monitor. 
#You can predict and segment images only belongs to these classes using pretrained model. 
#If you want to train the segmentation models on your own image dataset to segment the images and to produce high results 
#then follow the course Deep Learning for Semantic Segmentation with Python & Pytorch, link is given in the readme.

#The code first mounts the Google Colab drive to access the image file in Colab. 
#Then, the pre-trained DeepLabV3 model is loaded using models.segmentation.deeplabv3_resnet101 from torchvision library.

from google.colab import drive
drive.mount('/content/drive')

#Load the pretrained Model
from torchvision import models
SegModel=models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

#Next, the image is loaded using the Image module from PIL library and displayed using matplotlib. 
#The image is then transformed using the Compose method from torchvision.transforms library. 
#The transformed image is passed through the DeepLabV3 model, which returns the segmented image in the form of a tensor.
#The segmented image is then converted to a numpy array and the unique segments are found using np.unique method.

from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

#Load Input image from google drive in Colab
img = Image.open('/content/drive/My Drive/Colab Notebooks/car.jpg')
plt.imshow(img); plt.show()


#Define Transformations
import torchvision.transforms as T
trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
inp = trf(img).unsqueeze(0)

#Predict the output
out = SegModel(inp)['out']

predicted = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print (np.unique(predicted))

#The decode_segmap function is defined to convert the segmented image to an RGB image. The final segmented RGB image is then displayed using matplotlib.

def decode_segmap(image, nc=21):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb

rgb = decode_segmap(predicted)
plt.imshow(rgb); plt.show()

#If you want to train the segmentation models on your own image dataset to segment the images and to produce high results 
#then follow the course Deep Learning for Semantic Segmentation with Python & Pytorch, link is given in the readme..
