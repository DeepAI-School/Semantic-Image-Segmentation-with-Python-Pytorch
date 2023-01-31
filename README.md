# Semantic Image Segmentation with Pytorch and Python 

Semantic segmentation is a computer vision task that involves classifying every pixel in an image into predefined classes or categories. For example, in an image with multiple objects, we want to know which pixel belongs to which object. 

![s1](https://user-images.githubusercontent.com/123977559/215550848-17e3893a-a783-47f8-8478-f2bc9609896a.png)

The goal of semantic segmentation is to assign a semantic label to each object in the image. This is a challenging task because it requires a high level of detail and accuracy, as well as the ability to handle variations in scale, orientation, and appearance.

![s2](https://user-images.githubusercontent.com/123977559/215550865-260b001d-3fb0-4e65-bcc5-34fd1970bf00.png)

Here is the course [Deep Learning for Image Segmentation with Python & Pytorch](https://www.udemy.com/course/deep-learning-for-semantic-segmentation-with-python-pytorh/?referralCode=0009C809CCE66FFAADA3) that provides a comprehensive, hands-on experience in applying Deep Learning techniques to Semantic Image Segmentation problems and applications. Segmentation has a wide range of potential applications in various fields.

![New Semantic Segmentation](https://user-images.githubusercontent.com/123977559/215546907-8c7ace81-ee20-432f-977f-bf0fffc8650f.png)

In [Deep Learning for Semantic Image Segmentation with Python & Pytorch](https://www.udemy.com/course/deep-learning-for-semantic-segmentation-with-python-pytorh/?referralCode=0009C809CCE66FFAADA3) course, you'll learn how to use the power of Deep Learning to segment images and extract meaning from visual data. You'll start with an introduction to the basics of Semantic Segmentation using Deep Learning, then move on to implementing and training your own models for Semantic Segmentation with Python and PyTorch.
This course is designed for a wide range of students and professionals, including but not limited to:
Machine Learning Engineers, Deep Learning Engineers, and Data Scientists who want to apply Deep Learning to Image Segmentation tasks

+ Computer Vision Engineers and Researchers who want to learn how to use PyTorch to build and train Deep Learning models for Semantic Segmentation
+ evelopers who want to incorporate Semantic Segmentation capabilities into their projects
+ Graduates and Researchers in Computer Science, Electrical Engineering, and other related fields who want to learn about the latest advances in Deep Learning for Semantic Segmentation
+ In general, the course is for anyone who wants to learn how to use Deep Learning to extract meaning from visual data and gain a deeper understanding of the theory and practical applications of Semantic Segmentation using Python and PyTorch

The course [Deep Learning for Semantic Segmentation with Python & Pytorch](https://www.udemy.com/course/deep-learning-for-semantic-segmentation-with-python-pytorh/?referralCode=0009C809CCE66FFAADA3) covers the complete pipeline with hands-on experience of Semantic Segmentation using Deep Learning with Python and PyTorch as follows:

+ Semantic Image Segmentation and its Real-World Applications in Self Driving Cars or Autonomous Vehicles etc.
+ Deep Learning Architectures for Semantic Segmentation including Pyramid Scene Parsing Network (PSPNet), UNet, UNet++, Pyramid Attention Network (PAN),  Multi-Task Contextual Network (MTCNet), DeepLabV3, etc.
+ Datasets and Data annotations Tool for Semantic Segmentation
+ Google Colab for Writing Python Code
+ Data Augmentation and Data Loading in PyTorch
+ Performance Metrics (IOU) for Segmentation Models Evaluation
+ Transfer Learning and Pretrained Deep Resnet Architecture
+ Segmentation Models Implementation in PyTorch using different Encoder and Decoder Architectures
+ Hyperparameters Optimization and Training of Segmentation Models
+ Test Segmentation Model and Calculate IOU, Class-wise IOU, Pixel Accuracy, Precision, Recall and F-score
+ Visualize Segmentation Results and Generate RGB Predicted Segmentation Map

A complete segmmentation pipeline is followed in [Deep Learning for Semantic Segmentation with Python & Pytorch](https://www.udemy.com/course/deep-learning-for-semantic-segmentation-with-python-pytorh/?referralCode=0009C809CCE66FFAADA3). 
![s3](https://user-images.githubusercontent.com/123977559/215627933-3087fdad-3bf6-4457-9d44-3deb14178f1b.png)

# Semantic Segmentation using Pretrained Model with Pytorch
This code uses the DeepLabV3 decoder and resnet101 encoder from torchvision library to perform semantic segmentation on an input image. The models have been trained on  COCO dataset with total of 21 classes including background. Model trained on the following Classes: background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, and tv/monitor. You can predict and segment images only belongs to these classes using pretrained model. If you want to train the segmentation models on your own image dataset to segment the images and to produce high results then follow the course [Deep Learning for Semantic Segmentation with Python & Pytorch](https://www.udemy.com/course/deep-learning-for-semantic-segmentation-with-python-pytorh/?referralCode=0009C809CCE66FFAADA3).  

The code first mounts the Google Colab drive to access the image file in Colab. Then, the pre-trained DeepLabV3 model is loaded using models.segmentation.deeplabv3_resnet101 from torchvision library.
```
from google.colab import drive
drive.mount('/content/drive')

#Load the pretrained Model
from torchvision import models
SegModel=models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
```
Next, the image is loaded using the Image module from PIL library and displayed using matplotlib. The image is then transformed using the Compose method from torchvision.transforms library. The transformed image is passed through the DeepLabV3 model, which returns the segmented image in the form of a tensor.The segmented image is then converted to a numpy array and the unique segments are found using np.unique method.
```
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

#Load Input image from google drive in Colab
img = Image.open('/content/drive/My Drive/Colab Notebooks/car.jpg')
plt.imshow(img); plt.show()
```
![input](https://user-images.githubusercontent.com/123977559/215645749-e2062ef6-f4df-444b-8a28-b17fa18b84be.jpg)


```
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
```

The decode_segmap function is defined to convert the segmented image to an RGB image. The final segmented RGB image is then displayed using matplotlib.
```
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
```
![output1](https://user-images.githubusercontent.com/123977559/215646280-3f09fd98-a543-453b-89d6-a0f5920c8e9c.JPG)

If you want to train the segmentation models on your own image dataset to segment the images and to produce high results then follow the course [Deep Learning for Semantic Segmentation with Python & Pytorch](https://www.udemy.com/course/deep-learning-for-semantic-segmentation-with-python-pytorh/?referralCode=0009C809CCE66FFAADA3).  

