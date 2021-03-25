#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

originalImage = cv2.imread('Head-MRI.tif',0)


# In[5]:



resized_image = cv2.resize(originalImage, originalImage.shape, interpolation = cv2.INTER_CUBIC)
resized_image = resized_image.astype(np.float32)
resized_image /= 255.

floatImage = np.expand_dims(resized_image, 0)


# In[11]:


floatImage = floatImage.reshape(256,256)
fig, axs = plt.subplots(2, 1)

axs[0].imshow(originalImage, cmap="gray" , vmin = 0 , vmax= 255)
axs[0].set_title('original Image');axs[0].axis('off')
axs[1].imshow(floatImage, cmap="gray", vmin = 0 , vmax= 1)
axs[1].set_title('float image');axs[1].axis('off')

plt.subplots_adjust(bottom = 0.025 , hspace =0.5)


# In[ ]:


row180_orginal = originalImage[180 , :]
row150_orginal = originalImage[150 , :]
row180_float = floatImage[180 , :]
row150_float = floatImage[150 , :]


# In[ ]:


plt.plot(row180_float)


# In[ ]:


fig, axs = plt.subplots(2, 1)
axs[0].plot(row180_orginal , color='red',label = "row 180 of original image")
axs[0].plot(row150_orginal, color='blue',label = "row 150 of original image")
axs[0].set_title('orginal image') 
axs[0].legend()
axs[1].plot(row180_float , color='red',label = "row 180 of float image")
axs[1].plot(row150_float , color='blue',label = "row 150 of float image")
axs[1].set_title('float image') 
axs[1].legend()
plt.subplots_adjust(bottom = 0.025 , hspace =0.5)


# In[ ]:


plt.figure(figsize=(35, 10))
row150_orginal = row150_orginal.reshape(1,256)
plt.imshow(row150_orginal , cmap = 'gray')
plt.title('row 150', fontsize=30)
plt.axis('off')


# In[ ]:


plt.figure(figsize=(35, 15))
row180_orginal = row180_orginal.reshape(1,256)
plt.imshow(row180_orginal , cmap = 'gray')
plt.title('row 180', fontsize=30)
plt.axis('off')


# In[ ]:




