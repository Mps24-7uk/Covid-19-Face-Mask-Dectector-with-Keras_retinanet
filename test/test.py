import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import argparse
import json
import cv2
import os
import numpy as np
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True, help = "path of input image")

args = vars(ap.parse_args())

model_path =  "/home/lenovo/Downloads/keras-retinanet/inference/covid.h5"

model = models.load_model(model_path, backbone_name='resnet50')


labels_to_names = {0: 'Mask', 1: 'No Mask'} 

color_coding = {'Mask': (0, 255, 0) , 'No Mask': (255, 0, 0)}


# load image
img_path = args["image"]
image = read_image_bgr(img_path)

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR), axis=0))
#boxes, scores, labels = model.predict_on_batch(cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR))
#print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale
labels_info = []
# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.6:
        break
        
    color = color_coding[labels_to_names[label]]
    b = box.astype(int)
    caption = "{},{:.3f}".format(labels_to_names[label], score)
    cv2.putText(draw, caption, (b[0],b[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)
    cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), color, 5)


cv2.imwrite("result/out.jpg",cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))

