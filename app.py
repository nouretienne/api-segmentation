from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import json
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import os
import sys
from tensorflow import keras
from json import JSONEncoder
from tensorflow.keras import backend as K
from segmentation_models import Unet

app=FastAPI()

def jaccard_metric(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    Jaccard = (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
    return Jaccard

def jaccard_loss(y_true, y_pred):
    return 1-jaccard_metric(y_true, y_pred)

BACKBONE = 'resnet34'

class Npencode(JSONEncoder):
    def default(self, o ):
        if isinstance(o,np.ndarray):
            return o.tolist()
        return JSONEncoder.default(self, o)
     
model = Unet(BACKBONE, encoder_weights='imagenet', classes=8,activation='softmax')
model.load_weights('model_Unet_resNet_1.hdf5')

@app.post('/predict_image')
async def Predict_Image(file : UploadFile=File(...)):
    
    image = Image.open(BytesIO(await file.read()))
    image = np.array(image)
    image = Image.fromarray(image).resize((256,256),resample=Image.BILINEAR)
    batch_image = np.empty((16,256,256,3))
    batch_image[0] = image
    mask = model.predict(batch_image)
    dico = {'array':mask[0]}
    result_json = json.dumps(dico, cls=Npencode)

    return Response(result_json)

