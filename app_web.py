import streamlit as st
import requests
import numpy as np
import json
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img

st.title('prediction de mask')

def load_image(image_file):
	img = Image.open(image_file)
	return img

def recolor(img):
    prediction_color = {0:[0, 0, 0],
                        1:[204, 204, 204],
                        2:[148, 148, 148],
                        3:[220, 220, 0],
                        4:[111, 196, 53],
                        5:[188, 230, 254],
                        6:[191, 30, 62],
                        7:[3, 50, 126]}
    new_img = np.zeros((*img.shape, 3))
    for k, v in prediction_color.items():
        new_img[img==k] = v
    new_img = new_img.astype('uint8')
    return new_img



def prediction(file):
    data_input = {'file': file}
    json_data = requests.post('http://20.216.178.111:80/predict_image',files=data_input).content
    json_data = json.loads(json_data)
    y = np.asarray(json_data['array'])
    return recolor(np.argmax(y, axis=2))






image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])



if image_file is not None:
    # To See details
	file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
	st.image(prediction(image_file),width=250)

    #st.image(json_data,width=250)