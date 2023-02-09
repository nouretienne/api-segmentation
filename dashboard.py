import streamlit as st
import requests
import json
from PIL import Image

st.title('prediction de mask')

def load_image(image_file):
	img = Image.open(image_file)
	return img
def prediction(file):
    data_input = {'file': file}
    return requests.post('http://20.216.178.111:80/predict_image',files=data_input).content

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])



if image_file is not None:
    # To See details
	file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
	json_data = prediction(image_file)
    json_data 
