import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow import keras

"""
When the Classification class is called the funciton in the Classification class 
predicts the type of WBC and returns the result
"""

class Classification:
    def classification_type(self, image_org):
        #image_orig = Image.open(uploaded_file).convert('RGB')
        MODEL_PATH = 'E:\Microscopic Insights\Microscopic\Microscopic Images\Microscopic-Images-Insights\model_2.h5'
        model = keras.models.load_model(MODEL_PATH)
        img_array = np.array(image_org)
        img_array = cv2.resize(img_array, (320, 240), interpolation=cv2.INTER_NEAREST)  # norm
        test_im = np.expand_dims(img_array, axis=0)
        prediction = model.predict(test_im / 255)
        prediction = np.argmax(prediction)
        if prediction == 0:
            string = "EOSINOPHIL"
        elif prediction == 1:
            string = "LYMPHOCYTE"
        elif prediction == 2:
            string = "MONOCYTE"
        elif prediction == 3:
            string = "NEUTROPHIL"
        
        return string