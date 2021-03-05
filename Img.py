import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow import keras



st.title("White Blood Cell Classification")


uploaded_file = st.file_uploader("Upload the Microscopic Image of the Blood Sample:",
                                 type=["jpeg","jpg", "png"])

if uploaded_file is not None:
    image_orig = Image.open(uploaded_file).convert('RGB')
    st.image(image_orig, caption='Uploaded image', use_column_width=True)
    st.write("Classifying...")
    MODEL_PATH = 'E:\Microscopic Insights\Microscopic\Microscopic Images\Microscopic-Images-Insights\weights\model_2.h5'
    model = keras.models.load_model(MODEL_PATH)  # can be problem with path use (custom_model.hdf5)
    my_bar = st.progress(0)

    img_array = np.array(image_orig)
    img_array = cv2.resize(img_array, (320, 240), interpolation=cv2.INTER_NEAREST)  # norm
    my_bar.progress(25)
    test_im = np.expand_dims(img_array, axis=0)

    my_bar.progress(50)
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
    st.write(f"WBC type :- {string}")

    my_bar.progress(100)
    st.success('Done')

