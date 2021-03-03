import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

st.sidebar.write('#### Select an image to upload.')

uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)


## Add in sliders.
confidence_threshold = st.sidebar.slider('Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0.0, 1.0, 0.5, 0.01)
overlap_threshold = st.sidebar.slider('Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?', 0.0, 1.0, 0.5, 0.01)


default_url = 'https://www.microscopemaster.com/images/neutrophilsinchemopatient.jpg'
st.write('# Blood Cell Count Object Detection')

if uploaded_file:
    image = Image.open(uploaded_file)
else:
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)
    
## Subtitle.
st.write('### Inferenced Image')


buffered = io.BytesIO()
image.save(buffered, quality=90, format='JPEG')
# Base 64 encode.
img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode('ascii')

## Construct the URL to retrieve image.
upload_url = ''.join([
    'https://infer.roboflow.com/rf-bccd-bkpj9--1',
    '?access_token=vbIBKNgIXqAQ',
    '&format=image',
    f'&overlap={overlap_threshold * 100}',
    f'&confidence={confidence_threshold * 100}',
    '&stroke=2',
    '&labels=True'
])



## POST to the API.
r = requests.post(upload_url,
                  data=img_str,
                  headers={
    'Content-Type': 'application/x-www-form-urlencoded'
})


image = Image.open(BytesIO(r.content))

# Convert to JPEG Buffer.
buffered = io.BytesIO()
image.save(buffered, quality=90, format='JPEG')



# Display image.
st.image(image,
         use_column_width=True)


## Construct the URL to retrieve JSON.
upload_url = ''.join([
    'https://infer.roboflow.com/rf-bccd-bkpj9--1',
    '?access_token=vbIBKNgIXqAQ'
])

## POST to the API.
r = requests.post(upload_url,
                  data=img_str,
                  headers={
    'Content-Type': 'application/x-www-form-urlencoded'
})



## Save the JSON.
output_file = r.json()


countR = 0
countW = 0
for i in range(len(output_file["predictions"])):
  if output_file["predictions"][i]['class'] == 'RBC':
    countR += 1
  else:
    countW += 1
print("WBC : ", countW)
print("RBC : ", countR)
st.write(f"WBC count : {countW}")
st.write(f"RBC count : {countR}")

## Generate list of confidences.
confidences = [box['confidence'] for box in output_file['predictions']]

## Summary statistics section in main app.
st.write('### Summary Statistics')
st.write(f'Number of Bounding Boxes (ignoring overlap thresholds): {len(confidences)}')
st.write(f'Average Confidence Level of Bounding Boxes: {(np.round(np.mean(confidences),4))}')

## Histogram in main app.
st.write('### Histogram of Confidence Levels')
fig, ax = plt.subplots()
ax.hist(confidences, bins=10, range=(0.0,1.0))
st.pyplot(fig)



## Display the JSON in main app.
st.write('### JSON Output')
st.write(r.json())



