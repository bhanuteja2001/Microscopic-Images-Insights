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
import Classification_Model
import pandas as pd

st.sidebar.write('#### Select an image to upload.')

uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)

st.sidebar.markdown('---')
URL = st.sidebar.text_input("Enter the Image URL")
#print("UPLOADED IMAGE : ",uploaded_file)
st.sidebar.markdown('---')
## Add in sliders.
confidence_threshold = st.sidebar.slider('Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0.0, 1.0, 0.5, 0.01)
overlap_threshold = st.sidebar.slider('Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?', 0.0, 1.0, 0.5, 0.01)


#default_url = 'https://www.microscopemaster.com/images/neutrophilsinchemopatient.jpg'
st.write('# WBC Classification and Blood Cells Count')

if uploaded_file:
    image = Image.open(uploaded_file)
elif URL:
    image = Image.open(requests.get(URL,stream=True).raw)
else:
    url = 'https://github.com/bhanuteja2001/Microscopic-Images-Insights/blob/main/Images/neutrophilsinchemopatient.jpg?raw=true'
    url1 = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00062_jpg.rf.d54b89916d935069b63e08dee5cbfc27.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)
#print(image)  
## Subtitle.
st.write('### Inferenced Image')

Obj = Classification_Model.Classification()
Pred = Obj.classification_type(image)


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


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'




## Save the JSON.
output_file = r.json()


countR = 0
countW = 0
countP = 0
for i in range(len(output_file["predictions"])):
  if output_file["predictions"][i]['class'] == 'RBC':
    countR += 1
  elif output_file["predictions"][i]['class'] == 'Platelets':
      countP += 1
  else:
    countW += 1


st.write('### Insights from the above Image')

st.write(f"The WBC Class is : {Pred}")
print("WBC : ", countW)
print("RBC : ", countR)
print("Platelets : ", countP)
st.write(f"WBC count : {countW}")
st.write(f"RBC count : {countR}")
st.write(f"Platelets count : {countP}")

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

average = np.round(np.mean(confidences),4)
if average < 0.60:
    st.markdown("<b><font color=‘#FF0000’>The Image Quailty is not good enough to be identified!!</font></b>", unsafe_allow_html=True)

df = pd.DataFrame.from_dict(output_file['predictions'])


#Table and Download Button
st.write('### Table')
st.write(df)
if st.button('Download Dataframe as CSV'):
    tmp_download_link = download_link(df, 'YOUR_DF.csv', 'Click here to download your data!')
    st.markdown(tmp_download_link, unsafe_allow_html=True)
## Display the JSON in main app.
#st.write('### JSON Output')
#st.write(r.json())




