import pandas as pd
import numpy as np
import streamlit as st
import cv2
import tensorflow as tf
from PIL import Image,ImageOps


st.title('Image classification')
st.subheader('mask detection')


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('mobilenet.hdf5')
    return model
with st.spinner('Model is being loaded'):
    model = load_model()

file = st.sidebar.file_uploader('Upload Image File',type=['jpg','png'])

st.set_option('deprecation.showfileUploaderEncoding',False)

def import_and_predict(image_data,model):
    size = (224,224)
    image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    img_reshape = img[np.newaxis,...]

    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text('Uplaod Image')
else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    class_names = ['mask','no_mask']
    predictions = import_and_predict(image,model)
    score = tf.nn.sigmoid(predictions[0])
    st.write(predictions)
    st.write(score)
    print('image belongs to {} with {:.2f} percent confidence'
    .format(class_names[np.argmax(score)],100*np.max(score)))
    st.sidebar.text('image belongs to {predictions}')
