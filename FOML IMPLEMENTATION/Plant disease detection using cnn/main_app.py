import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions)
    return result_index

st.header("PLANT DISEASE DETECTION")
image_path="agriculture.jpeg"
st.image(image_path,use_column_width=True)
test_image=st.file_uploader("Choose an Image:")
if(st.button("Show Image")):
    st.image(test_image,use_column_width=True)
if(st.button("Predict")):
    st.write("Prediction")
    result_index=model_prediction(test_image)
    class_name=['Healthy', 'Powdery', 'Rust']
    st.success("The Result: {}".format(class_name[result_index]))
    data=class_name[result_index]
    img=0
    if(data=="Rust"):
        img="Fertilizers/rust_fertilizer.jpg"
        st.write("The Fertilizer recommended is: ")
        st.image(img,use_column_width=True)
    elif(data=="Powdery"):
        img="Fertilizers/Powdery.jpg"
        st.write("The Fertilizer recommended is: ")
        st.image(img,use_column_width=True)
    elif(data=="Healthy"):
        st.success("No disease detected, the plant is healthy.")
    else:
        st.write("Invalid")
        

