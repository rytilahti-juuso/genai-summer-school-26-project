import streamlit as st
from PIL import Image  


img = Image.open("test_image.png") 
 


st.title("ArXiv article categories")
status = st.radio("Select view:", ['Category listing', 'Category distribution chart'])
categorylist = ['Dancing', 'Reading', 'Sports']


# Display the selected option using success message
if status == 'Category listing':
    categories = st.selectbox("Select a category:", categorylist)
else:
    st.image(img, width=200)




