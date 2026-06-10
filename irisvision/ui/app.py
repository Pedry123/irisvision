import asyncio

import streamlit as st
from sqlalchemy.orm import Session

from irisvision.database import engine
from irisvision.utils import predict_image

if 'button' not in st.session_state:
    st.session_state['button'] = False

st.title('IrisVision - An Iris Flower Classifier')
st.markdown(
    'This app classifies iris flowers based on their'
    ' images using a pre-trained model.'
)

st.header('Input Image')
st.markdown('Upload an image of an iris flower to classify it.')
uploaded_file = st.file_uploader(
    'Choose an image...', type=['jpg', 'jpeg', 'png']
)


def click_button():
    st.session_state['button'] = not st.session_state['button']


if uploaded_file is not None:
    with Session(engine) as session:
        image_bytes = uploaded_file.read()
        response = asyncio.run(predict_image(image_bytes, session))
        st.markdown(f'**Prediction:** {response.prediction}')
        st.markdown(f'**Confidence:** {response.confidence:.2f}')

        col1, col2 = st.columns(2, gap='large')
        col1.vertical_alignment = 'center'
        col1.image(uploaded_file, caption=f'Response: {response.prediction}')
        col2.header('Is the prediction correct?', text_alignment='center')
        col3, col4 = col2.columns(2, vertical_alignment='center')

        buttonyes = col3.button('Yes', use_container_width=True)
        buttonno = col4.button('No', use_container_width=True)

        if not st.session_state['button']:
            click_button()
        if buttonyes:
            st.session_state['button'] = True
            st.success('Thank you for your feedback!')
        elif buttonno:
            st.session_state['button'] = True
            st.error(
                'Thank you for your feedback!'
                ' We will work on improving the model.'
            )
