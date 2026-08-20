import joblib
import numpy as np
import streamlit as st


model = joblib.load('linear_house_model.pkl')


st.set_page_config(
    page_title='House Price Predictor',
    page_icon='🏠',
    layout='centered'
)


st.title('🏠 House Price Prediction App')
st.write(
    "Enter the property details below to estimate the market price "
    "based on your trained model."
)


st.subheader('🏡 Property Features')

col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input(
        '📐 Square Feet',
        min_value=500,
        max_value=10000,
        value=1800,
        step=50,
        help="Enter the total built-up area of the property."
    )

    age = st.number_input(
        '⏳ Property Age (Years)',
        min_value=0,
        max_value=100,
        value=10,
        step=1,
        help="How old is the property in years?"
    )

with col2:
    bedrooms = st.slider(
        '🛏️ Bedrooms',
        min_value=1,
        max_value=10,
        value=3,
        help="Select the number of bedrooms in the house."
    )


st.info("💡 Tip: Larger homes with more bedrooms and lower age generally predict higher prices.")


if st.button('Estimate Price', type='primary'):
    input_data = np.array([[sqft, bedrooms, age]])

    predicted_price = model.predict(input_data)[0]


    st.markdown('---')
    st.success(f'### Estimated House Price: **${predicted_price:,.2f}**')
