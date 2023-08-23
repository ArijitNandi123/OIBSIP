import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

# Create a Streamlit app
st.title('Car Price Predictor')

# Sidebar widgets
companies = sorted(car['company'].unique())
selected_company = st.sidebar.selectbox('Select the company:', companies)

# Filter car models based on selected company
car_models = sorted(car[car['company'] == selected_company]['name'].unique())
selected_car_model = st.sidebar.selectbox('Select the model:', car_models)

# Rest of the sidebar widgets
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()
selected_year = st.sidebar.selectbox('Select Year of Purchase:', years)
selected_fuel_type = st.sidebar.selectbox('Select the Fuel Type:', fuel_types)
kilometers_driven = st.sidebar.text_input('Enter the Number of Kilometres driven:', '')

# Make a prediction
if st.sidebar.button('Predict Price'):
    prediction_data = np.array([selected_car_model, selected_company, selected_year, kilometers_driven, selected_fuel_type]).reshape(1, 5)
    predicted_price = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=prediction_data))
    st.write(f'Predicted Price: ₹{np.round(predicted_price[0], 2)}')

# Display app
st.write('### Car Details')
st.write(f'**Company:** {selected_company}')
st.write(f'**Car Model:** {selected_car_model}')
st.write(f'**Year of Purchase:** {selected_year}')
st.write(f'**Fuel Type:** {selected_fuel_type}')
st.write(f'**Kilometers Driven:** {kilometers_driven}')

