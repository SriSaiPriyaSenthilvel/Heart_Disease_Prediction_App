import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

st.title("Power Consumption Forecasting Using ARIMA")

st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Convert 'Datetime' column to datetime format
    if 'Datetime' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Datetime'], format='%m/%d/%Y %H:%M')
        data.set_index('Datetime', inplace=True)

        # Select target variable
        target_variable = st.sidebar.selectbox("Select the target variable", data.columns)

        # ARIMA parameters
        st.sidebar.header("ARIMA Parameters")
        p = st.sidebar.slider("Select p (AR term)", 0, 10, 1)
        d = st.sidebar.slider("Select d (I term)", 0, 2, 1)
        q = st.sidebar.slider("Select q (MA term)", 0, 10, 1)

        # Train-test split
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[:train_size], data[train_size:]

        # Fit ARIMA model
        if st.sidebar.button("Train Model"):
            st.write("### Training ARIMA Model...")
            model = ARIMA(train_data[target_variable], order=(p, d, q))
            fitted_model = model.fit()
            st.write("Model trained successfully!")

            # Forecast
            forecast = fitted_model.forecast(steps=len(test_data))

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))
            st.write(f"### Root Mean Squared Error (RMSE): {rmse}")

            # Plot results
            st.write("### Forecast vs Actual")
            plt.figure(figsize=(10, 6))
            plt.plot(train_data.index, train_data[target_variable], label='Training Data', color='blue')
            plt.plot(test_data.index, test_data[target_variable], label='Testing Data', color='green')
            plt.plot(test_data.index, forecast, label='Forecasted Data', color='orange')
            plt.xlabel('Date')
            plt.ylabel(target_variable)
            plt.title('ARIMA Forecasting for ' + target_variable)
            plt.legend()
            st.pyplot(plt)
    else:
        st.error("The dataset must contain a 'Datetime' column.")
else:
    st.info("Please upload a CSV file to get started.")
