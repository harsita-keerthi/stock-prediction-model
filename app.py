# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st

import tensorflow as tf
print(tf.__version__)

# define starting and ending point of data
start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df =  yf.download(user_input, start=start, end=end)

# describe data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

# visualizations
st.subheader('Closing Price vs. Time')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs. Time with 100 Days Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, label='100 Days Moving Average', color='red')
st.pyplot(fig)

st.subheader('Closing Price vs. Time with 100 Days and 200 Days Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, label='100 Days Moving Average', color='red')
plt.plot(ma200, label='200 Days Moving Average', color='green')
st.pyplot(fig)

# split data into training and test (70% in training, 30% in testing)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_arr = scaler.fit_transform(data_training)

# load model
model = load_model('stock_prediction_model.h5')

# testing
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

# split data for testing 
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# final prediction graph
st.subheader('Predictions vs. Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'g', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

explanation = (
    f"**Stock Ticker:** {user_input}\n\n"
    "The model predicts the stock price trend based on historical data. "
    "The green line represents the actual closing prices, while the red line represents the predicted prices. "
    "This visualization helps in assessing the performance of the model and understanding how well it predicts future trends. "
    "The accuracy of predictions may vary based on market conditions and historical data used."
)
st.markdown(explanation)


