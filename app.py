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


