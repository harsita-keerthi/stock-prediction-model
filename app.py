
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st
import plotly.graph_objs as go

import tensorflow as tf
print(tf.__version__)

# define starting and ending point of data
start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction')

st.sidebar.subheader('Customize Your View')
user_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime(start))
end_date = st.sidebar.date_input('End Date', pd.to_datetime(end))
ma_period_100 = st.sidebar.slider('100 Days Moving Average Period', min_value=50, max_value=200, value=100)
ma_period_200 = st.sidebar.slider('200 Days Moving Average Period', min_value=100, max_value=400, value=200)

df =  yf.download(user_input, start=start, end=end)

# describe data
st.subheader('{} Data from {} to {}'.format(user_input, start_date, end_date))
st.write(df.describe())

# visualizations
st.subheader('Closing Price vs. Time')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))
fig.update_layout(xaxis_title='Date', width=1200, height=700, yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

st.subheader('Closing Price and Moving Averages')
ma100 = df['Close'].rolling(ma_period_100).mean()
ma200 = df['Close'].rolling(ma_period_200).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))
fig.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='{} Days Moving Average'.format(ma_period_100), line=dict(color='red')))
fig.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines', name='{} Days Moving Average'.format(ma_period_200), line=dict(color='green')))
fig.update_layout(xaxis_title='Date', width=1200, height=700, yaxis_title='Price', 
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
st.plotly_chart(fig, use_container_width=True)


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
y_predicted = y_predicted.flatten() if y_predicted.ndim > 1 else y_predicted

fig = go.Figure()

fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Original Price', line=dict(color='green')))
fig.add_trace(go.Scatter(y=y_predicted, mode='lines', name='Predicted Price', line=dict(color='red')))

fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Price',
    legend_title='Legend',
    width=1200,
    height=700,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    yaxis=dict(
        range=[min(min(y_test), min(y_predicted)), max(max(y_test), max(y_predicted))]
    )
)

st.plotly_chart(fig, use_container_width=True)

explanation = (
    f"**Stock Ticker:** {user_input}\n\n"
    "The model predicts the stock price trend based on historical data. "
    "The green line represents the actual closing prices, while the red line represents the predicted prices. "
    "This visualization helps in assessing the performance of the model and understanding how well it predicts future trends. "
    "The accuracy of predictions may vary based on market conditions and historical data used."
)
st.markdown(explanation)
