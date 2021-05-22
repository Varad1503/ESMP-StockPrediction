
import streamlit as st
from datetime import date
import sklearn
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

import pickle

pickle_in = open("stpredict.pkl","rb")
mode1 = pickle.load(pickle_in)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('ESMP : Stock Prediction using Machine Learning ')
st.subheader("by Varad Deshmukh (32171) Amey Todkar (32168)")

selected_stock = st.text_input("Enter a ticker (if indian add .NS) :")

week = st.slider('Weeks of prediction:', 1, 4)
period = week * 7

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail(period))
    
st.write(f'Forecast plot for {week} week')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

