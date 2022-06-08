import numpy as np
import pandas as pd
import math
import streamlit as st
from datetime import datetime, date
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

import yfinance as yf

header=st.container()
dataset=st.container()
features=st.container()
modelTraining=st.container()

a = st.selectbox("Select the currency code ", (
    'EURUSD=X', 'JPY=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'EURJPY=X', 'GBPYJPY=X', 'EURGBP=X', 'EURCAD=X',
    'EURSEK=X', 'EURCHF=X', 'EURHUF=X', 'EURJPY=X', 'CNY=X', 'HKD=X', 'SGD=X', 'INR=X', 'MXN=X''PHP=X', 'IDR=X',
    'MYR=X''ZAR=X''RUB=X'))
c = st.selectbox('Select the time Period', ('max', '10y', '5y', '1y', '1m', '10D', '2D'))
df = yf.Ticker(a)
df = df.history(period=c)
print(df)

b = st.selectbox("Select the Column you want to view", ('High', 'Low', 'Open', 'Close'))
def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")
    # st.write("f")

    with header:
        st.title('Currency Analysis and Forecasting')
    with dataset:
        st.text(df)

def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")
    st.write("e")
    with features:
        st.header("Features")
        plt.show()
        st.subheader('Trend of the currency ')
        st.line_chart(df[b])
        if c == 'max':
            st.subheader('Maximum Attained in Each Year')
            st.bar_chart(df.resample(rule='A').max()[b], 10)

            st.subheader('Minimum Attained in Each Year')
            st.bar_chart(df.resample(rule='A').min()[b], 10)

            st.subheader('Average Attained in Each Year')
            st.bar_chart(df.resample(rule='A').mean()[b], 10)

        # st.subheader('Moving Average with window 10 (Smoothens the graph)')
        df['10 days rolling'] = df[b].rolling(10).mean()
        # st.subheader('Moving Average with window 10 (Smoothens the graph)')
        df['30 days rolling'] = df[b].rolling(30).mean()
        df['50 days rolling'] = df[b].rolling(50).mean()

        # /df_last_y['30 days rolling'] = df_last_y[b].rolling(30).mean()
        st.subheader('Moving Average with window 30 (Smoothens the graph)')

        st.line_chart(df['30 days rolling'])
        # st.subheader('Moving Average with window 30 of last year (Smoothens the graph)')

        # st.line_chart(df_last_y['30 days rolling'])

        st.subheader('Moving Average with window 30 and Actual graph (Smoothens the graph)')
        st.line_chart(df[[b, '30 days rolling', ]])

        # st.subheader('Moving Average with window 30 and Actual graph of  last year (Smoothens the graph)')
        # st.line_chart(df_last_y[[b, '30 days rolling']])
        acf = plot_acf(df[b])

        # st.line_chart(acf)

        G = ['df[b].expanding().mean()', 'df[b]']
        st.subheader('Cumulative moving Average')
        st.line_chart(df[b].expanding().mean())
        # st.line_chart(df[b].resample(rule='A').mean())

        st.subheader('Exponential Moving Average')
        df['Ema_0.1'] = df[b].ewm(alpha=0.1, adjust=False).mean()
        st.line_chart(df['Ema_0.1'])

        st.subheader('Exponential Moving Average with Actual graph')
        st.line_chart(df[[b, 'Ema_0.1']])


def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")
    model = SARIMAX(df[b], order=(1, 1, 1))
    model_fit = model.fit()
    x = pd.date_range(end = datetime.today(), periods = 100).to_pydatetime().tolist()

    forecast_1 = model_fit.forecast(steps=100, exog=x)
    st.line_chart(forecast_1)
    st.write("Predicted Values")
    temp = pd.DataFrame({"Date":x,"Forecast":forecast_1})
    st.write(temp)

page_names_to_funcs = {
    "Main Page": main_page,
    "Page 2": page2,
    "Page 3": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()




