# load library
import streamlit as st
import numpy as np
import pandas as pd
#from pandas_datareader import data as pdr

import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
#import matplotlib.pyplot as plt
#from matplotlib import style
import datetime as dt
import yfinance as yf
import datetime

# print title of web app
st.title("Stock Market Analysis and Prediction")
st.markdown("> Stock Market Analysis and Prediction is the project on technical analysis, visualization and prediction using data provided by Yahoo Finance.")
st.markdown("> It is web app which predicts the future value of company stock or other ﬁnancial instrument traded on an exchange.")

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Load data from yahoo finance.
#start=dt.date(2010,1,1)
#end=dt.date.today()
#data=yf.download("GOOG", start, end)
data=pd.read_csv('SPY.csv')


#fill nan vale with next value within columns
data.fillna(method="ffill",inplace=True)

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

# create checkbox
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
 
# show the description of data
st.subheader('Detail description about Datasets:-')
descrb=data.describe()
st.write(descrb)

#create new columns like year, month, day
#data["Year"]=data.index.year
#data["Month"]=data.index.month
#data["Weekday"]=data.index.day_name()

# dislay graph of open and close column
st.subheader('Graph of Close & Open:-')
st.line_chart(data[["Open","Close"]])

# display plot of Adj Close column in datasets
st.subheader('Graph of Adjacent Close:-')
st.line_chart(data['Adj Close'])

# display plot of volume column in datasets
st.subheader('Graph of Volume:-')
st.line_chart(data['Volume'])

# create new cloumn for data analysis.
data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
data['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0
data = data[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]

# display the new dataset after modificaton
st.subheader('Newly format DataSet:-')
st.dataframe(data.tail(500))

forecast_col = 'Adj Close'
forecast_out = int(math.ceil(0.01 * len(data)))
data['label'] = data[forecast_col].shift(-forecast_out)

X = np.array(data.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
data.dropna(inplace=True)
y = np.array(data['label'])

# split dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)


