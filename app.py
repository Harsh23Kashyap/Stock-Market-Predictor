import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import plotly.express as px
import streamlit as st
import requests
from streamlit_lottie import st_lottie
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title=" Stonks Trends Prediction", #The page title, shown in the browser tab.(should be Placement Details)
    initial_sidebar_state="auto", #The way sidebar should start out. Auto shows it in desktop.
    page_icon=":computer:", #The page favicon. Use the computer emoji
    layout="wide", #The way page content should be laid out. "wide" uses the entire screen.
    menu_items={ #Configure the menu that appears on the top-right side of this app.
            'About': 'https://www.linkedin.com/in/harsh-kashyap-79b87b193/', #A markdown string to show in the About dialog. Used my linkedIn id
     }
)
def load_lottieurl(url: str):
        r = requests.get(url) #Make a request to a web page, and return the status code:
        if r.status_code != 200: #200 is the HTTP status code for "OK", a successful response.
            return None
        return r.json() #return the animated gif

from datetime import date
from datetime import timedelta
today = date.today()
# Yesterday date
yesterday = today - timedelta(days = 1)
start='2010-01-01'
end=yesterday;


if(today.isoweekday()==1):
    current = yesterday = today - timedelta(days = 2)
else:  
    current = yesterday = today - timedelta(days = 1)
    


st.title(":computer:  Stock Market Predictor") #Title heading of the page
st.markdown("##") 

with st.sidebar:
    st.title("World Market")
    st.title("NIFTY")
    nif = data.DataReader('^NSEI','yahoo',current)['Close']
    st.header(nif.iloc[0].round(2))
    st.markdown("""---""")
    
    st.title("SENSEX")
    sen = data.DataReader('^BSESN','yahoo',current)['Close']
    st.header(sen.iloc[0].round(2))
    st.markdown("""---""")
    
    st.title("S&P FUTURES")
    sp = data.DataReader('ES=F','yahoo',current)['Close']
    st.header(sp.iloc[0].round(2))
    st.markdown("""---""")
    
    st.title("GOLD")
    gold = data.DataReader('GC=F','yahoo',current)['Close']
    st.header(gold.iloc[0].round(2))
    st.markdown("""---""")
    
    st.title("DOW")
    dow = data.DataReader('YM=F','yahoo',current)['Close']
    st.header(dow.iloc[0].round(2))
    st.markdown("""---""")
    
    st.title("NASDAQ")
    nas = data.DataReader('NQ=F','yahoo',current)['Close']
    st.header(nas.iloc[0].round(2))
    st.markdown("""---""")
    
    st.title("CRUDE OIL")
    gold = data.DataReader('CL=F','yahoo',current)['Close']
    st.header(gold.iloc[0].round(2))
    st.markdown("""---""")

   


st.subheader("Enter Stock Ticker")
user_input=st.text_input('','HDB')

val=True
try:
    df = data.DataReader(user_input,'yahoo',start,end)
except:
    val=False
    st.write("Wrong ticker. Select again")
    st.markdown("""---""")
    error = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_k1rx9jox.json") #get the animated gif from file
    st_lottie(error, key="Dashboard1", height=400) #change the size to height 400

if val==True:
    date=df.index

    #Checks if which parameters in hsc_s which is named as branch in sidebar is checked or not and display results accordingly
    

    left_column, right_column = st.columns(2) #Columns divided into two parts
    with left_column:
        dashboard1 = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_kuhijlvx.json") #get the animated gif from file
        st_lottie(dashboard1, key="Dashboard1", height=400) #change the size to height 400
    with right_column:
        dashboard2 = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_i2eyukor.json") #get the animated gif from file
        st_lottie(dashboard2, key="Dashboard2", height=400) #change the size to height 400


    st.markdown("""---""")
    #Describing data 
    st.subheader('Data from 2008 to '+str(end.year))
    st.write(df.describe())

    st.markdown("""---""")
    #Visualisations
    st.subheader("Closing Price vs Time Chart of "+str(user_input)) #Header
    #plot a line graph
    fig_line = px.line(
        df,  
        x = df.index,  
        y = "Close", 
        width=1400, #width of the chart
        height=750, #height of the chart
    )
    #remove the background of the back label
    fig_line.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",  #rgba means transparent
        xaxis=(dict(showgrid=False)) #dont show the grid
    )
    #plot the chart
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("""---""")


    st.subheader("Closing Price vs Time with 100MA of "+str(user_input)) #Header
    ma100=df.Close.rolling(100).mean()
    #plot a line graph
    fig_line = px.line( 
        ma100,
        x = df.index,  
        y = ma100, 
        width=1400, #width of the chart
        height=750, #height of the chart
    )
    #remove the background of the back label
    fig_line.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",  #rgba means transparent
        xaxis=(dict(showgrid=False)) #dont show the grid
    )
    #plot the chart
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("""---""")


    st.subheader("Closing Price vs Time with 1 year moving average of  "+str(user_input)) #Header
    ma365=df.Close.rolling(365).mean()
    #plot a line graph
    fig_line = px.line( 
        ma365,
        x = df.index,  
        y = ma365, 
        width=1400, #width of the chart
        height=750, #height of the chart
    )
    #remove the background of the back label
    fig_line.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",  #rgba means transparent
        xaxis=(dict(showgrid=False)) #dont show the grid
    )
    #plot the chart
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("""---""")



    #Splitting data into training and testing

    data_training= pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
    data_testing= pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])
    ydate= date[int(len(df)*0.7):int(len(df))]
    print(data_training.shape)
    print(data_testing.shape)

    #normalising data
    
    scaler=MinMaxScaler(feature_range=(0,1))

    dataset_train = scaler.fit_transform(data_training)
    dataset_test = scaler.transform(data_testing)

    def create_dataset(df):
        x = []
        y = []
        for i in range(50, df.shape[0]):
            x.append(df[i-50:i, 0])
            y.append(df[i, 0])
        x = np.array(x)
        y = np.array(y)
        return x,y

    #Creating dataset
    x_train, y_train = create_dataset(dataset_train)
    x_test, y_test = create_dataset(dataset_test)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #Load my model
    model=load_model('stock_prediction.h5')

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    cydate=ydate[50:]
    st.markdown("""---""")
    st.subheader("Actual Vs Predicted Price Graph for "+user_input)
    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_facecolor('#000041')
    ax.plot(cydate,y_test_scaled, color='red', label='Original price')
    plt.plot(cydate,predictions, color='cyan', label='Predicted price')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stocks for the company "+str(user_input))
    plt.legend()
    st.pyplot(fig)

    st.markdown("""---""")
