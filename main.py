import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from datetime import date
import pmdarima as pm
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import base64


# Function to get or create session state
def get_session_state():
    return st.session_state

# Function to scrape data from the web
@st.cache_data()
def webscrap():
    url = 'https://www.moneycontrol.com/news/photos/business/stocks/'
    agent = {"User-Agent": "Mozilla/5.0"}
    page = requests.get(url, headers=agent)
    soup = BeautifulSoup(page.text, 'html.parser')
    section = soup.find_all('a', attrs={"title": re.compile("^Buzzing Stocks")})[0]
    link = section["href"]
    url2 = link
    page = requests.get(url2, headers=agent)
    soup2 = BeautifulSoup(page.text, 'html.parser')
    l1 = soup2.find_all('strong')
    l2 = [data.contents[0].strip(":").replace('&', '') for data in l1]
    return l2

# Function to get ticker symbol
@st.cache_data()
def get_ticker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "India"}
    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    if 'quotes' in data and data['quotes']:
        company_code = data['quotes'][0]['symbol']
        return company_code
    else:
        return ''

# Function to get name and ticker
@st.cache_data()
def get_name_ticker():
    l2 = webscrap()
    l3 = {}
    for name in l2:
        ticker_symbol = get_ticker(name)
        if ticker_symbol != '' and enough_historical_data(ticker_symbol):
            l3[name] = ticker_symbol
    return l3

@st.cache_data()
def enough_historical_data(ticker_symbol):
    min_data_points_threshold = 50

    try:
        # Download historical data for the specified stock
        historical_data = yf.download(ticker_symbol, start="2010-01-01", end="2023-12-31", progress=False)
        return len(historical_data) >= min_data_points_threshold
    except:
        # Handle any exceptions, such as the stock not being found or insufficient data
        return False


selected = option_menu(
    menu_title="Watchlist.io",
    options=["Home", "Downloads"],
    orientation="horizontal",
    default_index=0
)

# Function for Exploratory Data Analysis
def perform_eda(data):
    st.header("Exploratory Data Analysis (EDA)")

    # 1. Overview of Historical Stock Price Data
    st.subheader("Overview of Historical Stock Price Data")
    st.write(data.head())
    st.write(data.describe())

    # 2. Time Series Visualization
    st.subheader("Time Series Visualization")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data['Date'], data['Close'], label="Close Price")
    ax.set(title="Time Series Analysis", xlabel="Date", ylabel="Close Price")
    ax.legend()
    st.pyplot(fig)

    # 3. Distribution of Stock Metrics
    st.subheader("Distribution of Stock Metrics")
    selected_metric = st.radio("Select a metric to visualize", ['Close', 'Volume'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[selected_metric], kde=True, bins=30, color='skyblue', ax=ax)
    ax.set(title=f"Distribution of {selected_metric}", xlabel=selected_metric, ylabel="Frequency")
    st.pyplot(fig)

# Main function
if selected == "Home":
    # Initialize session state
    session_state = get_session_state()
    st.header('Trending Stocks of the Day :money_with_wings:', divider='green')
    
    l3 = get_name_ticker()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    n = len(l3)
    
    c = 0
    for key in l3:
        if c % 3 == 0:
            with col1:
                if st.button(key):
                    # Store the selected stock in session state
                    session_state.selected_stock = l3[key]
                    # Navigate to another page (you can replace this with your Bollinger Bands page logic)
                    st.experimental_rerun()
            c += 1
        elif c % 3 == 1:
            with col2:
                if st.button(key):
                    session_state.selected_stock = l3[key]
                    st.experimental_rerun()
            c += 1
        else:
            with col3:
                if st.button(key):
                    session_state.selected_stock = l3[key]
                    st.experimental_rerun()
            c += 1
    if hasattr(session_state, "selected_stock"):
        t_symbol = session_state.selected_stock

        START = "2015-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365

        @st.cache_data
        def load_data(ticker):
            data = yf.download(t_symbol, START, TODAY)
            data.reset_index(inplace=True)
            return data
        data_load_state = st.text('Loading data...')
        data = load_data(t_symbol)
        data_load_state.text('Loading data... done!')

        #perform_eda(data)

        st.subheader('Raw data')
        st.write(data.tail())

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
        plot_raw_data()

        # Predict forecast with Prophet.
        df_train = data[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast.tail())
            
        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)
       


# Downloads section
if selected == "Downloads":
    st.title(f"You have selected {selected}")
    l3 = get_name_ticker()
    # Downloadable CSV button
    st.markdown("""
    ### Download Stock Ticker Data
    Click the button below to download the stock ticker data as a CSV file.
    """)
    download_button = st.button("Download CSV")

    # Check if the button is clicked
    if download_button:
        # Create a DataFrame from the dictionary
        df_ticker = pd.DataFrame(list(l3.items()), columns=['Company', 'Ticker Symbol'])

        # Save DataFrame to CSV
        csv_file = df_ticker.to_csv(index=False)

        # Create a download link
        b64 = base64.b64encode(csv_file.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="stock_tickers.csv">Download CSV File</a>', unsafe_allow_html=True)
