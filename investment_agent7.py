import streamlit as st
from transformers import AutoTokenizer
import requests
import yfinance as yf
import pandas as pd
import time
from googletrans import Translator

# Hugging Face API Token
HUGGINGFACE_API_TOKEN = "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"

# Language selection
LANGUAGES = {
    "en": "English",
    "zh-cn": "中文",  # Simplified Chinese
    "es": "Español",  # Spanish
    "fr": "Français"  # French
}

# Streamlit setup
st.title("AI Investment Agent 📈🤖")
st.caption("This app allows you to compare the performance of two stocks and generate detailed reports.")

# Language selector
lang = st.selectbox("Select language", list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])

# Initialize global translator
translator = Translator()

# Translation function with retries
def translate(text, lang, retries=3):
    for attempt in range(retries):
        try:
            translated = translator.translate(text, src='en', dest=lang)
            return translated.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)  # Retry with delay
                continue
            st.error(f"Translation error: {e}")
            return text

# Input fields for the stocks
stock1 = st.text_input(translate("Enter the first stock symbol", lang))
stock2 = st.text_input(translate("Enter the second stock symbol", lang))

# Time range selection
time_period = st.selectbox(translate("Select the time period for stock comparison", lang), 
                           ["1d", "5d", "1mo", "6mo", "1y", "5y", "max"])

# Load tokenizer for Hugging Face API
@st.cache_resource
def load_tokenizer():
    model_name = "tiiuae/falcon-7b-instruct"  # Hugging Face model
    return AutoTokenizer.from_pretrained(model_name)

# Function to fetch stock data
def get_stock_data(stock_symbol, period):
    try:
        ticker = yf.Ticker(stock_symbol)
        history = ticker.history(period=period)
        if history.empty:
            return None
        return history
    except Exception as e:
        st.error(f"Error fetching data for {stock_symbol}: {e}")
        return None

# Function to query Hugging Face API
def query_huggingface_api(payload):
    api_url = f"https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        st.error(f"API Error: {response.status_code} - {response.json()}")
        return None

# Main logic for the app
if stock1 and stock2:
    tokenizer = load_tokenizer()

    # Fetch stock data
    stock1_data = get_stock_data(stock1, time_period)
    stock2_data = get_stock_data(stock2, time_period)

    if stock1_data is not None and stock2_data is not None:
        # Display comparison chart
        st.subheader(translate("Stock Price Comparison Chart", lang))
        combined_data = pd.DataFrame({
            stock1: stock1_data['Close'],
            stock2: stock2_data['Close']
        })
        st.line_chart(combined_data)

        # Prepare summaries for both stocks
        stock1_summary = {
            translate("Latest Price", lang): stock1_data['Close'].iloc[-1],
            translate("Start Price", lang): stock1_data['Close'].iloc[0],
            translate("Period Change (%)", lang): ((stock1_data['Close'].iloc[-1] - stock1_data['Close'].iloc[0]) / stock1_data['Close'].iloc[0]) * 100
        }
        stock2_summary = {
            translate("Latest Price", lang): stock2_data['Close'].iloc[-1],
            translate("Start Price", lang): stock2_data['Close'].iloc[0],
            translate("Period Change (%)", lang): ((stock2_data['Close'].iloc[-1] - stock2_data['Close'].iloc[0]) / stock2_data['Close'].iloc[0]) * 100
        }

        # Prepare query for Hugging Face API
        query = f"{translate('Compare the performance of', lang)} {stock1} {translate('and', lang)} {stock2} {translate('over a period of', lang)} {time_period}:\n\n"
        query += f"{stock1} {translate('Summary', lang)}:\n{stock1_summary}\n\n"
        query += f"{stock2} {translate('Summary', lang)}:\n{stock2_summary}\n\n"
        query += translate("Provide a detailed analysis and investment recommendation.", lang)

        # Tokenize input
        inputs = tokenizer(query, truncation=True, max_length=1024, return_tensors="pt")
        input_text = tokenizer.decode(inputs["input_ids"][0])

        # Query Hugging Face API
        st.write(f"### {translate('AI Analysis', lang)}")
        with st.spinner(translate("Generating analysis...", lang)):
            response = query_huggingface_api({"inputs": input_text})
            if response:
                st.write(response)

        # Display stock data in tables
        st.subheader(f"{stock1} {translate('Data Summary', lang)}")
        st.table(pd.DataFrame(stock1_summary, index=[translate("Value", lang)]).T)

        st.subheader(f"{stock2} {translate('Data Summary', lang)}")
        st.table(pd.DataFrame(stock2_summary, index=[translate("Value", lang)]).T)

    else:
        st.error(translate("Failed to fetch stock data. Please check the stock symbols and try again.", lang))
