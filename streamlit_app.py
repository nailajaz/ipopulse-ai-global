import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from io import BytesIO


st.set_page_config(
    page_title="IPOPulse-AI Global",
    page_icon="📈",
    layout="wide"
)

st.title("IPOPulse-AI Global")
st.subheader("AI-Enabled Decision-Support Framework for IPO Performance Prediction")

st.caption(
    "Prototype using Yahoo Finance/yfinance, Python, KNN, benchmarking, and IPOPulse Score."
)

st.warning(
    "This prototype is for academic research and decision-support demonstration only. "
    "It does not provide financial or investment advice."
)


global_tickers = {
    "Canada": ["TD.TO", "RY.TO", "BNS.TO", "SHOP.TO", "WELL.TO", "LSPD.TO"],
    "USA": ["AAPL", "MSFT", "ABNB", "COIN", "RDDT"],
    "India": ["ZOMATO.NS", "PAYTM.NS", "NYKAA.NS"],
    "China": ["BABA", "JD", "PDD"]
}

country_map = {
    ticker: country
    for country, tickers in global_tickers.items()
    for ticker in tickers
}

currency_map = {
    "Canada": "CAD",
    "USA": "USD",
    "India": "INR",
    "China": "USD"
}

market_type_map = {
    "Canada": "Developed Market",
    "USA": "Developed Market",
    "India": "Emerging Market",
    "China": "Emerging Market"
}

category_map = {
    "TD.TO": "Large Cap",
    "RY.TO": "Large Cap",
    "BNS.TO": "Large Cap",
    "SHOP.TO": "Large Cap",
    "WELL.TO": "Small/Mid Cap",
    "LSPD.TO": "Small/Mid Cap",

    "AAPL": "Large Cap",
    "MSFT": "Large Cap",
    "ABNB": "Recent IPO / New Listing",
    "COIN": "Recent IPO / New Listing",
    "RDDT": "Recent IPO / New Listing",

    "ZOMATO.NS": "Recent IPO / New Listing",
    "PAYTM.NS": "Recent IPO / New Listing",
    "NYKAA.NS": "Recent IPO / New Listing",

    "BABA": "Large Cap",
    "JD": "Large Cap",
    "PDD": "Large Cap"
}

company_details = {
    "TD.TO": ["Toronto-Dominion Bank", "Financial Services", "TSX"],
    "RY.TO": ["Royal Bank of Canada", "Financial Services", "TSX"],
    "BNS.TO": ["Bank of Nova Scotia", "Financial Services", "TSX"],
    "SHOP.TO": ["Shopify Inc.", "Technology", "TSX"],
    "WELL.TO": ["WELL Health Technologies Corp.", "Healthcare Technology", "TSX"],
    "LSPD.TO": ["Lightspeed Commerce Inc.", "Technology", "TSX"],

    "AAPL": ["Apple Inc.", "Technology", "NASDAQ"],
    "MSFT": ["Microsoft Corporation", "Technology", "NASDAQ"],
    "ABNB": ["Airbnb Inc.", "Travel / Technology", "NASDAQ"],
    "COIN": ["Coinbase Global Inc.", "FinTech / Digital Assets", "NASDAQ"],
    "RDDT": ["Reddit Inc.", "Social Media / Technology", "NYSE"],

    "ZOMATO.NS": ["Zomato Ltd.", "Food Delivery / Technology", "NSE India"],
    "PAYTM.NS": ["One 97 Communications Ltd. / Paytm", "FinTech", "NSE India"],
    "NYKAA.NS": ["FSN E-Commerce Ventures Ltd. / Nykaa", "E-Commerce / Beauty Retail", "NSE India"],

    "BABA": ["Alibaba Group Holding Ltd.", "E-Commerce / Technology", "NYSE"],
    "JD": ["JD.com Inc.", "E-Commerce / Technology", "NASDAQ"],
    "PDD": ["PDD Holdings Inc.", "E-Commerce / Technology", "NASDAQ"]
}


@st.cache_data(ttl=3600)
def load_market_data(selected_tickers, period):
    results = []

    for ticker in selected_tickers:
        try:
            df = yf.download(
                ticker,
                period=period,
                auto_adjust=True,
                progress=False
            )

            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df["Ticker"] = ticker
            df["Country"] = country_map.get(ticker, "Not available")
            df["Currency"] = currency_map.get(df["Country"].iloc[0], "Not available")
            df["Market_Type"] = market_type_map.get(df["Country"].iloc[0], "Not available")
            df["Category"] = category_map.get(ticker, "Not Classified")

            results.append(df)

        except Exception as e:
            st.error(f"Error loading {ticker}: {e}")

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def create_features(master):
    master = master.sort_values(["Ticker", "Date"]).copy()

    master["Return"] = master.groupby("Ticker")["Close"].pct_change()

    master["Volume_Avg_5D"] = (
        master.groupby("Ticker")["Volume"]
        .transform(lambda x: x.rolling(5).mean())
    )

    master["Volume_Surge"] = master["Volume"] / master["Volume_Avg_5D"]

    master["Volatility_5D"] = (
        master.groupby("Ticker")["Return"]
        .transform(lambda x: x.rolling(5).std())
    )

    master["Target"] = np.where(
        master.groupby("Ticker")["Return"].shift(-1) > 0,
        1,
        0
    )

    master = master.replace([np.inf, -np.inf], np.nan).dropna()

    return master


def train_model(master):
    X = master[["Close", "Volume", "Volume_Surge", "Volatility_5D"]]
    y = master["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    return model, accuracy


def create_report(master, model):
    latest_rows = master.sort_values("Date").groupby("Ticker").tail(1).copy()

    features = latest_rows[["Close", "Volume", "Volume_Surge", "Volatility_5D"]]

    latest_rows["Prediction_Label"] = np.where(
        model.predict(features) == 1,
        "Positive",
        "Negative"
    )

    latest_rows["IPOPulse_Score"] = (
        latest_rows["Return"].rank(pct=True) * 35 +
        latest_rows["Volume_Surge"].rank(pct=True) * 30 +
        latest_rows["Volume"].rank(pct=True) * 20 +
        (1 - latest_rows["Volatility_5D"].rank(pct=True)) * 15
    )

    latest_rows["Risk_Level"] = pd.cut(
        latest_rows["IPOPulse_Score"],
        bins=[0, 40, 70, 100],
        labels=["High Risk", "Moderate", "Attractive"],
        include_lowest=True
    )

    latest_rows["AI_Assessment"] = np.where(
        (latest_rows["IPOPulse_Score"] >= 70) &
        (latest_rows["Prediction_Label"] == "Positive"),
        "Attractive for further review based on model indicators.",
        np.where(
            latest_rows["IPOPulse_Score"] >= 40,
            "Moderate signal; requires additional financial and sector review.",
            "High-risk signal; caution is required before further consideration."
        )
    )

    report_final = latest_rows.copy()

    report_final["Company"] = report_final["Ticker"].map(
        lambda x: company_details.get(x, [x, "Not available", "Not available"])[0]
    )

    report_final["Sector"] = report_final["Ticker"].map(
        lambda x: company_details.get(x, [x, "Not available", "Not available"])[1]
    )

    report_final["Exchange"] = report_final["Ticker"].map(
        lambda x: company_details.get(x, [x, "Not available", "Not available"])[2]
    )

    report_final = report_final.rename(
        columns={
            "Close": "Close Price",
            "Return": "Daily Return",
            "Volume_Surge": "Volume Surge",
            "Volatility_5D": "5-Day Volatility"
        }
    )

    report_final = report_final[
        [
            "Date",
            "Company",
            "Ticker",
            "Country",
            "Market_Type",
            "Category",
            "Sector",
            "Exchange",
            "Currency",
            "Close Price",
            "Daily Return",
            "Volume",
            "Volume Surge",
            "5-Day Volatility",
            "Prediction_Label",
            "IPOPulse_Score",
            "Risk_Level",
            "AI_Assessment"
        ]
    ].sort_values("IPOPulse_Score", ascending=False)

    return report_final


def convert_df_to_excel(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="IPOPulse Report")

    return output.getvalue()


st.sidebar.header("Prototype Settings")

countries = st.sidebar.multiselect(
    "Select countries",
    list(global_tickers.keys()),
    default=list(global_tickers.keys())
)

period = st.sidebar.selectbox(
    "Select data period",
    ["3mo", "6mo", "1y", "2y"],
    index=2
)

selected_tickers = []

for country in countries:
    selected_tickers.extend(global_tickers[country])

st.sidebar.write("Selected tickers:")
st.sidebar.write(selected_tickers)


if st.button("Run IPOPulse-AI Global Analysis"):

    raw_data = load_market_data(selected_tickers, period)

    if raw_data.empty:
        st.error("No data loaded. Please check tickers or internet connection.")
        st.stop()

    feature_data = create_features(raw_data)

    if feature_data.empty:
        st.error("Feature engineering failed. Not enough data available.")
        st.stop()

    model, accuracy = train_model(feature_data)

    report_final = create_report(feature_data, model)

    st.success("IPOPulse-AI analysis completed.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Tickers analyzed", len(selected_tickers))
    col2.metric("Model accuracy", f"{accuracy * 100:.2f}%")
    col3.metric("Countries", len(countries))

    st.subheader("IPOPulse-AI Global Assessment Report")
    st.dataframe(report_final, use_container_width=True)

    st.subheader("Top Ranked Firms")
    st.dataframe(report_final.head(5), use_container_width=True)

    st.subheader("IPOPulse Score by Company")
    st.bar_chart(
        report_final.set_index("Company")["IPOPulse_Score"]
    )

    st.subheader("Daily Return by Company")
    st.bar_chart(
        report_final.set_index("Company")["Daily Return"]
    )

    csv = report_final.to_csv(index=False).encode("utf-8")
    excel = convert_df_to_excel(report_final)

    st.download_button(
        label="Download CSV Report",
        data=csv,
        file_name="IPOPulse_AI_Global_Assessment_Report.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Excel Report",
        data=excel,
        file_name="IPOPulse_AI_Global_Assessment_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Select countries from the sidebar and click the button to run the prototype.")
