import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO


st.set_page_config(
    page_title="IPOPulse-AI Global",
    page_icon="📈",
    layout="wide"
)

st.title("IPOPulse-AI Global")
st.subheader("AI-Enabled Decision-Support Framework for IPO Performance Prediction")

st.caption(
    "Prototype using Yahoo Finance/yfinance, Python, technical indicators, "
    "KNN baseline, Random Forest enhancement, benchmarking, and IPOPulse Score."
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

    master["MA5"] = (
        master.groupby("Ticker")["Close"]
        .transform(lambda x: x.rolling(5).mean())
    )

    master["MA20"] = (
        master.groupby("Ticker")["Close"]
        .transform(lambda x: x.rolling(20).mean())
    )

    master["Momentum"] = (
        master.groupby("Ticker")["Close"]
        .pct_change(5)
    )

    master["Price_Trend"] = master["MA5"] / master["MA20"]

    delta = master.groupby("Ticker")["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = (
        gain.groupby(master["Ticker"])
        .transform(lambda x: x.rolling(14).mean())
    )

    avg_loss = (
        loss.groupby(master["Ticker"])
        .transform(lambda x: x.rolling(14).mean())
    )

    rs = avg_gain / avg_loss
    master["RSI"] = 100 - (100 / (1 + rs))

    ema12 = (
        master.groupby("Ticker")["Close"]
        .transform(lambda x: x.ewm(span=12, adjust=False).mean())
    )

    ema26 = (
        master.groupby("Ticker")["Close"]
        .transform(lambda x: x.ewm(span=26, adjust=False).mean())
    )

    master["MACD"] = ema12 - ema26

    master["Target"] = np.where(
        master.groupby("Ticker")["Return"].shift(-1) > 0,
        1,
        0
    )

    master = master.replace([np.inf, -np.inf], np.nan).dropna()

    return master


FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "Volume_Surge",
    "Volatility_5D",
    "MA5",
    "MA20",
    "Momentum",
    "Price_Trend",
    "RSI",
    "MACD"
]


def train_models(master):
    X = master[FEATURE_COLUMNS]
    y = master["Target"]

    # ========================================================
    # 1. 70% TRAINING / 30% INDEPENDENT TESTING
    # ========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    # ========================================================
    # 2. 10-FOLD STRATIFIED CROSS-VALIDATION
    # ========================================================
    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    # ========================================================
    # 3. KNN HYPERPARAMETER TUNING
    # ========================================================
    knn = KNeighborsClassifier()

    knn_param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11]
    }

    knn_grid = GridSearchCV(
        estimator=knn,
        param_grid=knn_param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=1
    )

    knn_grid.fit(X_train, y_train)
    best_knn = knn_grid.best_estimator_

    knn_pred = best_knn.predict(X_test)
    knn_prob = best_knn.predict_proba(X_test)[:, 1]

    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_precision = precision_score(y_test, knn_pred, zero_division=0)
    knn_recall = recall_score(y_test, knn_pred, zero_division=0)
    knn_f1 = f1_score(y_test, knn_pred, zero_division=0)
    knn_auc = roc_auc_score(y_test, knn_prob)
    knn_cv_accuracy = knn_grid.best_score_

    # ========================================================
    # 4. RANDOM FOREST HYPERPARAMETER TUNING
    # ========================================================
    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    rf_param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [8, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    rf_grid = GridSearchCV(
        estimator=rf,
        param_grid=rf_param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=1
    )

    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    rf_pred = best_rf.predict(X_test)
    rf_prob = best_rf.predict_proba(X_test)[:, 1]

    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred, zero_division=0)
    rf_recall = recall_score(y_test, rf_pred, zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
    rf_auc = roc_auc_score(y_test, rf_prob)
    rf_cv_accuracy = rf_grid.best_score_

    # ========================================================
    # 5. MODEL PERFORMANCE TABLE
    # ========================================================
    model_results = pd.DataFrame({
        "Model": ["KNN Baseline", "Random Forest Enhanced"],
        "Accuracy (%)": [
            round(knn_accuracy * 100, 2),
            round(rf_accuracy * 100, 2)
        ],
        "Precision": [
            round(knn_precision, 4),
            round(rf_precision, 4)
        ],
        "Recall": [
            round(knn_recall, 4),
            round(rf_recall, 4)
        ],
        "F1-Score": [
            round(knn_f1, 4),
            round(rf_f1, 4)
        ],
        "ROC-AUC": [
            round(knn_auc, 4),
            round(rf_auc, 4)
        ],
        "10-Fold CV Accuracy (%)": [
            round(knn_cv_accuracy * 100, 2),
            round(rf_cv_accuracy * 100, 2)
        ]
    })

    # ========================================================
    # 6. DISPLAY HYPERPARAMETER RESULTS
    # ========================================================
    st.subheader("Model Validation and Hyperparameter Tuning")
    st.write("Best KNN parameters:", knn_grid.best_params_)
    st.write("Best Random Forest parameters:", rf_grid.best_params_)

    # ========================================================
    # 7. SELECT RANDOM FOREST FOR IPOPULSE-AI
    # ========================================================
    selected_model = best_rf
    selected_accuracy = rf_accuracy

    return selected_model, selected_accuracy, model_results, best_rf


def create_report(master, model):
    latest_rows = master.sort_values("Date").groupby("Ticker").tail(1).copy()

    features = latest_rows[FEATURE_COLUMNS]

    latest_rows["Prediction_Label"] = np.where(
        model.predict(features) == 1,
        "Positive",
        "Negative"
    )

    latest_rows["IPOPulse_Score"] = (
        latest_rows["Return"].rank(pct=True) * 25 +
        latest_rows["Volume_Surge"].rank(pct=True) * 20 +
        latest_rows["Volume"].rank(pct=True) * 15 +
        (1 - latest_rows["Volatility_5D"].rank(pct=True)) * 15 +
        latest_rows["Momentum"].rank(pct=True) * 10 +
        latest_rows["Price_Trend"].rank(pct=True) * 10 +
        (1 - abs(latest_rows["RSI"] - 50).rank(pct=True)) * 5
    )

    latest_rows["Risk_Level"] = pd.cut(
        latest_rows["IPOPulse_Score"],
        bins=[0, 40, 70, 100],
        labels=["High Risk", "Moderate", "Attractive"],
        include_lowest=True
    )

    latest_rows["AI_Assessment"] = np.where(
        (latest_rows["IPOPulse_Score"] >= 70)
        & (latest_rows["Prediction_Label"] == "Positive"),
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
            "Volatility_5D": "5-Day Volatility",
            "MA5": "5-Day Moving Average",
            "MA20": "20-Day Moving Average",
            "RSI": "RSI",
            "MACD": "MACD",
            "Momentum": "Momentum",
            "Price_Trend": "Price Trend"
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
            "5-Day Moving Average",
            "20-Day Moving Average",
            "RSI",
            "MACD",
            "Momentum",
            "Price Trend",
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
    ["6mo", "1y", "2y", "5y"],
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

    model, accuracy, model_results, rf_model = train_models(feature_data)

    report_final = create_report(feature_data, model)

    st.success("IPOPulse-AI analysis completed.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Tickers analyzed", len(selected_tickers))
    col2.metric("Selected model accuracy", f"{accuracy * 100:.2f}%")
    col3.metric("Countries", len(countries))

    st.subheader("Model Performance Comparison")
    st.dataframe(model_results, use_container_width=True)

    expected_accuracy = pd.DataFrame(
        {
            "Model": [
                "KNN",
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "Neural Network"
            ],
            "Expected Accuracy Range": [
                "50-60%",
                "55-70%",
                "60-75%",
                "60-75%",
                "55-70%"
            ],
            "Prototype Status": [
                "Implemented",
                "Implemented",
                "Future Enhancement",
                "Future Enhancement",
                "Future Enhancement"
            ]
        }
    )

    st.subheader("Model Roadmap")
    st.dataframe(expected_accuracy, use_container_width=True)

    feature_importance = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": rf_model.feature_importances_
        }
    ).sort_values("Importance", ascending=False)

    st.subheader("Random Forest Feature Importance")
    st.dataframe(feature_importance, use_container_width=True)
    st.bar_chart(feature_importance.set_index("Feature")["Importance"])

    st.subheader("IPOPulse-AI Global Assessment Report")
    st.dataframe(report_final, use_container_width=True)

    st.subheader("Top Ranked Firms")
    st.dataframe(report_final.head(5), use_container_width=True)

    st.subheader("IPOPulse Score by Company")
    st.bar_chart(report_final.set_index("Company")["IPOPulse_Score"])

    st.subheader("Daily Return by Company")
    st.bar_chart(report_final.set_index("Company")["Daily Return"])

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
