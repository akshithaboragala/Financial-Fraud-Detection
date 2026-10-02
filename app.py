import os
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# LOAD DATA AND MODELS

@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "paysim.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path, nrows=100000)
    return None


@st.cache_resource
def load_model():
    return joblib.load(
        os.path.join(BASE_DIR, "xgb_fraud_model.pkl")
    )


@st.cache_data
def load_csv(filename):
    return pd.read_csv(
        os.path.join(BASE_DIR, filename)
    )


@st.cache_data
def load_drift():
    drift_path = os.path.join(BASE_DIR, "adwin_drift_points.pkl")
    if os.path.exists(drift_path):
        return joblib.load(drift_path)
    return []


@st.cache_data
def load_csv(filename):
    return pd.read_csv(
        os.path.join(BASE_DIR, filename)
    )


@st.cache_data
def load_drift():
    drift_path = os.path.join(BASE_DIR, "adwin_drift_points.pkl")
    if os.path.exists(drift_path):
        return joblib.load(drift_path)
    return []


# Load project files
df = load_data()
model = load_model()

shap_df = load_csv("shap_feature_importance.csv")
graph_summary = load_csv("graph_summary.csv")
graph_dashboard_data = load_csv("graph_dashboard_data.csv")
fraud_ring_summary = load_csv("fraud_ring_summary.csv")
top_nodes = load_csv("top_connected_nodes.csv")
top_node_fraud = load_csv(
    "top_connected_node_fraud_analysis.csv"
)

drift_points = load_drift()



# VERIFIED PROJECT RESULTS


TOTAL_TRANSACTIONS = 6362620
FRAUD_TRANSACTIONS = 8213
NORMAL_TRANSACTIONS = 6354407
FRAUD_RATE = 0.1291

PRECISION = 0.9987046632
RECALL = 0.7249647391
F1 = 0.8400980659
PR_AUC = 0.9623661188

TGN_PRECISION = 0.0
TGN_RECALL = 0.0
TGN_F1 = 0.0
TGN_PR_AUC = 0.0001913458


st.sidebar.title("🛡️ Fraud Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🚨 Fraud Detection",
        "📊 Transaction Analytics",
        "🤖 Model Performance",
        "🔍 Explainable AI",
        "🕸️ Graph Analysis",
        "📈 Concept Drift",
        "🧠 TGN Analysis",
        "📥 Download Results"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Real-time hybrid ML system with transaction-network analysis, "
    "fraud-ring detection, explainability, and drift adaptation."
)




if page == "🏠 Home":

    st.title("🛡️ Financial Fraud Detection Dashboard")

    st.write(
        "Financial Fraud Detection and Transaction Pattern Analysis using ML"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Transactions",
        f"{TOTAL_TRANSACTIONS:,}"
    )

    c2.metric(
        "Fraud Transactions",
        f"{FRAUD_TRANSACTIONS:,}"
    )

    c3.metric(
        "Normal Transactions",
        f"{NORMAL_TRANSACTIONS:,}"
    )

    c4.metric(
        "Fraud Rate",
        f"{FRAUD_RATE:.4f}%"
    )

    st.markdown("---")

    st.subheader("Project Approach")

    st.write(
        "Real-time hybrid ML system with transaction-network analysis, "
        "fraud-ring detection, explainability, and drift adaptation."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Core Components")

        st.markdown(
            "- XGBoost fraud classification\n"
            "- Transaction graph construction\n"
            "- Temporal Graph Neural Network (TGN)\n"
            "- Fraud-ring detection\n"
            "- SHAP explainability\n"
            "- ADWIN drift detection"
        )

    with col2:

        st.subheader("Technology Stack")

        st.markdown(
            "- Python\n"
            "- Pandas / NumPy\n"
            "- XGBoost\n"
            "- PyTorch / PyTorch Geometric\n"
            "- NetworkX\n"
            "- SHAP\n"
            "- River ADWIN\n"
            "- Streamlit"
        )


# FRAUD DETECTION


elif page == "🚨 Fraud Detection":

    st.title("🚨 Fraud Detection")

    st.write(
        "Enter transaction details to obtain an XGBoost-based fraud prediction."
    )

    col1, col2 = st.columns(2)

    with col1:

        step = st.number_input(
            "Step",
            min_value=1,
            value=100,
            step=1
        )

        amount = st.number_input(
            "Amount",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        oldbalanceOrg = st.number_input(
            "Old Balance Origin",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

        newbalanceOrig = st.number_input(
            "New Balance Origin",
            min_value=0.0,
            value=4000.0,
            step=100.0
        )

    with col2:

        oldbalanceDest = st.number_input(
            "Old Balance Destination",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        newbalanceDest = st.number_input(
            "New Balance Destination",
            min_value=0.0,
            value=2000.0,
            step=100.0
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "CASH_IN",
                "CASH_OUT",
                "DEBIT",
                "PAYMENT",
                "TRANSFER"
            ]
        )

    if st.button(
        "🔎 Analyze Transaction",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [
                {
                    "step": step,
                    "amount": amount,
                    "oldbalanceOrg": oldbalanceOrg,
                    "newbalanceOrig": newbalanceOrig,
                    "oldbalanceDest": oldbalanceDest,
                    "newbalanceDest": newbalanceDest,
                    "type_CASH_IN": int(
                        transaction_type == "CASH_IN"
                    ),
                    "type_CASH_OUT": int(
                        transaction_type == "CASH_OUT"
                    ),
                    "type_DEBIT": int(
                        transaction_type == "DEBIT"
                    ),
                    "type_PAYMENT": int(
                        transaction_type == "PAYMENT"
                    ),
                    "type_TRANSFER": int(
                        transaction_type == "TRANSFER"
                    )
                }
            ]
        )

        input_data = input_data[
            [
                "step",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "type_CASH_IN",
                "type_CASH_OUT",
                "type_DEBIT",
                "type_PAYMENT",
                "type_TRANSFER"
            ]
        ]

        probability = float(
            model.predict_proba(input_data)[0, 1]
        )

        prediction = int(
            probability >= 0.5
        )

        st.markdown("---")

        if prediction == 1:

            st.error(
                f"🚨 FRAUD DETECTED\n\n"
                f"Fraud Probability: {probability:.2%}"
            )

        else:

            st.success(
                f"✅ NORMAL TRANSACTION\n\n"
                f"Fraud Probability: {probability:.2%}"
            )


 # TRANSACTION ANALYTICS
elif page == "📊 Transaction Analytics":

    st.title("📊 Transaction Analytics")

    if df is None:
        st.warning(
            "Transaction Analytics requires the PaySim dataset "
            "(paysim.csv), which is not included in the deployed repository."
        )
        st.stop()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Transaction Type Distribution")

        st.bar_chart(
            df["type"].value_counts()
        )

    with col2:

        st.subheader("Fraud by Transaction Type")

        fraud_type = (
            df.groupby("type")["isFraud"]
            .sum()
        )

        st.bar_chart(
            fraud_type
        )

    st.subheader("Transaction Amount Trend")

    sample_df = df.sample(
        min(10000, len(df)),
        random_state=42
    )

    st.line_chart(
        sample_df[["amount"]]
        .reset_index(drop=True)
    )


elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Precision",
        f"{PRECISION:.4f}"
    )

    c2.metric(
        "Recall",
        f"{RECALL:.4f}"
    )

    c3.metric(
        "F1 Score",
        f"{F1:.4f}"
    )

    c4.metric(
        "PR-AUC",
        f"{PR_AUC:.4f}"
    )

    st.markdown("---")

    st.subheader("XGBoost Configuration")

    config = pd.DataFrame(
        {
            "Parameter": [
                "Model",
                "Estimators",
                "Max Depth",
                "Learning Rate",
                "Evaluation Metric"
            ],
            "Value": [
                "XGBClassifier",
                100,
                6,
                0.1,
                "logloss"
            ]
        }
    )

    st.dataframe(
        config,
        use_container_width=True
    )

    st.subheader("Performance Metrics")

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Precision",
                "Recall",
                "F1 Score",
                "PR-AUC"
            ],
            "Value": [
                PRECISION,
                RECALL,
                F1,
                PR_AUC
            ]
        }
    )

    st.dataframe(
        metrics_df,
        use_container_width=True
    )


elif page == "🔍 Explainable AI":

    st.title("🔍 Explainable AI")

    st.write(
        "SHAP-based feature importance for the XGBoost fraud detection model."
    )

    st.subheader("Feature Importance")

    if shap_df.shape[1] >= 2:

        display_shap = shap_df.copy().iloc[:, :2]

        display_shap.columns = [
            "Feature",
            "Mean Absolute SHAP Value"
        ]

        display_shap = display_shap.set_index(
            "Feature"
        )

        st.bar_chart(
            display_shap
        )

    st.subheader("Top Features")

    st.dataframe(
        shap_df,
        use_container_width=True
    )


elif page == "🕸️ Graph Analysis":

    st.title("🕸️ Transaction Graph Analysis")

    graph_dict = dict(
        zip(
            graph_dashboard_data["Metric"].astype(str),
            graph_dashboard_data["Value"]
        )
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Nodes",
        f"{int(float(graph_dict['Total Nodes'])):,}"
    )

    c2.metric(
        "Total Edges",
        f"{int(float(graph_dict['Total Edges'])):,}"
    )

    c3.metric(
        "Communities",
        f"{int(float(graph_dict['Total Communities'])):,}"
    )

    c4.metric(
        "Fraud Communities",
        f"{int(float(graph_dict['Fraud Communities'])):,}"
    )

    st.markdown("---")

    st.subheader("Fraud-Ring Summary")

    st.dataframe(
        fraud_ring_summary.head(10),
        use_container_width=True
    )

    st.subheader("Top Connected Nodes")

    st.dataframe(
        top_nodes.head(10),
        use_container_width=True
    )

    st.subheader("Top Connected Node Fraud Analysis")

    st.dataframe(
        top_node_fraud.head(10),
        use_container_width=True
    )


elif page == "📈 Concept Drift":

    st.title("📈 Concept Drift Monitoring")

    st.write(
        "ADWIN was applied to the XGBoost prediction-probability "
        "stream to identify changes in prediction behavior."
    )

    st.metric(
        "Detected Drift Points",
        len(drift_points)
    )

    drift_df = pd.DataFrame(
        {
            "Transaction Index": drift_points
        }
    )

    st.dataframe(
        drift_df,
        use_container_width=True
    )


elif page == "🧠 TGN Analysis":

    st.title("🧠 Temporal Graph Neural Network Analysis")

    st.info(
        "TGN was implemented for temporal transaction-graph analysis. "
        "The current evaluation uses the first 100,000 transactions "
        "and is intended as a prototype-level temporal graph analysis."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Graph Transactions",
        "100,000"
    )

    c2.metric(
        "Graph Nodes",
        "151,551"
    )

    c3.metric(
        "TGN Precision",
        f"{TGN_PRECISION:.4f}"
    )

    c4.metric(
        "TGN PR-AUC",
        f"{TGN_PR_AUC:.6f}"
    )

    tgn_results = pd.DataFrame(
        {
            "Metric": [
                "Precision",
                "Recall",
                "F1 Score",
                "PR-AUC"
            ],
            "Value": [
                TGN_PRECISION,
                TGN_RECALL,
                TGN_F1,
                TGN_PR_AUC
            ]
        }
    )

    st.dataframe(
        tgn_results,
        use_container_width=True
    )


elif page == "📥 Download Results":

    st.title("📥 Download Project Results")

    download_files = [
        "graph_summary.csv",
        "fraud_ring_summary.csv",
        "graph_dashboard_data.csv",
        "shap_feature_importance.csv",
        "top_connected_nodes.csv",
        "top_connected_node_fraud_analysis.csv",
        "adwin_drift_points.pkl",
        "xgb_fraud_model.pkl"
    ]

    for filename in download_files:

        file_path = os.path.join(
            BASE_DIR,
            filename
        )

        if os.path.exists(file_path):

            with open(
                file_path,
                "rb"
            ) as file:

                st.download_button(
                    label=f"⬇️ Download {filename}",
                    data=file.read(),
                    file_name=filename,
                    use_container_width=True
                )

        else:

            st.warning(
                f"{filename} not found."
            )


st.markdown("---")

st.caption(
    "Financial Fraud Detection and Transaction Pattern Analysis using ML"
)



        
    
        
