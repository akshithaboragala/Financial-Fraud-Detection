import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e6e9ef;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    text-align: center;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(
        os.path.join(BASE_DIR, "paysim.csv"),
        nrows=100000
    )

@st.cache_resource
def load_model():
    return joblib.load(
        os.path.join(BASE_DIR, "xgb_fraud_model.pkl")
    )

@st.cache_data
def load_shap():
    return pd.read_csv(
        os.path.join(BASE_DIR, "shap_feature_importance.csv")
    )

@st.cache_data
def load_graph_summary():
    return pd.read_csv(
        os.path.join(BASE_DIR, "graph_summary.csv")
    )

@st.cache_data
def load_fraud_rings():
    return pd.read_csv(
        os.path.join(BASE_DIR, "fraud_ring_summary.csv")
    )

@st.cache_data
def load_top_nodes():
    return pd.read_csv(
        os.path.join(BASE_DIR, "top_connected_nodes.csv")
    )

@st.cache_resource
def load_drift():
    return joblib.load(
        os.path.join(BASE_DIR, "adwin_drift_points.pkl")
    )


# Load everything

df = load_dataset()
model = load_model()
shap_df = load_shap()
graph_summary = load_graph_summary()
fraud_ring_summary = load_fraud_rings()
top_nodes = load_top_nodes()
drift_points = load_drift()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛡️ Fraud Detection")

st.sidebar.caption(
    "Financial Fraud Detection and Transaction Pattern Analysis"
)

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

# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title("🛡️ Financial Fraud Detection Dashboard")

    st.write(
        "AI-based financial fraud detection and transaction pattern "
        "analysis using Machine Learning and transaction-network analysis."
    )

    total_transactions = 6362620
    fraud_transactions = 8213
    normal_transactions = total_transactions - fraud_transactions
    fraud_rate = fraud_transactions / total_transactions * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    c2.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )

    c3.metric(
        "Normal Transactions",
        f"{normal_transactions:,}"
    )

    c4.metric(
        "Fraud Rate",
        f"{fraud_rate:.4f}%"
    )

    st.markdown("---")

    st.subheader("Project Approach")

    st.markdown("""
    **Real-time hybrid ML system with transaction-network analysis,
    fraud-ring detection, explainability, and drift adaptation.**
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Core Components")

        st.markdown("""
        - XGBoost fraud classification
        - Transaction graph construction
        - Temporal Graph Neural Network (TGN)
        - Fraud-ring detection
        - SHAP explainability
        - ADWIN drift detection
        """)

    with col2:

        st.subheader("Technology Stack")

        st.markdown("""
        - Python
        - Pandas / NumPy
        - XGBoost
        - PyTorch / PyTorch Geometric
        - NetworkX
        - SHAP
        - River ADWIN
        - Streamlit
        """)

# =========================================================
# FRAUD DETECTION
# =========================================================

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
            value=100
        )

        amount = st.number_input(
            "Amount",
            min_value=0.0,
            value=1000.0
        )

        oldbalanceOrg = st.number_input(
            "Old Balance Origin",
            min_value=0.0,
            value=5000.0
        )

        newbalanceOrig = st.number_input(
            "New Balance Origin",
            min_value=0.0,
            value=4000.0
        )

    with col2:

        oldbalanceDest = st.number_input(
            "Old Balance Destination",
            min_value=0.0,
            value=1000.0
        )

        newbalanceDest = st.number_input(
            "New Balance Destination",
            min_value=0.0,
            value=2000.0
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

        input_data = pd.DataFrame([{
            "step": step,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "type_CASH_IN": int(transaction_type == "CASH_IN"),
            "type_CASH_OUT": int(transaction_type == "CASH_OUT"),
            "type_DEBIT": int(transaction_type == "DEBIT"),
            "type_PAYMENT": int(transaction_type == "PAYMENT"),
            "type_TRANSFER": int(transaction_type == "TRANSFER")
        }])

        probability = model.predict_proba(
            input_data
        )[0, 1]

        prediction = int(probability >= 0.5)

        st.markdown("---")

        if prediction == 1:

            st.error(
                f"🚨 FRAUD DETECTED — "
                f"Fraud Probability: {probability:.2%}"
            )

        else:

            st.success(
                f"✅ NORMAL TRANSACTION — "
                f"Fraud Probability: {probability:.2%}"
            )

# =========================================================
# TRANSACTION ANALYTICS
# =========================================================

elif page == "📊 Transaction Analytics":

    st.title("📊 Transaction Analytics")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Transaction Type Distribution")

        type_counts = df["type"].value_counts()

        st.bar_chart(type_counts)

    with col2:

        st.subheader("Fraud by Transaction Type")

        fraud_type = df.groupby(
            "type"
        )["isFraud"].sum()

        st.bar_chart(fraud_type)

    st.subheader("Transaction Amount Distribution")

    sample_df = df.sample(
        min(10000, len(df)),
        random_state=42
    )

    st.line_chart(
        sample_df["amount"].reset_index(drop=True)
    )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    precision = 0.9987046632
    recall = 0.7249647391
    f1 = 0.8400980659
    pr_auc = 0.9623661188

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Precision", f"{precision:.4f}")
    c2.metric("Recall", f"{recall:.4f}")
    c3.metric("F1 Score", f"{f1:.4f}")
    c4.metric("PR-AUC", f"{pr_auc:.4f}")

    st.markdown("---")

    st.subheader("XGBoost Configuration")

    config = pd.DataFrame({
        "Parameter": [
            "Model",
            "Estimators",
            "Max Depth",
            "Learning Rate"
        ],
        "Value": [
            "XGBClassifier",
            100,
            6,
            0.1
        ]
    })

    st.dataframe(
        config,
        use_container_width=True
    )

# =========================================================
# EXPLAINABLE AI
# =========================================================

elif page == "🔍 Explainable AI":

    st.title("🔍 Explainable AI")

    st.write(
        "SHAP identifies the features that contribute most "
        "to the XGBoost fraud prediction."
    )

    st.subheader("Feature Importance")

    chart_df = shap_df.set_index("Feature")

    st.bar_chart(
        chart_df["Mean_Absolute_SHAP"]
    )

    st.subheader("SHAP Feature Importance Table")

    st.dataframe(
        shap_df,
        use_container_width=True
    )

# =========================================================
# GRAPH ANALYSIS
# =========================================================

elif page == "🕸️ Graph Analysis":

    st.title("🕸️ Transaction Graph Analysis")

    metrics = graph_summary.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Nodes",
        f"{int(metrics['total_nodes']):,}"
    )

    c2.metric(
        "Edges",
        f"{int(metrics['total_edges']):,}"
    )

    c3.metric(
        "Communities",
        f"{int(metrics['total_communities']):,}"
    )

    c4.metric(
        "Fraud Communities",
        f"{int(metrics['fraud_containing_communities']):,}"
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Fraud Transactions",
        f"{int(metrics['fraud_transactions']):,}"
    )

    c6.metric(
        "Fraud Related Nodes",
        f"{int(metrics['fraud_related_nodes']):,}"
    )

    c7.metric(
        "Largest Community",
        f"{int(metrics['largest_community_size']):,}"
    )

    c8.metric(
        "Maximum Degree",
        f"{int(metrics['maximum_node_degree']):,}"
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

# =========================================================
# CONCEPT DRIFT
# =========================================================

elif page == "📈 Concept Drift":

    st.title("📈 Concept Drift Monitoring")

    st.write(
        "ADWIN was applied to the XGBoost prediction-probability "
        "stream to identify points where the prediction distribution changed."
    )

    st.metric(
        "Detected Drift Points",
        len(drift_points)
    )

    drift_df = pd.DataFrame({
        "Transaction Index": drift_points
    })

    st.subheader("Detected Drift Points")

    st.dataframe(
        drift_df,
        use_container_width=True
    )

# =========================================================
# TGN ANALYSIS
# =========================================================

elif page == "🧠 TGN Analysis":

    st.title("🧠 Temporal Graph Neural Network Analysis")

    st.info(
        "TGN was implemented for temporal transaction-graph analysis. "
        "The current evaluation is based on the first 100,000 transactions "
        "and should be interpreted as a prototype-level temporal graph analysis."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Graph Transactions", "100,000")
    c2.metric("Graph Nodes", "151,551")
    c3.metric("TGN Precision", "0.0000")
    c4.metric("TGN PR-AUC", "0.000191")

    st.subheader("TGN Evaluation")

    tgn_results = pd.DataFrame({
        "Metric": [
            "Precision",
            "Recall",
            "F1 Score",
            "PR-AUC"
        ],
        "Value": [
            0.0,
            0.0,
            0.0,
            0.0001913458
        ]
    })

    st.dataframe(
        tgn_results,
        use_container_width=True
    )

# =========================================================
# DOWNLOAD RESULTS
# =========================================================

elif page == "📥 Download Results":

    st.title("📥 Download Project Results")

    download_files = [
        "graph_summary.csv",
        "fraud_ring_summary.csv",
        "graph_dashboard_data.csv",
        "shap_feature_importance.csv",
        "top_connected_nodes.csv",
        "top_connected_node_fraud_analysis.csv"
    ]

    for file in download_files:

        file_path = os.path.join(
            BASE_DIR,
            file
        )

        if os.path.exists(file_path):

            with open(file_path, "rb") as f:

                st.download_button(
                    label=f"⬇️ Download {file}",
                    data=f,
                    file_name=file,
                    use_container_width=True
                )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Financial Fraud Detection and Transaction Pattern Analysis using ML"
)