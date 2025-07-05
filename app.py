import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(page_title="ML Model Explorer", layout="wide")
st.title("🧠 ML Model Explorer App")

st.markdown("""
Upload a dataset, select a machine learning model, and visualize the results with ease. This app supports both classification and regression.
""")

# Sidebar
st.sidebar.header("🔧 Configuration")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # Select target column
    target_column = st.sidebar.selectbox("🎯 Select target column", df.columns)
    task_type = "Classification" if df[target_column].dtype == "object" or len(df[target_column].unique()) <= 10 else "Regression"
    st.sidebar.markdown(f"Detected Task: **{task_type}**")

    # Select features
    features = st.sidebar.multiselect("🧬 Select features (optional - default: all except target)", [col for col in df.columns if col != target_column])
    if not features:
        features = [col for col in df.columns if col != target_column]

    X = df[features]
    y = df[target_column]

    # Encode categorical features in X
    X = pd.get_dummies(X)

    # Encode target if it's categorical (for classification)
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split data
    test_size = st.sidebar.slider("📊 Test size (%)", min_value=10, max_value=50, value=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    # Model selection
    model_name = st.sidebar.selectbox("🧠 Select model", ["Random Forest", "XGBoost", "Logistic/Linear Regression"])

    if task_type == "Classification":
        if model_name == "Random Forest":
            n_estimators = st.sidebar.slider("🌲 n_estimators", 10, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators)
        elif model_name == "XGBoost":
            model = xgb.XGBClassifier(eval_metric="logloss")
        else:
            model = LogisticRegression(max_iter=1000)
    else:
        if model_name == "Random Forest":
            n_estimators = st.sidebar.slider("🌲 n_estimators", 10, 200, 100)
            model = RandomForestRegressor(n_estimators=n_estimators)
        elif model_name == "XGBoost":
            model = xgb.XGBRegressor()
        else:
            model = LinearRegression()

    if st.sidebar.button("🚀 Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "Classification":
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            st.subheader("📊 Classification Metrics")
            st.write(f"**Accuracy:** {acc:.2f}")
            st.write(f"**F1 Score:** {f1_score(y_test, y_pred, average='weighted'):.2f}")
            st.write(f"**Precision:** {precision_score(y_test, y_pred, average='weighted'):.2f}")
            st.write(f"**Recall:** {recall_score(y_test, y_pred, average='weighted'):.2f}")

            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # ROC Curve for binary classification
            if len(np.unique(y_test)) == 2:
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                st.write("**ROC Curve:**")
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                ax2.plot([0, 1], [0, 1], linestyle='--')
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.legend()
                st.pyplot(fig2)

        else:
            st.subheader("📈 Regression Metrics")
            st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
            st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

            # Plot predicted vs actual
            fig3 = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
            st.plotly_chart(fig3)

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
            feat_df = feat_df.sort_values(by="Importance", ascending=False).head(20)  # Show top 20 only
            st.subheader("🔍 Top 20 Feature Importances")
            fig4 = px.bar(feat_df, x='Feature', y='Importance', title="Feature Importance")
            fig4.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig4)
