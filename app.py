import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and columns
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# Page Title
st.title("🔧 Predictive Maintenance Dashboard")

# Tabs
tab1, tab2 ,tab3= st.tabs([
    "Prediction",
    "Feature Importance",
    "Dataset Statistics"
])


# Prediction Tab

with tab1:

    st.header("Machine Failure Prediction")

    air_temp = st.number_input(
        "Air Temperature [K]",
        value=300.0
    )

    process_temp = st.number_input(
        "Process Temperature [K]",
        value=310.0
    )

    rot_speed = st.number_input(
        "Rotational Speed [rpm]",
        value=1500
    )

    torque = st.number_input(
        "Torque [Nm]",
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        value=100
    )

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    if st.button("Predict"):

        data = pd.DataFrame({
            "Air temperature [K]": [air_temp],
            "Process temperature [K]": [process_temp],
            "Rotational speed [rpm]": [rot_speed],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear],
            "Type": [machine_type]
        })
        out_of_range=False

        if rot_speed>2886 or rot_speed<1168:
            out_of_range=True
            st.warning("Rotational speed is out of range")

        if torque>76.6 or torque<3.8:
            out_of_range=True
            st.warning("Torque is out of range")
        
        if tool_wear>253:
            out_of_range=True
            st.warning("Tool wear is out of range")
        
        if process_temp>314 or process_temp<305:
            out_of_range=True
            st.warning("Process temprature is out of range")

        if air_temp>305 or air_temp<295:
            out_of_range=True
            st.warning("Air temprature is out of range")

        if out_of_range:
            st.warning("Data is out of range so the prediction is based on extrapolation of model")
            

        data = pd.get_dummies(data)

        data = data.reindex(
            columns=columns,
            fill_value=0
        )

        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0]

        failure_prob = probability[1] * 100

        st.metric(
            "Failure Probability",
            f"{failure_prob:.2f}%"
        )

        if prediction == 1:
            st.error("⚠️ Machine Failure Predicted")
        else:
            st.success("✅ Machine Healthy")

        if failure_prob < 20:
            st.success("🟢 Low Risk")
        elif failure_prob < 50:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")


# Feature Importance Tab

with tab2:

    st.header("Feature Importance Analysis")

    importance_df = pd.read_csv(
        "feature_importance.csv"
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10,6))

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.set_title("Random Forest Feature Importance")

    ax.set_xlabel("Importance Score")

    st.pyplot(fig)

# Dataset Statistics
with tab3:
    df = pd.read_csv("ai4i2020.csv.csv")

    st.header("Dataset Statistics")

    total_samples = len(df)

    total_failures = df["Machine failure"].sum()

    healthy_samples = total_samples - total_failures

    failure_percentage = (
        total_failures / total_samples
    ) * 100

    healthy_percentage = (
        healthy_samples / total_samples
    ) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Samples",
            total_samples
        )

        st.metric(
            "Total Failures",
            int(total_failures)
        )

    with col2:
        st.metric(
            "Healthy Samples",
            int(healthy_samples)
        )

        st.metric(
            "Failure %",
            f"{failure_percentage:.2f}%"
        )

    st.subheader("Failure Distribution")

    chart_df = pd.DataFrame({
        "Category": ["Healthy", "Failure"],
        "Count": [
            healthy_samples,
            total_failures
        ]
    })

    st.bar_chart(
        chart_df.set_index("Category")
    )