import streamlit as st
import pandas as pd
import joblib

model = joblib.load("accident_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(
    page_title="SafeRoad AI",
    page_icon="🚗",
    layout="centered"
)

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🚗 SafeRoad AI")
st.sidebar.image(
    "https://tse2.mm.bing.net/th/id/OIP.UYUJI3uy5b_zP0Sr259e3QHaEP?r=0&rs=1&pid=ImgDetMain&o=7&rm=3",
    use_container_width=True
)
st.sidebar.write(
        "Welcome to SafeRoad AI! Use the sidebar to navigate through "
        "sections, learn why this app was built, and try out the "
        "accident severity predictor."
    )

page = st.sidebar.selectbox(
    "Select a Section",
    [ "About The App", "Prediction"]
)

st.sidebar.divider()
st.sidebar.caption("Built as a demo project for road safety awareness.")

# ----------------  ----------------
if page == "About The App":
    st.title("🚗 SafeRoad AI")
    st.subheader("Road Accident Severity Predictor")

    st.image(
        "https://tse4.mm.bing.net/th/id/OIP.-5qiUzo1xU5LFvcOsVEUgwHaEK?r=0&rs=1&pid=ImgDetMain&o=7&rm=3",
        use_container_width=True
    )
    st.write(
        "Road accidents remain one of the leading causes of death and injury worldwide, but the outcome rarely"
        "depends on speed or vehicle count alone. Weather, road surface, and lighting conditions all affect how well"
        "a driver can see and control their vehicle. Road type, vehicle type, and junction details add further risk,"
        "especially at intersections. Even the day of the week and whether the area is urban or rural can influence"
        "accident patterns. Together,these factors show that road safety is shaped by many conditions working together, not just one."
    )

    st.write(
        "SafeRoad AI uses a machine learning model trained on real "
        "accident records to estimate how severe an accident might be, "
        "based on the conditions you provide. The goal is to raise "
        "awareness of how these factors combine to affect outcomes — "
        "not to replace real safety assessments."
    )

# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":
    st.title("🚗 SafeRoad AI")
    st.subheader("Road Accident Severity Predictor")

    st.write(
        "Fill in the details below to see how factors like speed, "
        "weather, and road conditions can influence accident severity."
    )
    st.divider()

    FEATURES = [
        "Speed_limit",
        "Weather_Conditions",
        "Road_Surface_Conditions",
        "Light_Conditions",
        "Road_Type",
        "Vehicle_Type",
        "Number_of_Vehicles",
        "Number_of_Casualties",
        "Junction_Control",
        "Junction_Detail",
        "Day_of_Week",
        "Urban_or_Rural_Area"
    ]
    TARGET = "Accident_Severity"
    NUMERIC_FEATURES = [
        "Speed_limit",
        "Number_of_Vehicles",
        "Number_of_Casualties"
    ]

    user_input = {}

    st.markdown("## Numeric Details")
    user_input["Speed_limit"] = st.number_input(
        "Speed Limit",
        min_value=20,
        max_value=100,
        value=30
    )
    user_input["Number_of_Vehicles"] = st.number_input(
        "Number of Vehicles",
        min_value=1,
        max_value=10,
        value=2
    )
    user_input["Number_of_Casualties"] = st.number_input(
        "Number of Casualties",
        min_value=0,
        max_value=20,
        value=1
    )

    st.markdown("## Conditions")
    for col in FEATURES:
        if col in NUMERIC_FEATURES:
            continue
        options = list(encoders[col].classes_)
        user_input[col] = st.selectbox(
            col.replace("_", " "),
            options
        )

    if st.button("🚦 Predict Severity"):
        row = {}
        for col in FEATURES:
            if col in encoders:
                row[col] = encoders[col].transform(
                    [user_input[col]]
                )[0]
            else:
                row[col] = user_input[col]

        input_df = pd.DataFrame(
            [row],
            columns=FEATURES
        )

        prediction = model.predict(input_df)[0]
        prediction_label = encoders[TARGET].inverse_transform(
            [prediction]
        )[0]

        if prediction_label.lower() == "fatal":
            st.error("🚨 Predicted Severity: FATAL")
        elif prediction_label.lower() == "serious":
            st.warning("⚠️ Predicted Severity: SERIOUS")
        else:
            st.success("✅ Predicted Severity: SLIGHT")

        st.markdown("## Prediction Confidence")
        probabilities = model.predict_proba(input_df)[0]
        labels = encoders[TARGET].classes_
        colors = {
            "Fatal": "#FF2B2B",
            "Serious": "#FFA500",
            "Slight": "#00C853"
        }

        for label, prob in zip(labels, probabilities):
            percentage = prob * 100
            color = colors.get(
                label,
                "#3399FF"
            )
            html = f"""
<div style="margin-bottom:20px;">
    <div style="
        display:flex;
        justify-content:space-between;
        color:white;
        font-weight:bold;
        margin-bottom:6px;
    ">
        <span>{label}</span>
        <span>{percentage:.2f}%</span>
    </div>
    <div style="
        background:#2E2E3A;
        border-radius:10px;
        height:18px;
        width:100%;
        overflow:hidden;
    ">
        <div style="
            background:{color};
            width:{percentage:.2f}%;
            height:18px;
            border-radius:10px;
        ">
        </div>
    </div>
</div>
"""
            st.html(html)
