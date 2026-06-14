
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="WHO 기대수명 예측 서비스",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------
# 파일 로드
# -----------------------------
@st.cache_resource
def load_models():
    models = {
        "Linear": joblib.load("linear_model.pkl"),
        "Poly": joblib.load("poly_model.pkl"),
        "Ridge": joblib.load("ridge_model.pkl")
    }
    return models

@st.cache_data
def load_data():
    df = pd.read_csv("life_expectancy_clean.csv")
    features = joblib.load("features.pkl")
    metrics_df = pd.read_csv("model_metrics.csv")
    return df, features, metrics_df

models = load_models()
df, features, metrics_df = load_data()

# -----------------------------
# 제목
# -----------------------------
st.title("WHO 기대수명 예측 웹 서비스")
st.write("""
이 웹 서비스는 WHO 기대수명 데이터를 기반으로 사용자가 입력한 여러 건강·경제 관련 특성을 이용해
기대수명(Life expectancy)을 예측합니다.

본 과제에서는 기존 실습 특성인 **Schooling**을 제외하고,
**Adult Mortality, BMI, GDP, Alcohol, Polio** 특성을 사용했습니다.
""")

st.divider()

# -----------------------------
# 사이드바 입력
# -----------------------------
st.sidebar.header("입력값 조절")

input_data = {}

for feature in features:
    min_value = float(df[feature].min())
    max_value = float(df[feature].max())
    median_value = float(df[feature].median())

    step_value = (max_value - min_value) / 100

    if step_value == 0:
        step_value = 1.0

    input_data[feature] = st.sidebar.slider(
        label=feature,
        min_value=min_value,
        max_value=max_value,
        value=median_value,
        step=step_value
    )

# -----------------------------
# 모델 선택
# -----------------------------
st.sidebar.header("모델 선택")

selected_model_name = st.sidebar.selectbox(
    "예측에 사용할 모델을 선택하세요",
    ["Linear", "Poly", "Ridge"]
)

selected_model = models[selected_model_name]

# -----------------------------
# 실시간 예측
# -----------------------------
input_df = pd.DataFrame([input_data])

prediction = selected_model.predict(input_df)[0]

st.subheader("실시간 기대수명 예측 결과")

st.markdown(
    f"""
    <div style="
        background-color:#f0f2f6;
        padding:30px;
        border-radius:15px;
        text-align:center;
        font-size:32px;
        font-weight:bold;">
        선택한 모델: {selected_model_name}<br>
        예측 기대수명: {prediction:.2f} 세
    </div>
    """,
    unsafe_allow_html=True
)

st.write("입력된 데이터")
st.dataframe(input_df)

st.divider()

# -----------------------------
# 모델 성능 비교 테이블
# -----------------------------
st.subheader("각 모델의 성능 비교")

st.write("""
아래 표는 3가지 모델의 훈련 데이터와 테스트 데이터 성능을 비교한 결과입니다.
Complexity는 PolynomialFeatures 적용 후 생성된 총 특성 개수를 의미합니다.
""")

st.dataframe(metrics_df, use_container_width=True)

# -----------------------------
# Test R2 Bar Chart
# -----------------------------
st.subheader("Test R² 점수 비교 그래프")

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(metrics_df["Model"], metrics_df["Test R2"])
ax.set_xlabel("Model")
ax.set_ylabel("Test R² Score")
ax.set_title("Comparison of Test R² Scores")
ax.axhline(0, linewidth=1)

st.pyplot(fig)

st.divider()

# -----------------------------
# 해석 문구
# -----------------------------
st.subheader("결과 해석")

st.write("""
- **Linear 모델**은 1차 항만 사용하므로 구조가 단순하고 과대적합 위험이 상대적으로 낮습니다.
- **Poly 모델**은 3차 다항 특성을 사용하여 복잡도가 크게 증가하므로, 훈련 데이터에 과도하게 맞춰질 수 있습니다.
- **Ridge 모델**은 3차 다항 회귀를 사용하지만, 릿지 규제(alpha=1.0)를 적용하여 과대적합을 완화합니다.
""")
