import pandas as pd
import plotly.express as px
import streamlit as st

PAGE_TITLE = "탐색"
APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon="🔎", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중이에요...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔎 탐색")
st.caption(f"{APP_TITLE} · 데이터를 여러 방향에서 살펴봐요.")

try:
    df = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

STROKE_LABEL = {0: "겪지 않음", 1: "겪음"}
STROKE_ORDER = ["겪지 않음", "겪음"]
YESNO_LABEL = {0: "없음", 1: "있음"}
YESNO_ORDER = ["없음", "있음"]

# ---------- 1. 나이와 평균 혈당의 분포 ----------
st.header("1. 나이와 평균 혈당의 분포")

left, right = st.columns(2)
with left:
    fig = px.histogram(df, x="age", nbins=30, title="나이 분포", labels={"age": "나이", "count": "사람 수"})
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, width="stretch")
with right:
    fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=40,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, width="stretch")

# ---------- 2. 뇌졸중 여부에 따른 비교 ----------
st.header("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")

work = df.assign(뇌졸중=df["stroke"].map(STROKE_LABEL))

left, right = st.columns(2)
with left:
    fig = px.box(
        work,
        x="뇌졸중",
        y="age",
        color="뇌졸중",
        category_orders={"뇌졸중": STROKE_ORDER},
        title="나이 비교",
        labels={"age": "나이"},
    )
    fig.update_layout(showlegend=False, xaxis_title="뇌졸중 여부")
    st.plotly_chart(fig, width="stretch")
with right:
    fig = px.box(
        work,
        x="뇌졸중",
        y="avg_glucose_level",
        color="뇌졸중",
        category_orders={"뇌졸중": STROKE_ORDER},
        title="평균 혈당 비교",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(showlegend=False, xaxis_title="뇌졸중 여부")
    st.plotly_chart(fig, width="stretch")

group_means = (
    work.groupby("뇌졸중")
    .agg(사람_수=("stroke", "size"), 평균_나이=("age", "mean"), 평균_혈당=("avg_glucose_level", "mean"))
    .reindex(STROKE_ORDER)
    .reset_index()
    .rename(columns={"뇌졸중": "뇌졸중 여부", "사람_수": "사람 수", "평균_나이": "평균 나이", "평균_혈당": "평균 혈당"})
)
group_means["평균 나이"] = group_means["평균 나이"].round(1)
group_means["평균 혈당"] = group_means["평균 혈당"].round(1)
st.write("**두 그룹의 평균값**")
st.dataframe(group_means, hide_index=True, width="stretch")

# ---------- 3. 고혈압·심장병과 뇌졸중 비율 ----------
st.header("3. 고혈압·심장병과 뇌졸중 비율")


def stroke_rate_by(column: str) -> pd.DataFrame:
    """column 값(0/1)별 뇌졸중 비율(%)과 사람 수를 계산한다."""
    out = (
        df.groupby(column)
        .agg(사람_수=("stroke", "size"), 뇌졸중_비율=("stroke", "mean"))
        .reset_index()
    )
    out["구분"] = out[column].map(YESNO_LABEL)
    out["뇌졸중_비율"] = (out["뇌졸중_비율"] * 100).round(2)
    return out


left, right = st.columns(2)
for col, (column, name) in zip((left, right), (("hypertension", "고혈압"), ("heart_disease", "심장병"))):
    rate = stroke_rate_by(column)
    with col:
        fig = px.bar(
            rate,
            x="구분",
            y="뇌졸중_비율",
            color="구분",
            category_orders={"구분": YESNO_ORDER},
            text_auto=".1f",
            hover_data={"사람_수": True, "구분": False},
            title=f"{name} 여부에 따른 뇌졸중 비율",
            labels={"뇌졸중_비율": "뇌졸중 비율(%)", "사람_수": "사람 수"},
        )
        fig.update_layout(showlegend=False, xaxis_title=f"{name} 여부", yaxis_title="뇌졸중 비율(%)")
        st.plotly_chart(fig, width="stretch")

# ---------- 4. bmi가 비어 있는 사람들 ----------
st.header("4. bmi가 비어 있는 사람들")

missing = df[df["bmi"].isna()]
bmi_table = pd.DataFrame(
    {
        "구분": ["bmi가 비어 있는 사람들", "전체 사람들"],
        "사람 수": [len(missing), len(df)],
        "뇌졸중을 겪은 사람 수": [int((missing["stroke"] == 1).sum()), int((df["stroke"] == 1).sum())],
        "뇌졸중 비율(%)": [
            round((missing["stroke"] == 1).mean() * 100, 2) if len(missing) else 0.0,
            round((df["stroke"] == 1).mean() * 100, 2),
        ],
    }
)
st.dataframe(bmi_table, hide_index=True, width="stretch")

# ---------- 5. 흡연 상태별 사람 수 ----------
st.header("5. 흡연 상태별 사람 수")

smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking, hide_index=True, width="stretch")
