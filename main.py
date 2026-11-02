import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

BASE_DIR = Path(__file__).parent
RECORD_DIR = BASE_DIR / "내_기록"
RECORD_FILE = RECORD_DIR / "데이터_소개.json"

# 브라우저 탭 제목과 아이콘 (streamlit 명령 중 가장 먼저 실행해야 함)
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# 숫자 카드의 숫자를 크게 보이게 하는 스타일
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] p { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="데이터를 불러오는 중이에요...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_values(s: pd.Series) -> str:
    """열에 들어 있는 값의 종류를 짧은 글로 설명한다."""
    uniq = s.dropna().unique()
    if len(uniq) <= 10:
        return ", ".join(str(v) for v in sorted(uniq, key=str))
    if pd.api.types.is_numeric_dtype(s):
        return f"숫자 ({s.min():g} ~ {s.max():g}, {len(uniq)}가지 값)"
    return f"글자 ({len(uniq)}가지 값)"


def load_record() -> dict:
    """내_기록 폴더에 저장해 둔 내용을 불러온다. 없으면 빈 기록을 돌려준다."""
    try:
        with open(RECORD_FILE, encoding="utf-8") as f:
            data = json.load(f)
        meanings = data.get("우리말_뜻", {})
        source = data.get("출처", "")
        if isinstance(meanings, dict) and isinstance(source, str):
            return {"우리말_뜻": meanings, "출처": source}
    except (OSError, ValueError):
        pass
    return {"우리말_뜻": {}, "출처": ""}


def save_record(record: dict) -> bool:
    try:
        RECORD_DIR.mkdir(exist_ok=True)
        with open(RECORD_FILE, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


# ---------- 제목 ----------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("먼저 우리가 다룰 데이터가 무엇인지 살펴봐요.")

# ---------- 데이터 불러오기 ----------
try:
    df = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

# ---------- 1. 큰 숫자 카드 ----------
st.header("1. 데이터 한눈에 보기")

total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0.0

cards = [
    ("전체 사람 수", f"{total:,}명"),
    ("열 개수", f"{n_cols}개"),
    ("stroke가 1인 사람 수", f"{n_stroke:,}명"),
    ("stroke가 1인 비율", f"{ratio:.1f}%"),
]
for col, (label, value) in zip(st.columns(4), cards):
    with col.container(border=True):
        st.metric(label, value)

# ---------- 저장된 기록 불러오기 (새로고침 때 한 번만) ----------
if "saved_record" not in st.session_state:
    record = load_record()
    st.session_state["saved_record"] = record
    st.session_state["source_text"] = record["출처"]
    st.session_state["base_table"] = pd.DataFrame(
        {
            "열 이름": df.columns,
            "우리말 뜻": [record["우리말_뜻"].get(c, "") for c in df.columns],
            "값의 종류": [describe_values(df[c]) for c in df.columns],
            "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
        }
    )

# ---------- 2. 열 설명 표 ----------
st.header("2. 열 설명 표")
st.write("**우리말 뜻** 칸을 두 번 눌러 교재를 보고 직접 적어 넣어 보세요.")

edited = st.data_editor(
    st.session_state["base_table"],
    key="meaning_editor",
    hide_index=True,
    width="stretch",
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="교재를 보고 직접 적어요."),
        "값의 종류": st.column_config.TextColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
)

# ---------- 3. 처음 다섯 줄 ----------
st.header("3. 데이터 처음 다섯 줄")
st.dataframe(df.head(5), width="stretch")

# ---------- 4. 데이터 출처 ----------
st.header("4. 데이터 출처")
st.text_area(
    "교재에 나온 출처를 적어 주세요.",
    key="source_text",
    height=120,
    placeholder="여기에 출처를 적고, 입력창 밖을 한 번 눌러 주세요.",
)

# ---------- 자동 저장 ----------
current = {
    "우리말_뜻": {
        str(r["열 이름"]): ("" if pd.isna(r["우리말 뜻"]) else str(r["우리말 뜻"]))
        for _, r in edited.iterrows()
    },
    "출처": st.session_state["source_text"],
}

if current != st.session_state["saved_record"]:
    if save_record(current):
        st.session_state["saved_record"] = current
    else:
        st.warning("내_기록 폴더에 저장하지 못했어요. 폴더 쓰기 권한을 확인해 주세요.")

st.caption("적은 내용은 자동으로 `내_기록` 폴더의 json 파일에 저장되고, 새로고침해도 다시 불러와요.")
