import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(page_title=f"채점표 - {APP_TITLE}", page_icon="📝", layout="wide")

# '분류 모델' 페이지에서 처음 골라져 있던 속성(체질량지수를 뺀 넷)과 같다.
FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)"

NOT_AVAILABLE = "계산할 수 없음"
LIST_LIMIT = 20


@st.cache_data(show_spinner="데이터를 불러오는 중이에요...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("📝 채점표")
st.caption(f"{APP_TITLE} · 모델이 테스트 데이터를 얼마나 맞히고 틀렸는지 채점해 봐요. 뇌졸중(1)을 양성으로 두었어요.")

try:
    df = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

# ---------- 데이터 나누기 ('분류 모델' 페이지와 같은 방법) ----------
# 번호 순으로 정렬한 뒤 열 명씩 묶어, 각 묶음의 앞 세 명을 테스트용으로 고정한다.
df = df.sort_values("id").reset_index(drop=True)
test_mask = (df.index % 10) < 3
test_df = df[test_mask]
train_df = df[~test_mask]

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()
y_train = train_df["stroke"]
y_test = test_df["stroke"]

# ---------- 모델 만들기 ('분류 모델' 페이지와 같은 방법) ----------
scaler = StandardScaler().fit(X_train)  # 크기 맞추기는 훈련용으로만
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train_s, y_train)

# 테스트 데이터에 대한 예측 (행 번호를 테스트 데이터와 맞춘다)
predictions = {
    LR_NAME: pd.Series(lr.predict(X_test_s), index=y_test.index),
    TREE_NAME: pd.Series(tree.predict(X_test), index=y_test.index),
    BASE_NAME: pd.Series(base.predict(X_test_s), index=y_test.index),
}

st.caption(
    "입력 속성은 나이·평균 혈당·고혈압·심장병 넷이에요. "
    f"훈련용 {len(train_df):,}명으로 학습하고, 테스트용 {len(test_df):,}명으로 채점해요."
)


def confusion_counts(actual: pd.Series, predicted: pd.Series) -> dict:
    return {
        "TP": int(((actual == 1) & (predicted == 1)).sum()),
        "FP": int(((actual == 0) & (predicted == 1)).sum()),
        "FN": int(((actual == 1) & (predicted == 0)).sum()),
        "TN": int(((actual == 0) & (predicted == 0)).sum()),
    }


counts = {name: confusion_counts(y_test, pred) for name, pred in predictions.items()}

# ---------- 1. 혼동행렬 ----------
st.header("1. 혼동행렬")
st.write("행은 **실제**, 열은 **예측**이에요. 칸마다 사람 수와 약자를 함께 적었어요.")

left, right = st.columns(2)
for col, name in zip((left, right), (LR_NAME, TREE_NAME)):
    c = counts[name]
    matrix = pd.DataFrame(
        [
            [f"TP · {c['TP']:,}명", f"FN · {c['FN']:,}명", f"{c['TP'] + c['FN']:,}명"],
            [f"FP · {c['FP']:,}명", f"TN · {c['TN']:,}명", f"{c['FP'] + c['TN']:,}명"],
            [f"{c['TP'] + c['FP']:,}명", f"{c['FN'] + c['TN']:,}명", f"{sum(c.values()):,}명"],
        ],
        index=["실제 뇌졸중(1)", "실제 아님(0)", "합계"],
        columns=["예측 뇌졸중(1)", "예측 아님(0)", "합계"],
    )
    matrix.index.name = "실제 ↓ / 예측 →"
    with col:
        st.subheader(name)
        st.dataframe(matrix, width="stretch")

st.caption(
    "TP: 뇌졸중이고 뇌졸중이라 예측 · FN: 뇌졸중인데 아니라고 예측 · "
    "FP: 아닌데 뇌졸중이라 예측 · TN: 아니고 아니라고 예측"
)

# ---------- 2. 네 가지 지표 ----------
st.header("2. 두 모델의 채점표")

n_test = len(y_test)
n_stroke = int((y_test == 1).sum())
st.write(f"테스트용 {n_test:,}명 가운데 뇌졸중인 사람은 {n_stroke:,}명이에요.")


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def four_metrics(c: dict) -> dict:
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    total = tp + fp + fn + tn

    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) else None  # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) else None  # 분모: 뇌졸중이라 예측한 사람 수
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    def show(v):
        return NOT_AVAILABLE if v is None else percent(v)

    return {"정확도": show(accuracy), "재현율": show(recall), "정밀도": show(precision), "F1": show(f1)}


score_rows = []
for name in (LR_NAME, TREE_NAME, BASE_NAME):
    row = {"모델": name}
    row.update(four_metrics(counts[name]))
    score_rows.append(row)
st.dataframe(pd.DataFrame(score_rows), hide_index=True, width="stretch")

st.caption(
    "정확도 = (TP + TN) ÷ 전체 사람 수 · "
    "재현율 = TP ÷ (TP + FN), 곧 실제 뇌졸중인 사람 수로 나눠요 · "
    "정밀도 = TP ÷ (TP + FP), 곧 뇌졸중이라 예측한 사람 수로 나눠요 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    "나누는 수가 0이면 '계산할 수 없음'이라고 적어요."
)

# ---------- 3. 놓친 환자와 헛짚은 사람 ----------
st.header("3. 놓친 환자와 헛짚은 사람")

model_name = st.selectbox("어느 모델의 목록을 볼까요?", [LR_NAME, TREE_NAME])
pred = predictions[model_name]

LIST_COLUMNS = ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]
yes_no = {0: "없음", 1: "있음"}


def make_list(mask: pd.Series) -> tuple:
    """조건에 맞는 사람을 나이가 많은 순으로 정렬해 (전체 인원수, 보여 줄 표)를 돌려준다."""
    people = test_df[mask][LIST_COLUMNS].sort_values(["age", "id"], ascending=[False, True])
    total = len(people)
    shown = people.head(LIST_LIMIT)
    table = pd.DataFrame(
        {
            "번호": shown["id"].tolist(),
            "나이": [f"{v:g}" for v in shown["age"]],
            "평균 혈당": [f"{v:.1f}" for v in shown["avg_glucose_level"]],
            "고혈압 여부": [yes_no[int(v)] for v in shown["hypertension"]],
            "심장병 여부": [yes_no[int(v)] for v in shown["heart_disease"]],
        }
    )
    return total, table


missed_total, missed_table = make_list((y_test == 1) & (pred == 0))  # 놓친 환자 (FN)
false_total, false_table = make_list((y_test == 0) & (pred == 1))  # 헛짚은 사람 (FP)

st.write(f"선택한 모델: **{model_name}** · 나이가 많은 순서로 보여 줘요.")

left, right = st.columns(2)
with left:
    title = f"놓친 환자 (실제 뇌졸중인데 아니라고 예측) · 전체 {missed_total:,}명"
    if missed_total > LIST_LIMIT:
        title += f" · 앞 {LIST_LIMIT}명만 표시"
    st.subheader(title)
    if missed_total:
        st.dataframe(missed_table, hide_index=True, width="stretch")
    else:
        st.info("해당하는 사람이 없어요.")
with right:
    title = f"헛짚은 사람 (아닌데 뇌졸중이라고 예측) · 전체 {false_total:,}명"
    if false_total > LIST_LIMIT:
        title += f" · 앞 {LIST_LIMIT}명만 표시"
    st.subheader(title)
    if false_total:
        st.dataframe(false_table, hide_index=True, width="stretch")
    else:
        st.info("해당하는 사람이 없어요.")

selected_counts = counts[model_name]
st.write(
    f"놓친 환자 {missed_total:,}명은 혼동행렬의 FN 칸({selected_counts['FN']:,}명)과, "
    f"헛짚은 사람 {false_total:,}명은 FP 칸({selected_counts['FP']:,}명)과 같아요."
)
