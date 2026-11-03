import random
from itertools import product

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(page_title=f"분류 모델 - {APP_TITLE}", page_icon="🤖", layout="wide")

# ---------- 이름과 색 ----------
NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}  # 0 또는 1 값만 가지는 속성
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]  # bmi를 뺀 넷

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

BLUE = "#4285f4"
RED = "#ea4335"
BLUE_REGION = "rgba(66, 133, 244, 0.25)"
RED_REGION = "rgba(234, 67, 53, 0.25)"
BLUE_NODE = "#cfe2ff"
RED_NODE = "#ffd0cc"
LINE_COLOR = "#1a9850"


@st.cache_data(show_spinner="데이터를 불러오는 중이에요...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🤖 분류 모델")
st.caption(f"{APP_TITLE} · stroke가 1이면 뇌졸중, 0이면 아님이에요. 뇌졸중(1)을 양성으로 두었어요.")

try:
    df = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

# ---------- 1. 입력 속성 고르기 ----------
st.header("1. 입력으로 사용할 속성 고르기")

chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=list(NAMES),
    default=DEFAULT_FEATURES,
    format_func=lambda c: NAMES[c],
)
selected = [c for c in NAMES if c in chosen]  # 항상 같은 순서로 맞춘다

if len(selected) < 2:
    st.warning("속성을 두 개 이상 골라 주세요. 그림을 그리려면 적어도 두 개가 필요해요.")
    st.stop()

# ---------- 2. 데이터 나누기 ----------
# 번호 순으로 정렬한 뒤 열 명씩 묶어, 각 묶음의 앞 세 명을 테스트용으로 고정한다.
df = df.sort_values("id").reset_index(drop=True)
test_mask = (df.index % 10) < 3
test_df = df[test_mask]
train_df = df[~test_mask]

X_train = train_df[selected].copy()
X_test = test_df[selected].copy()
y_train = train_df["stroke"]
y_test = test_df["stroke"]

# 체질량지수(bmi)는 고른 경우에만, 훈련용의 중앙값으로 빈 값을 채운다.
if "bmi" in selected:
    bmi_fill = float(X_train["bmi"].median())
    miss_train = int(X_train["bmi"].isna().sum())
    miss_test = int(X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_fill)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_fill)
    st.caption(
        f"체질량지수의 빈 값(훈련용 {miss_train:,}명, 테스트용 {miss_test:,}명)을 "
        f"훈련용의 중앙값 {bmi_fill:.1f}로 채웠어요."
    )

st.caption(
    f"훈련용 {len(train_df):,}명, 테스트용 {len(test_df):,}명으로 나눴어요. "
    "크기 맞추기(표준화)는 훈련용 사람들로만 맞추고, 로지스틱 회귀에 써요. "
    "의사결정트리는 크기를 맞추지 않은 원래 값으로 질문해서 가지 그림을 읽기 쉬워요."
)

# ---------- 모델 만들기 ----------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train_s, y_train)

# ---------- 2. 정확도 카드 ----------
st.header("2. 모델의 정확도")


def accuracy_card(name: str, train_acc: float, test_acc: float) -> None:
    with st.container(border=True):
        st.markdown(
            f'<div style="font-size:0.95rem; line-height:1.35; min-height:3.8em">{name}</div>'
            f'<div style="font-size:0.8rem; opacity:0.7">테스트 데이터 정확도</div>'
            f'<div style="font-size:3rem; font-weight:700; line-height:1.15">{test_acc * 100:.1f}%</div>'
            f'<div style="display:flex; gap:1.2rem; font-size:0.85rem; opacity:0.75; margin-top:0.3rem">'
            f"<span>훈련 {train_acc * 100:.1f}%</span><span>테스트 {test_acc * 100:.1f}%</span></div>",
            unsafe_allow_html=True,
        )


cards = [
    (LR_NAME, lr, X_train_s, X_test_s),
    (TREE_NAME, tree, X_train, X_test),
    (BASE_NAME, base, X_train_s, X_test_s),
]
for col, (name, model, Xtr, Xte) in zip(st.columns(3), cards):
    train_acc = accuracy_score(y_train, model.predict(Xtr))
    test_acc = accuracy_score(y_test, model.predict(Xte))
    with col:
        accuracy_card(name, train_acc, test_acc)

# ---------- 3. 산점도와 경계 ----------
st.header("3. 두 속성으로 보는 경계")

left, right = st.columns(2)
with left:
    x_col = st.selectbox("가로축", selected, format_func=lambda c: NAMES[c])
with right:
    y_col = st.selectbox("세로축", [c for c in selected if c != x_col], format_func=lambda c: NAMES[c])

GRID_N = 120


def axis_range(series: pd.Series, col: str) -> tuple:
    lo, hi = float(series.min()), float(series.max())
    pad = 0.4 if col in BINARY else ((hi - lo) * 0.05 or 0.5)
    return lo - pad, hi + pad


x_lo, x_hi = axis_range(X_test[x_col], x_col)
y_lo, y_hi = axis_range(X_test[y_col], y_col)
xs = [x_lo + (x_hi - x_lo) * i / (GRID_N - 1) for i in range(GRID_N)]
ys = [y_lo + (y_hi - y_lo) * i / (GRID_N - 1) for i in range(GRID_N)]

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 둔다.
medians = X_test.median()
rows = list(product(ys, xs))  # 세로축 값이 바깥, 가로축 값이 안쪽
grid_cols = {}
for c in selected:
    if c == x_col:
        grid_cols[c] = [r[1] for r in rows]
    elif c == y_col:
        grid_cols[c] = [r[0] for r in rows]
    else:
        grid_cols[c] = float(medians[c])
grid = pd.DataFrame(grid_cols, index=range(len(rows)))

pos_idx = list(lr.classes_).index(1)  # 뇌졸중(1)이 양성
Z = lr.predict_proba(scaler.transform(grid[selected]))[:, pos_idx].reshape(GRID_N, GRID_N)
tree_pred = tree.predict(grid[selected]).reshape(GRID_N, GRID_N)

fig = go.Figure()

# 의사결정트리가 나눈 칸 (옅은 색)
fig.add_trace(
    go.Heatmap(
        x=xs,
        y=ys,
        z=tree_pred,
        zmin=0,
        zmax=1,
        colorscale=[[0, BLUE_REGION], [1, RED_REGION]],
        showscale=False,
        hoverinfo="skip",
        showlegend=False,
    )
)

# 로지스틱 회귀가 0.5로 가르는 선
fig.add_trace(
    go.Contour(
        x=xs,
        y=ys,
        z=Z,
        contours=dict(start=0.5, end=0.5, size=1, coloring="lines", showlabels=False),
        colorscale=[[0, LINE_COLOR], [1, LINE_COLOR]],
        line=dict(width=3),
        showscale=False,
        hoverinfo="skip",
        showlegend=False,
    )
)

# 테스트 데이터 점 (이진 속성은 겹치지 않게 살짝 흩어 찍는다)
rng = random.Random(0)


def jitter(values, col):
    if col in BINARY:
        return [v + rng.uniform(-0.12, 0.12) for v in values]
    return list(values)


points = pd.DataFrame({"x": X_test[x_col], "y": X_test[y_col], "stroke": y_test})
points["xj"] = jitter(points["x"], x_col)
points["yj"] = jitter(points["y"], y_col)

hover = (
    f"{NAMES[x_col]}: %{{customdata[0]:~g}}<br>{NAMES[y_col]}: %{{customdata[1]:~g}}"
    "<extra>%{fullData.name}</extra>"
)
for value, name, color, size, opacity in [
    (0, "실제 아님(0)", BLUE, 6, 0.55),
    (1, "실제 뇌졸중(1)", RED, 8, 0.9),
]:
    sub = points[points["stroke"] == value]
    fig.add_trace(
        go.Scatter(
            x=sub["xj"],
            y=sub["yj"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=size, opacity=opacity, line=dict(width=0.5, color="white")),
            customdata=sub[["x", "y"]],
            hovertemplate=hover,
        )
    )

# 선과 칸을 범례에 달기 위한 안내용 표시
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        name="로지스틱 회귀의 0.5 경계선",
        line=dict(color=LINE_COLOR, width=3),
    )
)
present = set(tree_pred.ravel().tolist())
if 0 in present:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="의사결정트리가 '아님'이라고 답하는 칸",
            marker=dict(symbol="square", size=14, color=BLUE_REGION),
        )
    )
if 1 in present:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="의사결정트리가 '뇌졸중'이라고 답하는 칸",
            marker=dict(symbol="square", size=14, color=RED_REGION),
        )
    )

fig.update_xaxes(title_text=NAMES[x_col], range=[x_lo, x_hi])
fig.update_yaxes(title_text=NAMES[y_col], range=[y_lo, y_hi])
for axis_update, col in ((fig.update_xaxes, x_col), (fig.update_yaxes, y_col)):
    if col in BINARY:
        axis_update(tickvals=[0, 1], ticktext=["없음", "있음"])
fig.update_layout(
    height=600,
    legend=dict(title_text="범례", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig, width="stretch")


def fmt_value(col: str, v: float) -> str:
    if col in BINARY:
        if v == 0:
            return "0(없음)"
        if v == 1:
            return "1(있음)"
        return f"{v:g}"
    return f"{v:.1f}"


others = [c for c in selected if c not in (x_col, y_col)]
if others:
    fixed = ", ".join(f"{NAMES[c]} = {fmt_value(c, float(medians[c]))}" for c in others)
    st.write(f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed}")
else:
    st.write("고른 속성이 두 개뿐이라 따로 세워 둔 속성은 없어요.")

if Z.min() >= 0.5:
    st.write("이 그림 범위 어디에서나 뇌졸중일 확률이 0.5 이상이라, 로지스틱 회귀의 0.5 경계선은 그림 밖에 있어요.")
elif Z.max() < 0.5:
    st.write("이 그림 범위 어디에서나 뇌졸중일 확률이 0.5보다 낮아서, 로지스틱 회귀의 0.5 경계선은 그림 밖에 있어요.")
else:
    st.write("로지스틱 회귀의 0.5 경계선이 그림 안에 있어요. 선의 한쪽은 확률이 0.5보다 높고, 다른 쪽은 낮아요.")

if any(c in BINARY for c in (x_col, y_col)):
    st.caption("고혈압·심장병처럼 0과 1만 있는 축은 점이 겹치지 않게 살짝 흩어서 찍었어요. 계산에는 원래 값을 썼어요.")

# ---------- 4. 의사결정트리가 던진 질문 ----------
st.header("4. 의사결정트리가 던진 질문")

t = tree.tree_
path = tree.decision_path(X_train)
stroke_counts = path.T.dot(y_train.to_numpy())  # 마디마다 훈련용 중 뇌졸중인 사람 수

dot = [
    "digraph T {",
    'graph [bgcolor="transparent", nodesep=0.35, ranksep=0.5];',
    'node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=12, '
    'fontcolor="#111111", color="#888888", fillcolor="#f2f2f2"];',
    'edge [color="#888888", fontcolor="#888888", fontname="sans-serif", fontsize=12];',
]
leaf_total = 0
leaf_no = 0
used_features = set()

for i in range(t.node_count):
    n = int(t.n_node_samples[i])
    k = int(stroke_counts[i])
    stats = f"훈련용 {n:,}명 중 뇌졸중 {k:,}명 ({k / n * 100:.1f}%)"
    left_child, right_child = int(t.children_left[i]), int(t.children_right[i])

    if left_child == -1:  # 더 묻지 않고 답을 내는 마디
        answer = 1 if 2 * k > n else 0
        leaf_total += 1
        leaf_no += 1 if answer == 0 else 0
        label = f"답: {'뇌졸중' if answer else '아님'}\\n{stats}"
        fill = RED_NODE if answer else BLUE_NODE
        dot.append(f'n{i} [label="{label}", fillcolor="{fill}"];')
    else:
        feature_index = int(t.feature[i])
        used_features.add(feature_index)
        col = selected[feature_index]
        if col in BINARY:
            question = f"{NAMES[col]}이 있나요?"
            left_label, right_label = "아니요", "예"
        else:
            question = f"{NAMES[col]} ≤ {float(t.threshold[i]):.1f} ?"
            left_label, right_label = "예", "아니요"
        dot.append(f'n{i} [label="{question}\\n{stats}"];')
        dot.append(f'n{i} -> n{left_child} [label="{left_label}"];')
        dot.append(f'n{i} -> n{right_child} [label="{right_label}"];')

dot.append("}")
st.graphviz_chart("\n".join(dot))
st.caption("파란 칸은 '아님', 빨간 칸은 '뇌졸중'이라고 답하는 마디예요. 회색 칸은 질문을 더 던지는 마디예요.")

st.write(f"답을 내는 마디는 모두 {leaf_total}칸이고, 그중 {leaf_no}칸이 '아님'이라고 답해요.")
asked = [NAMES[c] for idx, c in enumerate(selected) if idx in used_features]
if asked:
    st.write(f"고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 것: {', '.join(asked)} ({len(asked)}개)")
else:
    st.write(f"고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 것: 없음 (질문 없이 바로 답을 내요)")
