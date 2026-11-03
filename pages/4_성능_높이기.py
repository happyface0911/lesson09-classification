import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(page_title=f"성능 높이기 - {APP_TITLE}", page_icon="🚀", layout="wide")

# ---------- 이름과 색 ----------
NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}  # 0 또는 1 값만 가지는 속성
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"

NOT_AVAILABLE = "계산할 수 없음"

BLUE = "#4285f4"  # 실제 아님
RED = "#ea4335"  # 실제 뇌졸중
BEFORE_COLOR = "#8e44ad"  # 가중치를 맞추기 전
AFTER_COLOR = "#1a9850"  # 가중치를 맞춘 뒤


@st.cache_data(show_spinner="데이터를 불러오는 중이에요...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🚀 성능 높이기")
st.caption(f"{APP_TITLE} · 뇌졸중을 겪은 사람이 아주 드물어서, 양쪽의 가중치를 맞추면 어떻게 달라지는지 살펴봐요.")

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

# ---------- 데이터 나누기 ('채점표' 페이지와 같은 방법) ----------
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
    f"훈련용 {len(train_df):,}명으로 학습하고, 테스트용 {len(test_df):,}명으로 채점해요. "
    "크기 맞추기(표준화)는 훈련용 사람들로만 맞추고 로지스틱 회귀에 써요. "
    "의사결정트리는 크기를 맞추지 않은 원래 값으로 질문해요."
)

# ---------- 모델 만들기: 가중치를 맞추기 전과 맞춘 뒤 ----------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# class_weight="balanced": 드문 뇌졸중 쪽의 틀림에 더 큰 무게를 줘서 양쪽의 가중치를 맞춘다.
lr_before = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
lr_after = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train_s, y_train)
tree_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
tree_after = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42, class_weight="balanced"
).fit(X_train, y_train)

predictions = {
    (LR_NAME, "전"): pd.Series(lr_before.predict(X_test_s), index=y_test.index),
    (LR_NAME, "후"): pd.Series(lr_after.predict(X_test_s), index=y_test.index),
    (TREE_NAME, "전"): pd.Series(tree_before.predict(X_test), index=y_test.index),
    (TREE_NAME, "후"): pd.Series(tree_after.predict(X_test), index=y_test.index),
}


def confusion_counts(actual: pd.Series, predicted: pd.Series) -> dict:
    return {
        "TP": int(((actual == 1) & (predicted == 1)).sum()),
        "FP": int(((actual == 0) & (predicted == 1)).sum()),
        "FN": int(((actual == 1) & (predicted == 0)).sum()),
        "TN": int(((actual == 0) & (predicted == 0)).sum()),
    }


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


counts = {key: confusion_counts(y_test, pred) for key, pred in predictions.items()}

# ---------- 2. 가중치를 맞추기 전과 맞춘 뒤 ----------
st.header("2. 가중치를 맞추기 전과 맞춘 뒤")
st.write(
    f"테스트용 {len(y_test):,}명 가운데 뇌졸중인 사람은 {int((y_test == 1).sum()):,}명이에요. "
    "뇌졸중(1)을 양성으로 두었고, '맞춘 뒤'는 드문 뇌졸중 쪽을 틀렸을 때 더 크게 벌점을 주도록 "
    "양쪽의 가중치를 맞추고 다시 학습한 결과예요."
)

left, right = st.columns(2)
for col, name in zip((left, right), (LR_NAME, TREE_NAME)):
    before, after = counts[(name, "전")], counts[(name, "후")]
    m_before, m_after = four_metrics(before), four_metrics(after)
    table = pd.DataFrame(
        {
            "맞추기 전": [
                f"{before['TP']:,}명",
                f"{before['FN']:,}명",
                f"{before['FP']:,}명",
                f"{before['TN']:,}명",
                m_before["정확도"],
                m_before["재현율"],
                m_before["정밀도"],
                m_before["F1"],
            ],
            "맞춘 뒤": [
                f"{after['TP']:,}명",
                f"{after['FN']:,}명",
                f"{after['FP']:,}명",
                f"{after['TN']:,}명",
                m_after["정확도"],
                m_after["재현율"],
                m_after["정밀도"],
                m_after["F1"],
            ],
        },
        index=[
            "찾아낸 환자 (TP)",
            "놓친 환자 (FN)",
            "헛짚은 사람 (FP)",
            "맞게 아니라고 한 사람 (TN)",
            "정확도",
            "재현율",
            "정밀도",
            "F1",
        ],
    )
    with col:
        st.subheader(name)
        st.dataframe(table, width="stretch")
        st.write(
            f"찾아낸 환자 {before['TP']:,}명에서 {after['TP']:,}명으로, "
            f"놓친 환자 {before['FN']:,}명에서 {after['FN']:,}명으로, "
            f"헛짚은 사람 {before['FP']:,}명에서 {after['FP']:,}명으로 바뀌었어요."
        )

st.caption(
    "재현율 = TP ÷ (TP + FN), 곧 실제 뇌졸중인 사람 수로 나눠요 · "
    "정밀도 = TP ÷ (TP + FP), 곧 뇌졸중이라 예측한 사람 수로 나눠요 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). 나누는 수가 0이면 '계산할 수 없음'이라고 적어요."
)


# ---------- 경계선 계산에 쓰는 도구 ----------
def linspace(lo: float, hi: float, n: int) -> list:
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def axis_range(series: pd.Series, col: str) -> tuple:
    lo, hi = float(series.min()), float(series.max())
    pad = 0.4 if col in BINARY else ((hi - lo) * 0.05 or 0.5)
    return lo - pad, hi + pad


medians = X_test.median()  # 그림의 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 둔다.


def linear_terms(model: LogisticRegression, axes: list) -> tuple:
    """0.5 경계는 (결정 점수) = 0 인 곳이다. 축 속성의 원래 값에 대한 계수와 상수항을 돌려준다."""
    coef = model.coef_[0]
    constant = float(model.intercept_[0])
    for j, col in enumerate(selected):
        w = float(coef[j]) / float(scaler.scale_[j])  # 원래 값 한 단위당 결정 점수의 변화
        constant -= w * float(scaler.mean_[j])
        if col not in axes:
            constant += w * float(medians[col])
    slopes = []
    for col in axes:
        j = selected.index(col)
        slopes.append(float(coef[j]) / float(scaler.scale_[j]))
    return slopes, constant


def fixed_text(axes: list) -> str:
    others = [c for c in selected if c not in axes]
    if not others:
        return "고른 속성이 모두 축으로 쓰여서 따로 세워 둔 속성은 없어요."

    def fmt(col: str, v: float) -> str:
        if col in BINARY:
            if v == 0:
                return "0(없음)"
            if v == 1:
                return "1(있음)"
            return f"{v:g}"
        return f"{v:.1f}"

    fixed = ", ".join(f"{NAMES[c]} = {fmt(c, float(medians[c]))}" for c in others)
    return f"축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed}"


def outside_text(label: str, score_at_center: float) -> str:
    side = "0.5 이상" if score_at_center >= 0 else "0.5보다 낮아서"
    if score_at_center >= 0:
        return f"{label}: 그림 범위 어디에서나 뇌졸중일 확률이 {side}이라 경계가 그림 밖에 있어요."
    return f"{label}: 그림 범위 어디에서나 뇌졸중일 확률이 {side} 경계가 그림 밖에 있어요."


# 이진 속성은 점이 겹치지 않게 살짝 흩어서 찍는다 (계산에는 원래 값을 쓴다).
rng = random.Random(0)


def jitter(values, col):
    if col in BINARY:
        return [v + rng.uniform(-0.12, 0.12) for v in values]
    return list(values)


def tick_labels(col: str) -> dict:
    return dict(tickvals=[0, 1], ticktext=["없음", "있음"]) if col in BINARY else {}


# ---------- 3. 2차원 산점도 ----------
st.header("3. 두 속성으로 보는 0.5 경계선")
st.write("로지스틱 회귀(확률로 답하는 모델)가 뇌졸중일 확률을 0.5로 가르는 선을, 가중치를 맞추기 전과 맞춘 뒤로 겹쳐 그려요.")

c1, c2 = st.columns(2)
with c1:
    x_col = st.selectbox("가로축", selected, format_func=lambda c: NAMES[c])
with c2:
    y_col = st.selectbox("세로축", [c for c in selected if c != x_col], format_func=lambda c: NAMES[c])

x_lo, x_hi = axis_range(X_test[x_col], x_col)
y_lo, y_hi = axis_range(X_test[y_col], y_col)

points = pd.DataFrame({"x": X_test[x_col], "y": X_test[y_col], "stroke": y_test})
points["xj"] = jitter(points["x"], x_col)
points["yj"] = jitter(points["y"], y_col)

fig2d = go.Figure()
hover2d = (
    f"{NAMES[x_col]}: %{{customdata[0]:~g}}<br>{NAMES[y_col]}: %{{customdata[1]:~g}}"
    "<extra>%{fullData.name}</extra>"
)
for value, name, color, size, opacity in [
    (0, "실제 아님(0)", BLUE, 6, 0.55),
    (1, "실제 뇌졸중(1)", RED, 8, 0.9),
]:
    sub = points[points["stroke"] == value]
    fig2d.add_trace(
        go.Scatter(
            x=sub["xj"],
            y=sub["yj"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=size, opacity=opacity, line=dict(width=0.5, color="white")),
            customdata=sub[["x", "y"]],
            hovertemplate=hover2d,
        )
    )


def boundary_line(slopes: list, constant: float, n: int = 400) -> tuple:
    """a_x * x + a_y * y + 상수 = 0 인 선을 그림 범위 안에서만 잡는다. (x 목록, y 목록, 보이는 점 수)"""
    a_x, a_y = slopes
    if abs(a_y) >= abs(a_x) and abs(a_y) > 1e-12:
        xs = linspace(x_lo, x_hi, n)
        ys = [-(constant + a_x * x) / a_y for x in xs]
        ys = [y if y_lo <= y <= y_hi else None for y in ys]
    elif abs(a_x) > 1e-12:
        ys = linspace(y_lo, y_hi, n)
        xs = [-(constant + a_y * y) / a_x for y in ys]
        xs = [x if x_lo <= x <= x_hi else None for x in xs]
    else:
        return [None], [None], 0
    visible = sum(1 for x, y in zip(xs, ys) if x is not None and y is not None)
    return xs, ys, visible


axes2d = [x_col, y_col]
notes2d = []
for label, model, color in (
    ("가중치를 맞추기 전의 0.5 경계선", lr_before, BEFORE_COLOR),
    ("가중치를 맞춘 뒤의 0.5 경계선", lr_after, AFTER_COLOR),
):
    slopes, constant = linear_terms(model, axes2d)
    line_x, line_y, visible = boundary_line(slopes, constant)
    fig2d.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name=label,
            line=dict(color=color, width=3),
            hoverinfo="skip",
            connectgaps=False,
        )
    )
    if visible:
        notes2d.append(f"{label}: 그림 안에 있어요.")
    else:
        center = slopes[0] * (x_lo + x_hi) / 2 + slopes[1] * (y_lo + y_hi) / 2 + constant
        notes2d.append(outside_text(label, center))

fig2d.update_xaxes(title_text=NAMES[x_col], range=[x_lo, x_hi], **tick_labels(x_col))
fig2d.update_yaxes(title_text=NAMES[y_col], range=[y_lo, y_hi], **tick_labels(y_col))
fig2d.update_layout(
    height=600,
    legend=dict(title_text="범례", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig2d, width="stretch")

st.write(fixed_text(axes2d))
for note in notes2d:
    st.write(note)
if any(c in BINARY for c in axes2d):
    st.caption("고혈압·심장병처럼 0과 1만 있는 축은 점이 겹치지 않게 살짝 흩어서 찍었어요. 계산에는 원래 값을 썼어요.")

# ---------- 4. 3차원 산점도 ----------
st.header("4. 세 속성으로 보는 0.5 평면")

if len(selected) < 3:
    st.info("속성을 세 개 이상 고르면, 세 속성을 축으로 하는 3차원 그림과 0.5 평면이 나타나요.")
    st.stop()

st.write("마우스로 끌어서 돌려 볼 수 있어요. 평면은 확률이 0.5인 곳을 이은 면이에요.")

d1, d2, d3 = st.columns(3)
with d1:
    ax_x = st.selectbox("3차원 가로축", selected, format_func=lambda c: NAMES[c])
with d2:
    ax_y = st.selectbox("3차원 세로축", [c for c in selected if c != ax_x], format_func=lambda c: NAMES[c])
with d3:
    ax_z = st.selectbox(
        "3차원 높이축",
        [c for c in selected if c not in (ax_x, ax_y)],
        format_func=lambda c: NAMES[c],
    )

axes3d = [ax_x, ax_y, ax_z]
ranges3d = [axis_range(X_test[c], c) for c in axes3d]

pts3 = pd.DataFrame({"x": X_test[ax_x], "y": X_test[ax_y], "z": X_test[ax_z], "stroke": y_test})
pts3["xj"] = jitter(pts3["x"], ax_x)
pts3["yj"] = jitter(pts3["y"], ax_y)
pts3["zj"] = jitter(pts3["z"], ax_z)

fig3d = go.Figure()
hover3d = (
    f"{NAMES[ax_x]}: %{{customdata[0]:~g}}<br>{NAMES[ax_y]}: %{{customdata[1]:~g}}"
    f"<br>{NAMES[ax_z]}: %{{customdata[2]:~g}}<extra>%{{fullData.name}}</extra>"
)
for value, name, color, size, opacity in [
    (0, "실제 아님(0)", BLUE, 3, 0.45),
    (1, "실제 뇌졸중(1)", RED, 5, 0.95),
]:
    sub = pts3[pts3["stroke"] == value]
    fig3d.add_trace(
        go.Scatter3d(
            x=sub["xj"],
            y=sub["yj"],
            z=sub["zj"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=size, opacity=opacity),
            customdata=sub[["x", "y", "z"]],
            hovertemplate=hover3d,
        )
    )


def plane_mesh(slopes: list, constant: float, n: int = 24):
    """a·x + 상수 = 0 인 평면을 상자 안에서만 삼각형으로 만든다. 상자 안에 없으면 None."""
    w = max(range(3), key=lambda j: abs(slopes[j]))
    if abs(slopes[w]) < 1e-12:
        return None
    u, v = [j for j in range(3) if j != w]
    us = linspace(ranges3d[u][0], ranges3d[u][1], n)
    vs = linspace(ranges3d[v][0], ranges3d[v][1], n)
    w_lo, w_hi = ranges3d[w]

    coords = [[], [], []]
    valid = []
    for i in range(n):
        for j in range(n):
            w_val = -(constant + slopes[u] * us[i] + slopes[v] * vs[j]) / slopes[w]
            coords[u].append(us[i])
            coords[v].append(vs[j])
            coords[w].append(w_val)
            valid.append(w_lo <= w_val <= w_hi)

    tri_i, tri_j, tri_k = [], [], []
    for i in range(n - 1):
        for j in range(n - 1):
            p1, p2, p3, p4 = i * n + j, (i + 1) * n + j, i * n + j + 1, (i + 1) * n + j + 1
            if valid[p1] and valid[p2] and valid[p3] and valid[p4]:
                tri_i += [p1, p2]
                tri_j += [p2, p4]
                tri_k += [p3, p3]
    if not tri_i:
        return None
    return coords, tri_i, tri_j, tri_k


notes3d = []
for label, model, color in (
    ("가중치를 맞추기 전의 0.5 평면", lr_before, BEFORE_COLOR),
    ("가중치를 맞춘 뒤의 0.5 평면", lr_after, AFTER_COLOR),
):
    slopes, constant = linear_terms(model, axes3d)
    mesh = plane_mesh(slopes, constant)
    # 평면을 범례에 달기 위한 안내용 표시
    fig3d.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="lines",
            name=label,
            line=dict(color=color, width=8),
        )
    )
    if mesh is not None:
        coords, tri_i, tri_j, tri_k = mesh
        fig3d.add_trace(
            go.Mesh3d(
                x=coords[0],
                y=coords[1],
                z=coords[2],
                i=tri_i,
                j=tri_j,
                k=tri_k,
                color=color,
                opacity=0.45,
                flatshading=True,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        notes3d.append(f"{label}: 그림 안에 있어요.")
    else:
        center = sum(s * (r[0] + r[1]) / 2 for s, r in zip(slopes, ranges3d)) + constant
        notes3d.append(outside_text(label, center))

fig3d.update_layout(
    height=700,
    legend=dict(title_text="범례", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    scene=dict(
        xaxis=dict(title=NAMES[ax_x], range=list(ranges3d[0]), **tick_labels(ax_x)),
        yaxis=dict(title=NAMES[ax_y], range=list(ranges3d[1]), **tick_labels(ax_y)),
        zaxis=dict(title=NAMES[ax_z], range=list(ranges3d[2]), **tick_labels(ax_z)),
    ),
    margin=dict(l=0, r=0, t=40, b=0),
)
st.plotly_chart(fig3d, width="stretch")

st.write(fixed_text(axes3d))
for note in notes3d:
    st.write(note)
if any(c in BINARY for c in axes3d):
    st.caption("고혈압·심장병처럼 0과 1만 있는 축은 점이 겹치지 않게 살짝 흩어서 찍었어요. 계산에는 원래 값을 썼어요.")
