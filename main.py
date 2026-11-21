import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
import re

# ---------- 한글 폰트 CSS ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- CSV 읽기 (인코딩 자동 판단) ----------
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ---------- 열 이름 표준화 (KeyError 방지) ----------
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df

# ---------- 산란도 + 추세선 (np.polyfit 사용, statsmodels 불필요) ----------
def make_corr_fig(x, y, x_label, y_label):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return None, None
    corr = df["x"].corr(df["y"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(size=9, color="#2E86AB", opacity=0.75),
            name="측정값",
        )
    )
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    x_line = np.linspace(df["x"].min(), df["x"].max(), 100)
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=slope * x_line + intercept,
            mode="lines",
            line=dict(color="#E84545", width=2.5),
            name=f"추세선 (r = {corr:.3f})",
        )
    )
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=420,
        margin=dict(l=50, r=30, t=40, b=50),
    )
    return fig, corr

# ---------- 데이터 로딩 (캐시) ----------
@st.cache_data
def load_data(env_bytes, growth_bytes):
    env_df = pd.DataFrame()
    growth_df = pd.DataFrame()
    if env_bytes is not None:
        for enc in ("utf-8-sig", "cp949"):
            try:
                env_df = std_cols(pd.read_csv(io.BytesIO(env_bytes), encoding=enc))
                break
            except Exception:
                pass
    if growth_bytes is not None:
        for enc in ("utf-8-sig", "cp949"):
            try:
                growth_df = std_cols(pd.read_csv(io.BytesIO(growth_bytes), encoding=enc))
                break
            except Exception:
                pass
    return env_df, growth_df

# ============================================================
#  메인 앱
# ============================================================
def main():
    st.set_page_config(
        page_title="극지 프로젝트 – 극지 고추냉이의 성장", layout="wide"
    )

    st.title("🧊 극지 프로젝트 – 극지 고추냉이의 성장")

    # --- 사이드바: CSV 업로드 ---
    env_file = st.sidebar.file_uploader("🌡️ 환경 데이터 CSV 업로드", type="csv")
    growth_file = st.sidebar.file_uploader("🌱 생육 결과 데이터 CSV 업로드", type="csv")

    # --- 학교명 추출 (파일명에서 자동) ---
    school_name = "우리 학교"

    if env_file is not None:
        raw_name = os.path.splitext(env_file.name)[0]
        m = re.match(r"(.+?)_", raw_name)
        if m:
            school_name = m.group(1)
    if growth_file is not None and school_name == "우리 학교":
        raw_name = os.path.splitext(growth_file.name)[0]
        m = re.match(r"(.+?)_", raw_name)
        if m:
            school_name = m.group(1)

    # --- 생육 측정 항목 선택 ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📏 생육 측정 항목")
    metric_options = [
        "지상부 길이",
        "지하부 길이",
        "생중량",
        "잎 수",
        "지상부 질량",
        "지하부 질량",
    ]
    selected_metrics = st.sidebar.multiselect(
        "비교할 측정 항목",
        options=metric_options,
        default=["지상부 길이", "생중량"],
    )

    # --- 데이터 로딩 (한 번만 읽기) ---
    env_bytes = env_file.getvalue() if env_file is not None else None
    growth_bytes = growth_file.getvalue() if growth_file is not None else None

    with st.spinner("데이터를 불러오는 중…"):
        env_df, growth_df = load_data(env_bytes, growth_bytes)

    # --- 파일이 전혀 없으면 안내 ---
    if env_file is None and growth_file is None:
        st.info("📁 왼쪽 사이드바에서 CSV 파일을 업로드해 주세요.")
        return

    st.caption(f"🏫 {school_name} 극지식물 생육 분석 대시보드")

    # ============================
    #  탭 1 – 상단부 / 하단부 상관관계
    # ============================
    tab1, tab2, tab3 = st.tabs([
        "상단부–하단부 상관관계",
        "지상부 길이–생중량 상관관계",
        "온도 vs 생육 대조",
    ])

    with tab1:
        try:
            st.header("상단부(지상부)와 하단부(지하부)의 동반 성장")

            if growth_df.empty:
                st.info("📄 생육 결과 CSV를 업로드해 주세요.")
            else:
                # 지상부 길이 vs 지하부 길이
                if {"지상부 길이", "지하부 길이"} <= set(growth_df.columns):
                    x = growth_df["지상부 길이"]
                    y = growth_df["지하부 길이"]
                    fig, r = make_corr_fig(x, y, "지상부 길이 (mm)", "지하부 길이 (mm)")
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
                        st.metric("지상부–지하부 길이 상관계수 (r)", f"{r:.3f}")
                    else:
                        st.info("데이터가 부족하여 그래프를 표시할 수 없어요.")
                else:
                    st.info("⚠️ 지상부 길이 또는 지하부 길이 열이 없어요.")

                st.markdown("---")

                # 지상부 질량 vs 지하부 질량
                if {"지상부 질량", "지하부 질량"} <= set(growth_df.columns):
                    x = growth_df["지상부 질량"]
                    y = growth_df["지하부 질량"]
                    fig, r = make_corr_fig(x, y, "지상부 질량 (g)", "지하부 질량 (g)")
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
                        st.metric("지상부–지하부 질량 상관계수 (r)", f"{r:.3f}")
                    else:
                        st.info("데이터가 부족하여 그래프를 표시할 수 없어요.")
                else:
                    st.info("⚠️ 지상부 질량 또는 지하부 질량 열이 없어요.")

                # 요약 지표
                st.markdown("---")
                st.subheader("📊 생육 요약")
                if "지상부 길이" in growth_df.columns:
                    st.metric(
                        "평균 지상부 길이 (mm)",
                        f"{growth_df['지상부 길이'].dropna().mean():.1f}",
                    )
                if "지하부 길이" in growth_df.columns:
                    st.metric(
                        "평균 지하부 길이 (mm)",
                        f"{growth_df['지하부 길이'].dropna().mean():.1f}",
                    )
                if "생중량" in growth_df.columns:
                    st.metric(
                        "평균 생중량 (g)",
                        f"{growth_df['생중량'].dropna().mean():.2f}",
                    )
                if "잎 수" in growth_df.columns:
                    st.metric(
                        "평균 잎 수 (장)",
                        f"{growth_df['잎 수'].dropna().mean():.1f}",
                    )
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ============================
    #  탭 2 – 지상부 길이 vs 생중량
    # ============================
    with tab2:
        try:
            st.header("지상부 길이와 생중량 — 진짜 생산성 지표 찾기")

            if growth_df.empty:
                st.info("📄 생육 결과 CSV를 업로드해 주세요.")
            else:
                if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                    x = growth_df["지상부 길이"]
                    y = growth_df["생중량"]
                    fig, r = make_corr_fig(x, y, "지상부 길이 (mm)", "생중량 (g)")
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
                        st.metric("지상부 길이–생중량 상관계수 (r)", f"{r:.3f}")
                    else:
                        st.info("데이터가 부족하여 그래프를 표시할 수 없어요.")

                    # 상위·하위 개체 강조 바 차트
                    st.markdown("---")
                    st.subheader("🏆 생중량 상위·하위 개체")

                    g = growth_df[["생중량"]].dropna()
                    if len(g) > 0 and "개체번호" in growth_df.columns:
                        g["개체번호"] = growth_df.loc[g.index, "개체번호"]
                        g = g.sort_values("생중량", ascending=False)

                        top_n = min(5, len(g))
                        bottom_n = min(5, len(g))

                        colors = []
                        for i in range(len(g)):
                            if i < top_n:
                                colors.append("#2E86AB")
                            elif i >= len(g) - bottom_n:
                                colors.append("#E84545")
                            else:
                                colors.append("#B0B0B0")

                        labels = [str(v) for v in g["개체번호"].values]

                        fig_bar = go.Figure()
                        fig_bar.add_trace(
                            go.Bar(
                                x=labels,
                                y=g["생중량"].values,
                                marker_color=colors,
                                text=[f"{v:.2f}g" for v in g["생중량"].values],
                                textposition="outside",
                            )
                        )
                        fig_bar.update_layout(
                            font=dict(
                                family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                            ),
                            xaxis_title="개체번호",
                            yaxis_title="생중량 (g)",
                            height=400,
                            margin=dict(l=50, r=30, t=40, b=50),
                        )
                        st.plotly_chart(fig_bar, width="stretch")
                else:
                    missing = []
                    if "지상부 길이" not in growth_df.columns:
                        missing.append("지상부 길이")
                    if "생중량" not in growth_df.columns:
                        missing.append("생중량")
                    st.info(f"⚠️ 다음 열이 없어서 그래프를 표시할 수 없어요: {', '.join(missing)}")

                # 통계 용어 설명
                st.markdown("---")
                with st.expander("📖 통계 용어 설명"):
                    st.markdown(
                        """
- **상관계수 (r)**: 두 변수가 함께 움직이는 정도. `-1 ~ +1` 사이 값.
  +1에 가까우면 함께 증가, -1에 가까우면 한쪽 증가 시 다른 쪽은 감소.
- **추세선**: 산점도 위에 그리는 직선. 데이터 경향을 한눈에 보여줌.
- **지상부 길이**: 땅 위 부분(줄기·잎)의 길이. 키만 크다고 productive한 것은 아님.
- **생중량**: 식물을 바로 쟀을 때의 무게(물 포함). 실제 생물량 지표.
"""
                    )
        except Exception as e:
            st.warning(f"⚠️ 이
