import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── CSS: 한글 폰트 ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ── CSV 읽기 (인코딩 폴백) ───────────────────────────────────────
@st.cache_data
def read_csv_cached(file):
    """업로드된 파일을 utf-8-sig → cp949 순으로 시도하여 읽는다."""
    if file is None:
        return pd.DataFrame()
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ── 열 이름 표준화 ──────────────────────────────────────────────
def std_cols(df):
    """학교마다 다른 열 이름을 표준 이름으로 통일한다."""
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


# ── 학교에서 추출한 이름 표시 ────────────────────────────────────
def school_name_from_filename(filename):
    """파일명 앞부분에서 학교명을 추출한다."""
    if filename is None:
        return "미上传"
    base = os.path.splitext(os.path.basename(filename))[0]
    # 언더스코어/숫자 앞에서 자르기
    import re
    m = re.match(r"^[A-Za-z가-힣0-9]+", base)
    return m.group(0) if m else base[:10]


# ── 지표 카드 도움 함수 ──────────────────────────────────────────
def metric_card(label, value, format_str=None):
    """st.metric 한 줄 래퍼."""
    if format_str:
        st.metric(label, format_str.format(value))
    else:
        st.metric(label, value)


# ── 막대 그래프 공통 함수 ────────────────────────────────────────
@st.cache_data
def bar_chart(df, col, title, ylabel, top_n=None, highlight_col=None):
    """개체별 막대 그래프 (상위/하위 강조 가능)."""
    d = df[[ "개체번호", col]].dropna().copy()
    if d.empty:
        return None
    d = d.sort_values(col, ascending=False).reset_index(drop=True)
    if top_n is not None:
        top_ids = set(d.head(top_n)["개체번호"].tolist())
        bot_ids = set(d.tail(top_n)["개체번호"].tolist())
        colors = []
        for _, row in d.iterrows():
            if row["개체번호"] in top_ids:
                colors.append("#2ecc71")
            elif row["개체번호"] in bot_ids:
                colors.append("#e74c3c")
            else:
                colors.append("#3498db")
    else:
        colors = "#3498db"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["개체번호"].astype(str),
        y=d[col],
        marker_color=colors,
        hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="개체번호",
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(t=40, l=40, r=20, b=40),
        height=380,
    )
    return fig


# ── 산점도 + 추세선 공통 함수 ────────────────────────────────────
@st.cache_data
def scatter_with_trend(x, y, x_label, y_label, title):
    """산점도 + np.polyfit 추세선 + 상관계수."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 2:
        return None, None
    r = d["x"].corr(d["y"])
    slope, intercept = np.polyfit(d["x"], d["y"], 1)
    line_x = np.array([d["x"].min(), d["x"].max()])
    line_y = slope * line_x + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["x"], y=d["y"],
        mode="markers",
        marker=dict(size=7, color="#3498db", opacity=0.7),
        hovertemplate=f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
        name="데이터",
    ))
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y,
        mode="lines",
        line=dict(color="#e74c3c", width=2, dash="dot"),
        name=f"추세선 (r={r:.3f})",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(t=40, l=40, r=20, b=40),
        height=380,
    )
    return fig, r


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

st.sidebar.title("📁 데이터 업로드")
env_file   = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

st.sidebar.markdown("---")
st.sidebar.subheader("📏 생육 측정 항목 선택")
growth_metrics = st.sidebar.multiselect(
    "비교할 항목 (Tab 3 막대 그래프)",
    ["생중량", "잎 수", "지상부 길이", "지하부 길이"],
    default=["생중량", "잎 수", "지상부 길이", "지하부 길이"],
    help="Tab 3에서 개체별로 비교할 생육 지표를 선택합니다.",
)

st.sidebar.markdown("---")
st.sidebar.info("💡 CSV를 업로드하면 탭이 자동으로 활성화됩니다.\n"
                "두 파일 모두 utf-8-sig / cp949 인코딩을 지원합니다.\n"
                "학교명은 업로드한 파일명 앞부분에서 자동 추출됩니다.")

st.sidebar.markdown("### ℹ️ 통계 용어 간단 설명")
with st.sidebar.expander("📖 용어 설명"):
    st.markdown("""
- **평균**: 모든 값을 더해 개수로 나눈 대표값  
- **표준편차**: 값들이 평균에서 얼마나 퍼져 있는지 나타내는 숫자 (클수록 들쭉날쭉)  
- **CV(변동계수)**: 표준편차를 평균으로 나눈 값 → 단위가 다른 항목끼리 비교할 때 사용  
- **상관계수(r)**: 두 지표가 함께 움직이는 정도 (-1 ~ +1, +1에 가까울수록 함께 증가)
    """)


# ═══════════════════════════════════════════════════════════════
#  데이터 로딩 (한 번만)
# ═══════════════════════════════════════════════════════════════
env_df = read_csv_cached(env_file)
growth_df = read_csv_cached(growth_file)

if not env_df.empty:
    env_df = std_cols(env_df)
if not growth_df.empty:
    growth_df = std_cols(growth_df)

school_env = school_name_from_filename(env_file.name if env_file else None)
school_gro = school_name_from_filename(growth_file.name if growth_file else None)


# ═══════════════════════════════════════════════════════════════
#  TAB 1 — 실험 개요
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")
        st.caption(f"학교: {school_gro or school_env or '미上传'}  |  업로드한 CSV 기준 자동 분석")

        st.markdown("""
### 연구 배경 및 목적
극지식물은 낮은 온도와 제한된 수분 환경에서도 생존하는 식물로,  
개체마다 생장 속도와 형태에 큰 차이가 나타날 수 있습니다.  

이 대시보드는 **우리 학교 극지식물 재배 실험**에서  
각 개체의 **잎 수, 지상부 길이, 지하부 길이, 생중량**을 측정하고  
개체 간 생육 차이를 통계적으로 비교하는 도구입니다.  

평균·표준편차·상관계수 등 기본 통계 지표를 통해  
어떤 개체가 두드러지게 잘 자랐는지,  
어떤 측정 항목이 생중량과 관련이 깊은지 확인할 수 있습니다.
        """)

        # ── 데이터 요약 표 ──────────────────────────────────────
        st.subheader("📋 데이터 요약")

        summary_rows = []
        # 개체 수
        n_gro = len(growth_df) if not growth_df.empty else 0
        summary_rows.append(["생육 결과 데이터", school_gro or "-", f"{n_gro}개체" if n_gro else "-", 
                             ", ".join([c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"] if c in growth_df.columns]) if not growth_df.empty else "-"])

        n_env = len(env_df) if not env_df.empty else 0
        env_cols_str = ", ".join([c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]) if not env_df.empty else "-"
        time_col = "time" if "time" in env_df.columns else None
        if time_col and not env_df.empty:
            tmin = env_df[time_col].min()
            tmax = env_df[time_col].max()
            period = f"{tmin.strftime('%Y-%m-%d')} ~ {tmax.strftime('%Y-%m-%d')}"
        else:
            period = "-"
        summary_rows.append(["환경 데이터", school_env or "-", f"{n_env}행" if n_env else "-", env_cols_str])

        summary_df = pd.DataFrame(summary_rows, columns=["데이터 종류", "출처(학교명)", "규모", "측정 항목"])
        st.dataframe(summary_df, width="stretch", hide_index=True)

        # ── 주요 지표 카드 ──────────────────────────────────────
        st.subheader("📊 주요 지표")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total = len(growth_df) if not growth_df.empty else 0
            st.metric("총 개체수", f"{total}개체" if total else "CSV 업로드 필요")

        with col2:
            if "temperature" in env_df.columns:
                t_vals = env_df["temperature"].dropna()
                if len(t_vals) >= 2:
                    st.metric("평균 온도", f"{t_vals.mean():.1f}°C")
                else:
                    st.metric("평균 온도", "데이터 부족")
            else:
                st.metric("평균 온도", "CSV에 없음")

        with col3:
            if "humidity" in env_df.columns:
                h_vals = env_df["humidity"].dropna()
                if len(h_vals) >= 2:
                    st.metric("평균 습도", f"{h_vals.mean():.1f}%")
                else:
                    st.metric("평균 습도", "데이터 부족")
            else:
                st.metric("평균 습도", "CSV에 없음")

        with col4:
            if "생중량" in growth_df.columns:
                w_vals = growth_df["생중량"].dropna()
                if len(w_vals) >= 2:
                    st.metric("평균 생중량", f"{w_vals.mean():.2f}g")
                else:
                    st.metric("평균 생중량", "데이터 부족")
            else:
                st.metric("평균 생중량", "CSV에 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═══════════════════════════════════════════════════════════════
#  TAB 2 — 환경 데이터
# ═══════════════════════════════════════════════════════════════
with tab2:
    try:
        st.header("🌡️ 환경 데이터")
        if env_df.empty:
            st.info("📁 사이드바에서 환경 데이터 CSV를 업로드해 주세요.")
        else:
            st.caption(f"학교: {school_env}  |  총 {len(env_df):,}행  |  측정 기간: ", )
            if "time" in env_df.columns:
                tmin = env_df["time"].min()
                tmax = env_df["time"].max()
                st.caption(f"{tmin.strftime('%Y-%m-%d')} ~ {tmax.strftime('%Y-%m-%d')}")

            # ── 환경 요약 표 ──────────────────────────────────────
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_cols:
                st.subheader("📋 환경 지표 요약 (평균 · 최소 · 최대)")

                env_summary = env_df[env_cols].dropna()
                if len(env_summary) >= 2:
                    agg = env_summary.agg(["mean", "min", "max"]).T
                    agg.columns = ["평균", "최소", "최대"]
                    unit_map = {"temperature": "°C", "humidity": "%", "ph": "", "ec": "mS/cm"}
                    agg.index = [f"{idx} ({unit_map.get(idx, '')})" for idx in agg.index]
                    st.dataframe(agg, width="stretch", hide_index=False, use_container_width=False)
                else:
                    st.info("환경 데이터가 충분하지 않아 요약 통계를 표시할 수 없습니다.")

            # ── 시계열 차트 ──────────────────────────────────────
            if "time" in env_df.columns and env_cols:
                st.subheader("📈 일별 평균 시계열")

                ts_df = env_df[["time"] + env_cols].copy()
                ts_df["day"] = ts_df["time"].dt.date
                daily = ts_df.groupby("day")[env_cols].mean().reset_index()
                daily["day"] = pd.to_datetime(daily["day"])

                n_daily_cols = len(env_cols)
                ts_fig = make_subplots(
                    rows=n_daily_cols, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=[f"{c} — 일별 평균" for c in env_cols],
                )
                colors_env = {"temperature": "#e74c3c", "humidity": "#3498db",
                              "ph": "#9b59b6", "ec": "#1abc9c"}
                for i, c in enumerate(env_cols):
                    ts_fig.add_trace(go.Scatter(
                        x=daily["day"], y=daily[c],
                        mode="lines",
                        line=dict(color=colors_env.get(c, "#3498db"), width=2),
                        name=c,
                        hovertemplate=f"{c}: %{{y:.2f}}<extra></extra>",
                    ), row=i + 1, col=1)

                ts_fig.update_layout(
                    height=200 * n_daily_cols + 60,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    margin=dict(t=60, l=40, r=20, b=40),
                    hovermode="x unified",
                )
                ts_fig.update_xaxes(title_text="날짜", row=n_daily_cols, col=1)
                st.plotly_chart(ts_fig, width="stretch")
            else:
                st.info("시간(time) 열 또는 환경 지표 열이 없어 시계열을 그릴 수 없습니다.")

            # ── expander: 원본 테이블 + 다운로드 ─────────────────
            with st.expander("📄 환경 데이터 원본 테이블 및 CSV 다운로드"):
                st.dataframe(env_df, width="stretch", hide_index=True)
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ 환경 데이터 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_env or '환경데이터'}_원본.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═══════════════════════════════════════════════════════════════
#  TAB 3 — 생육 결과
# ═══════════════════════════════════════════════════════════════
with tab3:
    try:
        st.header("📊 생육 결과")

        if growth_df.empty:
            st.info("📁 사이드바에서 생육 결과 CSV를 업로드해 주세요.")
        else:
            st.caption(f"학교: {school_gro}  |  총 {len(growth_df)}개체")

            # 선택 항목
            selected = growth_metrics  # sidebar에서 가져온 목록

            # ── 🥇 핵심 결과 카드 ────────────────────────────────
            st.subheader("🥇 핵심 결과")
            kcol1, kcol2, kcol3 = st.columns(3)

            with kcol1:
                if "생중량" in growth_df.columns:
                    w = growth_df["생중량"].dropna()
                    if len(w) >= 2:
                        mean_w = w.mean()
                        st.metric("평균 생중량", f"{mean_w:.2f} g")
                    else:
                        st.metric("평균 생중량", "데이터 부족")
                else:
                    st.metric("평균 생중량", "CSV에 없음")

            with kcol2:
                if "생중량" in growth_df.columns:
                    w = growth_df["생중량"].dropna()
                    if len(w) >= 2:
                        top_id = growth_df.loc[w.idxmax(), "개체번호"]
                        top_val = w.max()
                        st.metric("최대 생중량 개체", f"개체 {top_id} · {top_val:.2f}g")
                    else:
                        st.metric("최대 생중량 개체", "데이터 부족")
                else:
                    st.metric("최대 생중량 개체", "CSV에 없음")

            with kcol3:
                if "생중량" in growth_df.columns:
                    w = growth_df["생중량"].dropna()
                    if len(w) >= 2:
                        bot_id = growth_df.loc[w.idxmin(), "개체번호"]
                        bot_val = w.min()
                        st.metric("최소 생중량 개체", f"개체 {bot_id} · {bot_val:.2f}g")
                    else:
                        st.metric("최소 생중량 개체", "데이터 부족")
                else:
                    st.metric("최소 생중량 개체", "CSV에 없음")

            # ── 개체별 생육 비교 (막대 그래프) ──────────────────
            st.subheader("📊 개체별 생육 비교")

            gro_cols_map = {
                "생중량": "생중량",
                "잎 수": "잎 수",
                "지상부 길이": "지상부 길이",
                "지하부 길이": "지하부 길이",
            }
            unit_map = {
                "생중량": "g",
                "잎 수": "장",
                "지상부 길이": "mm",
                "지하부 길이": "mm",
            }

            n_selected = len(selected)
            if n_selected > 0:
                # subplot: 선택된 항목 수만큼 세로 쌓기
                bar_sub = make_subplots(
                    rows=n_selected, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.10,
                    subplot_titles=[f"평균 {label} (개체별)" for label in selected],
                )
                colors_bar = {"생중량": "#2ecc71", "잎 수": "#3498db",
                              "지상부 길이": "#e67e22", "지하부 길이": "#9b59b6"}

                for i, sel in enumerate(selected):
                    col_name = gro_cols_map.get(sel)
                    if col_name and col_name in growth_df.columns:
                        d = growth_df[["개체번호", col_name]].dropna().copy()
                        if d.empty:
                            continue
                        d = d.sort_values(col_name, ascending=False).reset_index(drop=True)

                        # 상위 25% / 하위 25% 강조
                        n_top = max(1, int(np.ceil(len(d) * 0.25)))
                        top_ids = set(d.head(n_top)["개체번호"].tolist())
                        bot_ids = set(d.tail(n_top)["개체번호"].tolist())
                        bar_colors = []
                        for _, row in d.iterrows():
                            if row["개체번호"] in top_ids:
                                bar_colors.append("#2ecc71")
                            elif row["개체번호"] in bot_ids:
                                bar_colors.append("#e74c3c")
                            else:
                                bar_colors.append(colors_bar.get(sel, "#3498db"))

                        bar_sub.add_trace(go.Bar(
                            x=d["개체번호"].astype(str),
                            y=d[col_name],
                            marker_color=bar_colors,
                            hovertemplate=f"{sel}: %{{y:.2f}}{unit_map.get(sel, '')}<extra></extra>",
                            name=sel,
                        ), row=i + 1, col=1)

                bar_sub.update_layout(
                    height=220 * n_selected + 60,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    margin=dict(t=60, l=40, r=20, b=40),
                    showlegend=False,
                    hovermode="x unified",
                )
                bar_sub.update_xaxes(title_text="개체번호", row=n_selected, col=1)
                st.plotly_chart(bar_sub, width="stretch")

                # 범례 안내
                st.caption("🟩 초록 막대 = 생중량 상위 25% 개체  |  🟥 빨간 막대 = 생중량 하위 25% 개체")

            # ── 생중량 분포 (박스플롯) ───────────────────────────
            if "생중량" in growth_df.columns:
                st.subheader("📦 생중량 분포")
                w_df = growth_df[["생중량"]].dropna()
                if len(w_df) >= 2:
                    bp_fig = go.Figure()
                    bp_fig.add_trace(go.Box(
                        y=w_df["생중량"],
                        name="생중량(g)",
                        fillcolor="#3498db",
                        line=dict(color="#2980b9", width=1.5),
                        boxpoints="outliers",
                        hovertemplate="생중량: %{{y:.2f}}g<extra></extra>",
                    ))
                    bp_fig.update_layout(
                        title="생중량 분포 (박스플롯) — 상자: 25~75 백분위, 선: 중앙값, 점: 이상치",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(t=40, l=40, r=20, b=40),
                        height=350,
                    )
                    st.plotly_chart(bp_fig, width="stretch")
                else:
                    st.info("생중량 데이터가 충분하지 않아 분포를 표시할 수 없습니다.")
            else:
                st.info("생중량 열이 없어 분포 그래프를 표시할 수 없습니다.")

            # ── 상관관계 분석 (산점도 + 추세선) ────────────────
            st.subheader("🔗 상관관계 분석 — 산점도 + 추세선")

            corr_pairs = [
                ("잎 수", "잎 수", "생중량", "생중량"),
                ("지상부 길이", "지상부 길이", "생중량", "생중량"),
            ]

            corr_results = []
            for x_label, x_col, y_label, y_col in corr_pairs:
                if x_col in growth_df.columns and y_col in growth_df.columns:
                    fig, r = scatter_with_trend(
                        growth_df[x_col].dropna(),
                        growth_df.loc[growth_df[x_col].dropna().index, y_col].dropna(),
                        x_label,
                        y_label,
                        f"{x_label} vs {y_label}",
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
                        # 상관계수 카드
                        rcol1, rcol2 = st.columns([1, 4])
                        with rcol1:
                            st.metric(f"{x_label}–{y_label} 상관계수(r)", f"{r:.3f}",
                                      help="+1에 가까울수록 함께 증가, -1에 가까울수록 함께 감소, 0에 가까우면 관계 약함")
                        with rcol2:
                            st.caption(
                                f"Pearson 상관계수 r = {r:.3f}  |  "
                                f"해석: ", )
                            if r > 0.7:
                                st.caption("강한 양의 상관 — 한쪽이 증가하면 다른 쪽도 뚜렷하게 증가")
                            elif r > 0.4:
                                st.caption("중간 정도의 양의 상관 — 함께 증가하는 경향이 있음")
                            elif r > 0.1:
                                st.caption("약한 양의 상관 — 약간의 함께 증가 경향")
                            elif r > -0.1:
                                st.caption("거의 상관 없음")
                            elif r > -0.4:
                                st.caption("약한 음의 상관")
                            elif r > -0.7:
                                st.caption("중간 정도의 음의 상관")
                            else:
                                st.caption("강한 음의 상관 — 한쪽이 증가하면 다른 쪽은 뚜렷하게 감소")
                        corr_results.append((x_label, y_label, r))
                    else:
                        st.info(f"{x_label} 또는 {y_label} 데이터가 부족해 상관관계를 표시할 수 없습니다.")
                else:
                    missing = []
                    if x_col not in growth_df.columns:
                        missing.append(x_label)
                    if y_col not in growth_df.columns:
                        missing.append(y_label)
                    st.info(f"{', '.join(missing)} 열이 없어 이 상관관계 분석을 건너뜁니다.")

            # ── 통계 요약 텍스트 ────────────────────────────────
            if corr_results:
                st.markdown("---")
                st.subheader("📐 분석 요약")
                for x_l, y_l, r_val in corr_results:
                    symbol = "▲" if r_val > 0 else "▼"
                    st.markdown(
                        f"- **{x_l} vs {y_l}**: 상관계수 r = {r_val:.3f} {symbol} "
                        f"({'강한' if abs(r_val) > 0.7 else '중간' if abs(r_val) > 0.4 else '약한' if abs(r_val) > 0.1 else '거의 없는'} "
                        f"{'양의' if r_val > 0 else '음의'} 상관)")

            # ── expander: 원본 테이블 + 다운로드 ─────────────────
            with st.expander("📄 생육 결과 데이터 원본 테이블 및 CSV 다운로드"):
                st.dataframe(growth_df, width="stretch", hide_index=True)
                csv_bytes_g = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ 생육 결과 CSV 다운로드",
                    data=csv_bytes_g,
                    file_name=f"{school_gro or '생육결과'}_원본.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
