import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 모바일 최적화 기본 설정 ---
st.set_page_config(page_title="퀀트 백테스트 v15.1", layout="wide", initial_sidebar_state="collapsed")
st.title("📈 SOXL 듀얼모드 백테스트")

# --- 2. 입력 위젯 (모바일 친화적 상하 배치 및 1회 터치 탭) ---
st.markdown("### ⚙️ 기본 설정")
col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("시작일", datetime(2021, 1, 1))
with col2:
    end_date = st.date_input("종료일", datetime(2024, 4, 1))
with col3:
    slot_count = st.number_input("Slot 수", value=10, step=1)

st.markdown("### 🎛️ 모드별 세부 설정")
mode_tab = st.radio("설정 모드 선택", ["🔴 공세 모드", "🔵 안전 모드"], horizontal=True, label_visibility="collapsed")

if mode_tab == "🔴 공세 모드":
    st.info("현재 **공세 모드**의 변수를 설정 중입니다.")
    agg_hold = st.number_input("최대보유일 (공세)", value=20, step=1)
    agg_buy = st.number_input("매수 (%) (공세)", value=10.0, step=0.1, format="%.1f")
    agg_sell = st.number_input("매도 (%) (공세)", value=15.0, step=0.1, format="%.1f")
else:
    st.info("현재 **안전 모드**의 변수를 설정 중입니다.")
    safe_hold = st.number_input("최대보유일 (안전)", value=10, step=1)
    safe_buy = st.number_input("매수 (%) (안전)", value=5.0, step=0.1, format="%.1f")
    safe_sell = st.number_input("매도 (%) (안전)", value=5.0, step=0.1, format="%.1f")

# --- 3. 데이터 수집 및 엔진 로직 (캐싱하여 속도 향상) ---
@st.cache_data
def load_and_process_data(start, end):
    df = yf.download("SOXL", start=start, end=end, interval="1wk")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df['RSI'] = df.ta.rsi(close='Close', length=14, mamode='rma')
    df['RSI_1w_ago'] = df['RSI'].shift(1)
    df['RSI_2w_ago'] = df['RSI'].shift(2)
    
    def determine_mode(row):
        curr, prev = row['RSI_1w_ago'], row['RSI_2w_ago']
        if pd.isna(curr) or pd.isna(prev): return None
        if (curr > 65 and curr < prev) or (40 < curr < 50 and curr < prev) or (prev >= 50 and curr < 50): return "안전"
        if (prev <= 50 and curr > 50) or (50 < curr < 60 and curr > prev) or (curr < 35 and curr > prev): return "공세"
        return "유지"

    df['신호'] = df.apply(determine_mode, axis=1)
    df['최종_모드'] = df['신호'].replace('유지', np.nan).ffill()
    
    # 임시 자산 데이터 생성 (향후 Slot 로직이 들어갈 자리)
    # 현재는 차트 시각화를 위해 SOXL 수익률을 약간 변형한 가상 자산 흐름을 만듭니다.
    df['Daily_Return'] = df['Close'].pct_change()
    # 공세 모드일 때는 수익/손실 증폭, 안전 모드일 때는 축소 (가상 로직)
    df['Strategy_Return'] = np.where(df['최종_모드'] == '공세', df['Daily_Return'] * 1.2, df['Daily_Return'] * 0.5)
    df['Portfolio_Value'] = (1 + df['Strategy_Return'].fillna(0)).cumprod() * 10000 # 초기자본 1만불
    
    # MDD 계산
    rolling_max = df['Portfolio_Value'].cummax()
    df['Drawdown'] = (df['Portfolio_Value'] / rolling_max) - 1
    
    return df.dropna(subset=['최종_모드'])

# 실행 버튼
if st.button("🚀 백테스트 실행 (터치)", use_container_width=True):
    with st.spinner("데이터를 연산 중입니다..."):
        df = load_and_process_data(start_date, end_date)
        
        # --- 4. 요약 지표 출력 ---
        st.divider()
        st.markdown("### 📊 성과 요약")
        
        total_return = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) - 1
        days = (end_date - start_date).days
        cagr = ((1 + total_return) ** (365.25 / days)) - 1 if days > 0 else 0
        max_mdd = df['Drawdown'].min()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("총 수익률", f"{total_return * 100:.2f}%")
        m2.metric("CAGR", f"{cagr * 100:.2f}%")
        m3.metric("최대 낙폭 (MDD)", f"{max_mdd * 100:.2f}%")
        
        # --- 5. 시각화 (이중 Y축, 로그 스케일, 배경색) ---
        fig = go.Figure()
        
        # 기초 자산 (좌측 축)
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='기초자산 (SOXL)', yaxis='y1', line=dict(color='gray', width=1.5)))
        # 총 자산 (우측 축)
        fig.add_trace(go.Scatter(x=df.index, y=df['Portfolio_Value'], name='전략 총자산', yaxis='y2', line=dict(color='orange', width=2.5)))
        
        # 배경색 칠하기 (공세=연홍색, 안전=연청색)
        # 연속된 모드 구간을 찾아 사각형(vrect)으로 그림
        df['모드_변화'] = (df['최종_모드'] != df['최종_모드'].shift(1)).cumsum()
        for _, group in df.groupby('모드_변화'):
            mode = group['최종_모드'].iloc[0]
            color = "rgba(255, 99, 71, 0.1)" if mode == "공세" else "rgba(135, 206, 235, 0.1)"
            fig.add_vrect(x0=group.index[0], x1=group.index[-1], fillcolor=color, layer="below", line_width=0)

        # 차트 레이아웃 설정 (로그 스케일 적용)
        fig.update_layout(
            title="총 자산 변동 추이 (Log Scale)",
            xaxis=dict(title="Date"),
            yaxis=dict(title="기초자산 ($)", type="log", side="left", showgrid=False),
            yaxis2=dict(title="총자산 ($)", type="log", side="right", overlaying="y", showgrid=True),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), # 모바일용 하단 범례
            margin=dict(l=10, r=10, t=40, b=10) # 모바일 화면 꽉 차게 마진 최소화
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # MDD 차트
        fig_mdd = go.Figure()
        fig_mdd.add_trace(go.Scatter(x=df.index, y=df['Drawdown'] * 100, fill='tozeroy', name='MDD', line=dict(color='red')))
        fig_mdd.update_layout(title="낙폭 (MDD, %)", yaxis=dict(title="%", range=[df['Drawdown'].min()*100*1.1, 0]), margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_mdd, use_container_width=True)
