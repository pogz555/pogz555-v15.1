import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 모바일 최적화 기본 설정 ---
st.set_page_config(page_title="퀀트 백테스트 v15.1", layout="wide", initial_sidebar_state="collapsed")
st.title("📈 SOXL 듀얼모드 백테스트 (v15.1)")

# --- 2. 입력 위젯 (모바일 친화적) ---
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
    agg_hold = st.number_input("최대보유일 (공세)", value=20, step=1, key="agg_hold_input")
    agg_buy = st.number_input("매수 (%) (공세)", value=10.0, step=0.1, format="%.1f", key="agg_buy_input")
    agg_sell = st.number_input("매도 (%) (공세)", value=15.0, step=0.1, format="%.1f", key="agg_sell_input")
else:
    st.info("현재 **안전 모드**의 변수를 설정 중입니다.")
    safe_hold = st.number_input("최대보유일 (안전)", value=10, step=1, key="safe_hold_input")
    safe_buy = st.number_input("매수 (%) (안전)", value=5.0, step=0.1, format="%.1f", key="safe_buy_input")
    safe_sell = st.number_input("매도 (%) (안전)", value=5.0, step=0.1, format="%.1f", key="safe_sell_input")

# --- 3. 데이터 엔진 (구글 시트 1:1 완벽 이식) ---
@st.cache_data(show_spinner=False)
def load_and_process_data(start, end):
    # 예열 기간을 포함하여 데이터 호출 (QQQ, SOXL)
    warmup_start = "2017-01-01"
    df_qqq = yf.download("QQQ", start=warmup_start, end=end + timedelta(days=7), interval="1d", auto_adjust=False)
    df_soxl = yf.download("SOXL", start=warmup_start, end=end + timedelta(days=7), interval="1d", auto_adjust=False)
    
    if isinstance(df_qqq.columns, pd.MultiIndex):
        df_qqq.columns = df_qqq.columns.droplevel(1)
        df_soxl.columns = df_soxl.columns.droplevel(1)
        
    df_qqq_fri = df_qqq.resample('W-FRI').last().dropna()
    df_soxl_fri = df_soxl.resample('W-FRI').last().dropna()
    
    df = pd.DataFrame(index=df_qqq_fri.index)
    df['QQQ_Close'] = df_qqq_fri['Close']
    df['SOXL_Close'] = df_soxl_fri['Close']
    
    # RSI 연산 (구글 시트 D~I열 반올림 로직 완벽 복제)
    N = 14
    delta = df['QQQ_Close'].diff()
    df['Gain'] = np.where(delta > 0, delta, 0)
    df['Loss'] = np.where(delta < 0, -delta, 0)
    
    df['Gain'] = pd.Series(df['Gain'], index=df.index)
    df['Loss'] = pd.Series(df['Loss'], index=df.index)
    
    df['Avg_Gain'] = df['Gain'].rolling(window=N).mean().round(4)
    df['Avg_Loss'] = df['Loss'].rolling(window=N).mean().round(4)
    
    df['RS'] = np.where(df['Avg_Loss'] == 0, 0, (df['Avg_Gain'] / df['Avg_Loss'])).round(3)
    df['RSI'] = np.where(df['Avg_Loss'] == 0, 100, np.round((df['RS'] / (1 + df['RS'])) * 100, 2))
    
    # 모드 판별 (K열 로직)
    df['I33'] = df['RSI'].shift(1)
    df['I32'] = df['RSI'].shift(2)
    
    def determine_mode(row):
        i33, i32 = row['I33'], row['I32']
        if pd.isna(i33) or pd.isna(i32): return None
        if (i32 > 65 and i32 > i33) or (40 < i32 < 50 and i32 > i33) or (i33 < 50 and 50 < i32): return "안전"
        if (i32 < 35 and i32 < i33) or (50 < i32 < 60 and i32 < i33) or (i33 > 50 and 50 > i32): return "공세"
        return "유지"

    df['신호'] = df.apply(determine_mode, axis=1)
    df['최종_모드'] = df['신호'].replace('유지', np.nan).ffill()
    
    # 사용자가 선택한 날짜 구간만 필터링
    df = df.loc[pd.to_datetime(start):pd.to_datetime(end)].copy()
    
    # [가상 자산 흐름] 시각화를 위한 임시 로직 (추후 Slot 연산으로 교체될 부분)
    df['SOXL_Return'] = df['SOXL_Close'].pct_change()
    df['Strategy_Return'] = np.where(df['최종_모드'] == '공세', df['SOXL_Return'] * 1.0, df['SOXL_Return'] * 0.3)
    df['Portfolio_Value'] = (1 + df['Strategy_Return'].fillna(0)).cumprod() * 10000
    
    rolling_max = df['Portfolio_Value'].cummax()
    df['Drawdown'] = (df['Portfolio_Value'] / rolling_max) - 1
    
    return df

# 실행 버튼
if st.button("🚀 백테스트 실행 (터치)", use_container_width=True):
    with st.spinner("데이터를 연산 중입니다..."):
        df = load_and_process_data(start_date, end_date)
        
        # --- 4. 요약 지표 출력 ---
        st.divider()
        st.markdown("### 📊 성과 요약 (가상 슬롯 적용 전)")
        
        total_return = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) - 1
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        cagr = ((1 + total_return) ** (365.25 / days)) - 1 if days > 0 else 0
        max_mdd = df['Drawdown'].min()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("총 수익률", f"{total_return * 100:.2f}%")
        m2.metric("CAGR", f"{cagr * 100:.2f}%")
        m3.metric("최대 낙폭 (MDD)", f"{max_mdd * 100:.2f}%")
        
        # --- 5. 시각화 ---
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=df.index, y=df['SOXL_Close'], name='기초자산 (SOXL)', yaxis='y1', line=dict(color='gray', width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Portfolio_Value'], name='전략 총자산', yaxis='y2', line=dict(color='orange', width=2.5)))
        
        df['모드_변화'] = (df['최종_모드'] != df['최종_모드'].shift(1)).cumsum()
        for _, group in df.groupby('모드_변화'):
            mode = group['최종_모드'].iloc[0]
            color = "rgba(255, 99, 71, 0.1)" if mode == "공세" else "rgba(135, 206, 235, 0.1)"
            fig.add_vrect(x0=group.index[0], x1=group.index[-1], fillcolor=color, layer="below", line_width=0)

        fig.update_layout(
            title="총 자산 변동 추이 (Log Scale)",
            xaxis=dict(title="Date"),
            yaxis=dict(title="SOXL 가격 ($)", type="log", side="left", showgrid=False),
            yaxis2=dict(title="총자산 ($)", type="log", side="right", overlaying="y", showgrid=True),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig_mdd = go.Figure()
        fig_mdd.add_trace(go.Scatter(x=df.index, y=df['Drawdown'] * 100, fill='tozeroy', name='MDD', line=dict(color='red')))
        fig_mdd.update_layout(title="낙폭 (MDD, %)", yaxis=dict(title="%", range=[df['Drawdown'].min()*100*1.1, 0]), margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_mdd, use_container_width=True)
