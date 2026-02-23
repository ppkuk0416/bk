import io
import re
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="법인카드 이상징후 스크리닝",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DATE_KEYWORDS     = ["사용일", "거래일", "승인일", "결제일", "일자", "날짜", "date"]
TIME_KEYWORDS     = ["사용시간", "거래시간", "승인시간", "시간", "time"]
AMOUNT_KEYWORDS   = ["승인금액", "사용금액", "거래금액", "결제금액", "금액", "amount"]
MERCHANT_KEYWORDS = ["가맹점명", "가맹점", "상호명", "상호", "업체명", "업체", "merchant"]
CATEGORY_KEYWORDS = ["업종명", "업종", "가맹점업종", "업태", "분류", "category"]
CARD_KEYWORDS     = ["카드번호", "카드번", "카드", "card"]
USER_KEYWORDS     = ["사용자명", "사용자", "카드소유자", "소유자", "성명", "이름", "사원명", "사원", "user"]
DEPT_KEYWORDS     = ["부서명", "부서", "팀명", "팀", "department", "dept"]

DEFAULT_SUSPICIOUS_KEYWORDS = [
    "유흥", "나이트", "클럽", "룸살롱", "단란주점", "유흥주점", "소주방",
    "노래방", "가라오케", "노래클럽",
    "골프", "골프장", "골프클럽",
    "카지노",
    "안마", "안마시술소",
    "마사지",
    "성인",
    "명품", "루이비통", "구찌", "에르메스", "샤넬", "프라다", "버버리", "몽클레어",
    "호스트바", "호프바",
]
FLAG_LABEL = {
    "주말_공휴일": "주말/공휴일",
    "심야_새벽":   "심야/새벽",
    "유흥_사치성": "유흥·사치성 업종",
    "반복거래":    "반복거래",
    "고액_거래":   "고액 거래",
    "분할_결제":   "분할결제",
}
# ─────────────────────────────────────────────────────────────────────────────
# Helpers: column auto-detection
# ─────────────────────────────────────────────────────────────────────────────
def find_best_column(columns: list[str], keywords: list[str]) -> str | None:
    lower_cols = [(c, c.lower().replace(" ", "")) for c in columns]
    for kw in keywords:
        kw_l = kw.lower().replace(" ", "")
        for col, col_l in lower_cols:
            if kw_l in col_l:
                return col
    return None

def auto_detect_columns(columns: list[str]) -> dict:
    return {
        "date":     find_best_column(columns, DATE_KEYWORDS),
        "time":     find_best_column(columns, TIME_KEYWORDS),
        "amount":   find_best_column(columns, AMOUNT_KEYWORDS),
        "merchant": find_best_column(columns, MERCHANT_KEYWORDS),
        "category": find_best_column(columns, CATEGORY_KEYWORDS),
        "card":     find_best_column(columns, CARD_KEYWORDS),
        "user":     find_best_column(columns, USER_KEYWORDS),
        "dept":     find_best_column(columns, DEPT_KEYWORDS),
    }

def col_index(options: list[str], value: str | None) -> int:
    if value and value in options:
        return options.index(value)
    return 0

def to_none(v: str) -> str | None:
    return v if v != "(사용 안함)" else None
# ─────────────────────────────────────────────────────────────────────────────
# Helpers: datetime parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_datetimes(df: pd.DataFrame, date_col: str, time_col: str | None) -> pd.Series:
    try:
        if time_col and time_col in df.columns:
            combined = df[date_col].astype(str) + " " + df[time_col].astype(str)
            return pd.to_datetime(combined, errors="coerce")
        return pd.to_datetime(df[date_col], errors="coerce")
    except Exception:
        return pd.Series([pd.NaT] * len(df), index=df.index)

def series_has_time(df: pd.DataFrame, date_col: str, time_col: str | None) -> bool:
    if time_col:
        return True
    try:
        sample = df[date_col].astype(str).dropna().head(20)
        return bool(sample.str.contains(r"[:\-]\d{2}:\d{2}", regex=True).any())
    except Exception:
        return False
# ─────────────────────────────────────────────────────────────────────────────
# Anomaly detectors
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_kr_holidays() -> set[str]:
    try:
        import holidays
        kr = holidays.KR(years=range(2015, 2031))
        return {str(d) for d in kr.keys()}
    except ImportError:
        st.warning("`holidays` 패키지를 설치하면 공휴일 탐지가 가능합니다.")
        return set()

def detect_weekend_holiday(datetimes: pd.Series, kr_holidays: set[str]):
    DOW = ["월", "화", "수", "목", "금", "토", "일"]
    flags, reasons = [], []
    for dt in datetimes:
        if pd.isna(dt):
            flags.append(False); reasons.append(""); continue
        dow = dt.dayofweek
        if dow >= 5:
            flags.append(True); reasons.append(f"주말({DOW[dow]}요일)")
        elif str(dt.date()) in kr_holidays:
            flags.append(True); reasons.append("공휴일")
        else:
            flags.append(False); reasons.append("")
    return flags, reasons

def detect_late_night(datetimes: pd.Series, start_h: int = 22, end_h: int = 6):
    flags, reasons = [], []
    for dt in datetimes:
        if pd.isna(dt):
            flags.append(False); reasons.append(""); continue
        h = dt.hour
        if h >= start_h or h < end_h:
            flags.append(True); reasons.append(f"심야/새벽({h:02d}:{dt.minute:02d})")
        else:
            flags.append(False); reasons.append("")
    return flags, reasons

def detect_suspicious(df: pd.DataFrame, merchant_col: str | None,
                      category_col: str | None, keywords: list[str]):
    flags, reasons = [], []
    for i in range(len(df)):
        found, reason = False, ""
        for col in (category_col, merchant_col):
            if found or col is None:
                break
            val = df[col].iloc[i]
            if pd.isna(val):
                continue
            val_str = str(val)
            for kw in keywords:
                if kw in val_str:
                    label = "업종" if col == category_col else "가맹점"
                    found, reason = True, f"{label}주의({kw})"
                    break
        flags.append(found); reasons.append(reason)
    return flags, reasons

def detect_repeat(df: pd.DataFrame, amount_col: str, merchant_col: str,
                  date_col: str, window_days: int = 7, min_count: int = 2):
    n = len(df)
    flags = [False] * n
    reasons = [""] * n
    try:
        work = df[[date_col, amount_col, merchant_col]].copy()
        work["_dt_"]    = pd.to_datetime(df[date_col], errors="coerce")
        work["_amt_"]   = df[amount_col].astype(str).str.replace(",", "").str.strip()
        work["_merch_"] = df[merchant_col].astype(str).str.strip()
        work["_pos_"]   = range(n)
        for (merch, amt), grp in work.groupby(["_merch_", "_amt_"]):
            if len(grp) < min_count or merch in ("nan", "") or amt in ("nan", "0", ""):
                continue
            valid = grp.dropna(subset=["_dt_"]).sort_values("_dt_")
            if len(valid) < min_count:
                continue
            dates = valid["_dt_"].tolist()
            flagged_rows: set[int] = set()
            for i in range(len(dates)):
                for j in range(i + 1, len(dates)):
                    if (dates[j] - dates[i]).days <= window_days:
                        flagged_rows.add(valid.index[i])
                        flagged_rows.add(valid.index[j])
            for idx in flagged_rows:
                pos = work.loc[idx, "_pos_"]
                flags[pos] = True
                reasons[pos] = f"반복거래({len(grp)}회/{window_days}일내)"
    except Exception:
        pass
    return flags, reasons

def detect_high_amount(df: pd.DataFrame, amount_col: str, threshold: int):
    flags, reasons = [], []
    for val in df[amount_col]:
        try:
            amt = float(str(val).replace(",", "").strip())
            if amt >= threshold:
                flags.append(True)
                reasons.append(f"고액거래({amt:,.0f}원)")
            else:
                flags.append(False)
                reasons.append("")
        except Exception:
            flags.append(False)
            reasons.append("")
    return flags, reasons

def detect_split_payment(df: pd.DataFrame, merchant_col: str, date_col: str,
                         min_count: int = 2):
    n = len(df)
    flags = [False] * n
    reasons = [""] * n
    try:
        work = df.copy()
        work["_date_only_"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        work["_merch_"]     = df[merchant_col].astype(str).str.strip()
        work["_pos_"]       = range(n)
        for (_, merch), grp in work.groupby(["_date_only_", "_merch_"]):
            if len(grp) < min_count or merch in ("nan", ""):
                continue
            for idx in grp.index:
                pos = work.loc[idx, "_pos_"]
                flags[pos] = True
                reasons[pos] = f"분할결제({len(grp)}회/동일일)"
    except Exception:
        pass
    return flags, reasons
# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🔍 법인카드 이상징후 스크리닝 시스템")
    st.caption("엑셀/CSV 파일을 업로드하면 자동으로 이상징후를 분석합니다.")

    with st.sidebar:
        st.header("⚙️ 탐지 설정")
        use_weekend = st.checkbox("주말/공휴일 사용 탐지", value=True)
        use_late_night = st.checkbox("심야/새벽 사용 탐지", value=True)
        if use_late_night:
            late_start = st.slider("심야 시작 시간 (시)", 18, 23, 22)
            late_end   = st.slider("새벽 종료 시간 (시)",  1,  9,  6)
        else:
            late_start, late_end = 22, 6

        use_suspicious = st.checkbox("유흥·사치성 업종 탐지", value=True)

        use_repeat = st.checkbox("반복거래 탐지", value=True)
        if use_repeat:
            repeat_window = st.slider("반복 탐지 기간 (일)", 1, 30, 7)
            repeat_min    = st.slider("반복 최소 횟수", 2, 5, 2)
        else:
            repeat_window, repeat_min = 7, 2

        use_high_amount = st.checkbox("고액 거래 탐지", value=False)
        if use_high_amount:
            high_amount_threshold = st.number_input(
                "기준 금액 (원) — 이 금액 이상을 탐지",
                min_value=0,
                value=300000,
                step=10000,
                format="%d",
            )
            st.caption(f"현재 기준: **{int(high_amount_threshold):,}원** 이상")
        else:
            high_amount_threshold = 300000

        use_split = st.checkbox("분할결제 탐지", value=True)
        if use_split:
            split_min = st.slider("동일일 동일가맹점 최소 횟수", 2, 5, 2)
        else:
            split_min = 2

        st.divider()
        st.subheader("🔑 추가 키워드")
        custom_kw_input = st.text_area(
            "추가 탐지 키워드 (줄바꿈 구분)",
            placeholder="예:\n뷔페\n리조트\n아울렛",
            height=100,
        )

    suspicious_keywords = DEFAULT_SUSPICIOUS_KEYWORDS.copy()
    if custom_kw_input:
        suspicious_keywords.extend(
            k.strip() for k in custom_kw_input.strip().splitlines() if k.strip()
        )

    st.header("1️⃣ 파일 업로드")
    uploaded = st.file_uploader(
        "법인카드 내역 파일을 업로드하세요",
        type=["xlsx", "xls", "csv"],
        help="Excel(.xlsx .xls) 또는 CSV 파일을 지원합니다.",
    )
    if uploaded is None:
        st.info("👆 파일을 업로드하면 분석이 시작됩니다.")
        with st.expander("📋 지원하는 샘플 데이터 형식"):
            sample = pd.DataFrame({
                "사용일자":  ["2024-01-13", "2024-01-14", "2024-01-20", "2024-01-20"],
                "사용시간":  ["10:30",      "23:15",      "14:20",      "14:20"],
                "가맹점명":  ["스타벅스",   "강남 룸살롱", "구내식당",   "구내식당"],
                "업종명":    ["카페",        "유흥주점",    "일반음식점", "일반음식점"],
                "승인금액":  [6500,         350000,        15000,        15000],
                "카드번호":  ["1234-****-****-5678"] * 4,
                "사용자명":  ["홍길동"] * 4,
            })
            st.dataframe(sample, use_container_width=True, hide_index=True)
            st.caption("※ 컬럼명은 다양한 형태를 자동 감지합니다.")
        return

    try:
        if uploaded.name.lower().endswith(".csv"):
            for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
                try:
                    df = pd.read_csv(uploaded, encoding=enc)
                    uploaded.seek(0)
                    break
                except UnicodeDecodeError:
                    uploaded.seek(0)
        else:
            xl = pd.ExcelFile(uploaded)
            sheet = (
                st.selectbox("시트 선택", xl.sheet_names)
                if len(xl.sheet_names) > 1
                else xl.sheet_names[0]
            )
            df = pd.read_excel(uploaded, sheet_name=sheet)
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    if df.empty:
        st.error("파일에 데이터가 없습니다.")
        return

    st.success(f"✅ 파일 로드 완료: 총 **{len(df):,}건** · **{len(df.columns)}개** 컬럼")
    with st.expander("📄 원본 데이터 미리보기 (상위 10행)"):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.header("2️⃣ 컬럼 매핑")
    auto = auto_detect_columns(df.columns.tolist())
    opts = ["(사용 안함)"] + df.columns.tolist()
    with st.expander("컬럼 매핑 확인 / 수정", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            sel_date     = st.selectbox("날짜 컬럼 *",    opts, index=col_index(opts, auto["date"]))
            sel_time     = st.selectbox("시간 컬럼",      opts, index=col_index(opts, auto["time"]))
            sel_amount   = st.selectbox("금액 컬럼",      opts, index=col_index(opts, auto["amount"]))
            sel_merchant = st.selectbox("가맹점명 컬럼",  opts, index=col_index(opts, auto["merchant"]))
        with c2:
            sel_category = st.selectbox("업종 컬럼",      opts, index=col_index(opts, auto["category"]))
            sel_card     = st.selectbox("카드번호 컬럼",  opts, index=col_index(opts, auto["card"]))
            sel_user     = st.selectbox("사용자 컬럼",    opts, index=col_index(opts, auto["user"]))
            sel_dept     = st.selectbox("부서 컬럼",      opts, index=col_index(opts, auto["dept"]))

    date_col     = to_none(sel_date)
    time_col     = to_none(sel_time)
    amount_col   = to_none(sel_amount)
    merchant_col = to_none(sel_merchant)
    category_col = to_none(sel_category)
    card_col     = to_none(sel_card)
    user_col     = to_none(sel_user)
    dept_col     = to_none(sel_dept)

    if not date_col:
        st.warning("날짜 컬럼을 선택해야 분석을 진행할 수 있습니다.")
        return

    st.header("3️⃣ 스크리닝 실행")
    if not st.button("🔍 이상징후 스크리닝 시작", type="primary", use_container_width=True):
        return

    progress = st.progress(0, text="분석 준비 중...")
    result = df.copy()
    datetimes = parse_datetimes(df, date_col, time_col)
    result["_dt_"] = datetimes
    flag_cols: list[str] = []

    if use_weekend:
        progress.progress(10, text="공휴일 데이터 로드 중...")
        kr_hols = load_kr_holidays()
        progress.progress(25, text="주말/공휴일 탐지 중...")
        f, r = detect_weekend_holiday(datetimes, kr_hols)
        result["주말_공휴일"] = f
        result["주말_공휴일_사유"] = r
        flag_cols.append("주말_공휴일")

    if use_late_night:
        progress.progress(40, text="심야/새벽 탐지 중...")
        if series_has_time(df, date_col, time_col):
            f, r = detect_late_night(datetimes, late_start, late_end)
            result["심야_새벽"] = f
            result["심야_새벽_사유"] = r
            flag_cols.append("심야_새벽")
        else:
            st.info("시간 정보가 없어 심야/새벽 탐지를 건너뜁니다.")

    if use_suspicious and (merchant_col or category_col):
        progress.progress(60, text="유흥·사치성 업종 탐지 중...")
        f, r = detect_suspicious(df, merchant_col, category_col, suspicious_keywords)
        result["유흥_사치성"] = f
        result["유흥_사치성_사유"] = r
        flag_cols.append("유흥_사치성")

    if use_repeat and merchant_col and amount_col:
        progress.progress(75, text="반복거래 탐지 중...")
        f, r = detect_repeat(df, amount_col, merchant_col, date_col, repeat_window, repeat_min)
        result["반복거래"] = f
        result["반복거래_사유"] = r
        flag_cols.append("반복거래")

    if use_high_amount and amount_col:
        progress.progress(85, text="고액 거래 탐지 중...")
        f, r = detect_high_amount(df, amount_col, int(high_amount_threshold))
        result["고액_거래"] = f
        result["고액_거래_사유"] = r
        flag_cols.append("고액_거래")

    if use_split and merchant_col:
        progress.progress(88, text="분할결제 탐지 중...")
        f, r = detect_split_payment(df, merchant_col, date_col, split_min)
        result["분할_결제"] = f
        result["분할_결제_사유"] = r
        flag_cols.append("분할_결제")

    progress.progress(90, text="결과 집계 중...")
    result["위험점수"] = result[flag_cols].sum(axis=1).astype(int)
    result["위험등급"] = result["위험점수"].map(
        lambda s: "🔴 위험" if s >= 2 else ("🟡 주의" if s == 1 else "🟢 정상")
    )
    reason_cols = [c for c in result.columns if c.endswith("_사유")]
    result["이상사유"] = result[reason_cols].apply(
        lambda row: " | ".join(v for v in row if v and str(v) not in ("", "nan")),
        axis=1,
    )
    progress.progress(100, text="완료!")
    progress.empty()

    st.header("4️⃣ 분석 결과")
    total     = len(result)
    flagged   = int((result["위험점수"] > 0).sum())
    high_risk = int((result["위험점수"] >= 2).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("총 거래건수",   f"{total:,}건")
    m2.metric("이상 의심 거래", f"{flagged:,}건", f"{flagged/total*100:.1f}%")
    m3.metric("고위험 거래",   f"{high_risk:,}건")
    if amount_col:
        try:
            amt = pd.to_numeric(
                result[amount_col].astype(str).str.replace(",", ""), errors="coerce"
            )
            flagged_amt = amt[result["위험점수"] > 0].sum()
            m4.metric("이상 의심 금액합계", f"{flagged_amt:,.0f}원")
        except Exception:
            m4.metric("이상 의심 금액합계", "-")

    if flag_cols:
        chart1, chart2 = st.columns(2)
        with chart1:
            cnt_data = pd.DataFrame({
                "항목":  [FLAG_LABEL.get(c, c) for c in flag_cols],
                "건수":  [int(result[c].sum()) for c in flag_cols],
            })
            fig1 = px.bar(
                cnt_data, x="항목", y="건수",
                title="이상징후 유형별 건수",
                color="건수", color_continuous_scale="Reds",
                text="건수",
            )
            fig1.update_traces(textposition="outside")
            fig1.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig1, use_container_width=True)
        with chart2:
            risk_cnt = result["위험등급"].value_counts().reset_index()
            risk_cnt.columns = ["등급", "건수"]
            color_map = {"🔴 위험": "#e74c3c", "🟡 주의": "#f39c12", "🟢 정상": "#2ecc71"}
            fig2 = px.pie(
                risk_cnt, names="등급", values="건수",
                title="위험등급 분포",
                color="등급", color_discrete_map=color_map,
            )
            st.plotly_chart(fig2, use_container_width=True)

    if user_col:
        st.subheader("👤 사용자별 현황")
        user_stats = (
            result.groupby(user_col)
            .agg(
                총거래건수=(date_col, "count"),
                이상건수=("위험점수", lambda x: (x > 0).sum()),
                고위험건수=("위험점수", lambda x: (x >= 2).sum()),
            )
            .reset_index()
        )
        user_stats["이상율(%)"] = (
            user_stats["이상건수"] / user_stats["총거래건수"] * 100
        ).round(1)
        if amount_col:
            try:
                amt_s = pd.to_numeric(
                    result[amount_col].astype(str).str.replace(",", ""), errors="coerce"
                )
                result["_amt_num_"] = amt_s
                user_amt = (
                    result[result["위험점수"] > 0]
                    .groupby(user_col)["_amt_num_"]
                    .sum()
                    .reset_index()
                    .rename(columns={"_amt_num_": "이상금액합계"})
                )
                user_stats = user_stats.merge(user_amt, on=user_col, how="left")
                user_stats["이상금액합계"] = user_stats["이상금액합계"].fillna(0).astype(int)
            except Exception:
                pass
        user_stats = user_stats.sort_values("이상건수", ascending=False)
        col_cfg = {}
        if "이상금액합계" in user_stats.columns:
            col_cfg["이상금액합계"] = st.column_config.NumberColumn(
                "이상금액합계 (원)", format=",.0f"
            )
        st.dataframe(user_stats, use_container_width=True, hide_index=True,
                     column_config=col_cfg if col_cfg else None)

    if dept_col:
        st.subheader("🏢 부서별 현황")
        dept_stats = (
            result.groupby(dept_col)
            .agg(
                총거래건수=(date_col, "count"),
                이상건수=("위험점수", lambda x: (x > 0).sum()),
                고위험건수=("위험점수", lambda x: (x >= 2).sum()),
            )
            .reset_index()
        )
        dept_stats["이상율(%)"] = (
            dept_stats["이상건수"] / dept_stats["총거래건수"] * 100
        ).round(1)
        if amount_col:
            try:
                amt_s = pd.to_numeric(
                    result[amount_col].astype(str).str.replace(",", ""), errors="coerce"
                )
                result["_amt_num_"] = amt_s
                dept_amt = (
                    result[result["위험점수"] > 0]
                    .groupby(dept_col)["_amt_num_"]
                    .sum()
                    .reset_index()
                    .rename(columns={"_amt_num_": "이상금액합계"})
                )
                dept_stats = dept_stats.merge(dept_amt, on=dept_col, how="left")
                dept_stats["이상금액합계"] = dept_stats["이상금액합계"].fillna(0).astype(int)
            except Exception:
                pass
        dept_stats = dept_stats.sort_values("이상건수", ascending=False)
        dept_cfg = {}
        if "이상금액합계" in dept_stats.columns:
            dept_cfg["이상금액합계"] = st.column_config.NumberColumn(
                "이상금액합계 (원)", format=",.0f"
            )
        st.dataframe(dept_stats, use_container_width=True, hide_index=True,
                     column_config=dept_cfg if dept_cfg else None)

    st.subheader("📋 상세 결과")
    min_dt = datetimes.dropna().dt.date.min() if datetimes.notna().any() else None
    max_dt = datetimes.dropna().dt.date.max() if datetimes.notna().any() else None
    if min_dt and max_dt and min_dt != max_dt:
        date_range = st.date_input(
            "📅 기간 필터",
            value=(min_dt, max_dt),
            min_value=min_dt,
            max_value=max_dt,
        )
    else:
        date_range = None

    fa, fb = st.columns([1, 2])
    with fa:
        show_filter = st.selectbox(
            "표시 범위",
            ["전체", "이상 의심만 (주의+위험)", "고위험만 (🔴 위험)"],
        )
    with fb:
        type_opts = [FLAG_LABEL.get(c, c) for c in flag_cols]
        type_filter = st.multiselect("이상징후 유형 필터", options=type_opts)

    display = result.copy()
    if date_range and len(date_range) == 2:
        display = display[
            (display["_dt_"].dt.date >= date_range[0]) &
            (display["_dt_"].dt.date <= date_range[1])
        ]
    if show_filter == "이상 의심만 (주의+위험)":
        display = display[display["위험점수"] > 0]
    elif show_filter == "고위험만 (🔴 위험)":
        display = display[display["위험점수"] >= 2]

    if type_filter:
        rev_map = {v: k for k, v in FLAG_LABEL.items()}
        tf_cols = [rev_map.get(t, t) for t in type_filter if rev_map.get(t, t) in display.columns]
        if tf_cols:
            display = display[display[tf_cols].any(axis=1)]

    show_cols = ["위험등급", "이상사유"]
    for c in [date_col, time_col, user_col, dept_col, card_col,
              merchant_col, category_col, amount_col]:
        if c:
            show_cols.append(c)
    show_cols.append("위험점수")
    show_cols = [c for c in show_cols if c in display.columns]

    def row_style(row):
        s = row["위험점수"] if "위험점수" in row.index else 0
        if s >= 2:
            return ["background-color: #fde8e8"] * len(row)
        if s == 1:
            return ["background-color: #fef9e7"] * len(row)
        return [""] * len(row)

    fmt = {}
    if amount_col and amount_col in show_cols:
        fmt[amount_col] = lambda x: (
            f"{float(str(x).replace(',', '')):,.0f}"
            if str(x) not in ("nan", "") else "-"
        )

    st.caption(f"표시 건수: {len(display):,}건")
    styled = display[show_cols].style.apply(row_style, axis=1)
    if fmt:
        styled = styled.format(fmt, na_rep="-")
    st.dataframe(styled, use_container_width=True, height=420, hide_index=True)

    st.subheader("📥 결과 다운로드")
    export = result.drop(columns=["_dt_", "_amt_num_"], errors="ignore")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name="전체결과", index=False)
        export[export["위험점수"] > 0].to_excel(writer, sheet_name="이상의심", index=False)
        export[export["위험점수"] >= 2].to_excel(writer, sheet_name="고위험", index=False)
        summary = pd.DataFrame({
            "구분": ["총 거래건수", "이상 의심 건수", "고위험 건수", "이상 비율(%)"],
            "값":   [total, flagged, high_risk, f"{flagged/total*100:.1f}%"],
        })
        summary.to_excel(writer, sheet_name="요약", index=False)
    buf.seek(0)
    st.download_button(
        label="📥 분석 결과 엑셀 다운로드",
        data=buf.getvalue(),
        file_name=f"법인카드_이상징후_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

if __name__ == "__main__":
    main()
