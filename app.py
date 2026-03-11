import io
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
from openpyxl.styles import Font, PatternFill

APP_VERSION = "v2.2"
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
# 더존 iUERP 컬럼명을 우선순위 1순위로, 기타 일반 컬럼명 후순위
DATE_KEYWORDS          = ["승인일자", "사용일", "거래일", "결제일", "일자", "날짜", "date"]
TIME_KEYWORDS          = ["승인시간", "사용시간", "거래시간", "시간", "time"]
AMOUNT_KEYWORDS        = ["승인금액", "사용금액", "거래금액", "결제금액", "금액", "amount"]
MERCHANT_KEYWORDS      = ["가맹점명", "가맹점", "상호명", "상호", "업체명", "업체", "merchant"]
CATEGORY_KEYWORDS      = ["업종명", "업종", "가맹점업종", "업태", "분류", "category"]
CARD_KEYWORDS          = ["법인카드", "카드번호", "카드번", "카드", "card"]
USER_KEYWORDS          = ["소유자", "사용자명", "사용자", "카드소유자", "성명", "이름", "사원명", "사원", "user"]
DEPT_KEYWORDS          = ["관리부서", "부서명", "부서", "팀명", "팀", "department", "dept"]
APPROVAL_TYPE_KEYWORDS = ["구분"]           # 승인 / 취소 구분
BIZ_REG_KEYWORDS       = ["사업자등록번호", "사업자번호", "등록번호"]
SUPPLY_AMT_KEYWORDS    = ["공급가액"]
VAT_KEYWORDS           = ["부가세"]
VAT_TYPE_KEYWORDS      = ["부가세구분"]
VAT_YN_KEYWORDS        = ["부가세여부"]
SERVICE_FEE_KEYWORDS   = ["봉사료"]
APPROVAL_NO_KEYWORDS   = ["승인번호"]
ACCOUNT_CODE_KEYWORDS  = ["계정코드"]
ACCOUNT_NAME_KEYWORDS  = ["상대계정명", "계정명"]
BUDGET_CONTROL_KEYWORDS= ["통제성예산"]
COST_CENTER_KEYWORDS   = ["코스트센터명", "코스트센터", "cost center"]
BUDGET_UNIT_KEYWORDS   = ["예산단위"]
MEMO_KEYWORDS          = ["적요"]
BIZ_CAR_KEYWORDS       = ["업무용차량"]
SLIP_STATUS_KEYWORDS   = ["전표처리", "전표상태", "처리여부"]

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
    "주말_공휴일":    "주말/공휴일",
    "심야_새벽":      "심야/새벽",
    "유흥_사치성":    "유흥·사치성 업종",
    "고액_거래":      "고액 거래",
    "분할_결제":      "분할결제",
    "전표_미처리":    "전표 미처리",
    "월한도_초과":    "월 한도 초과",
}
# ─────────────────────────────────────────────────────────────────────────────
# Helpers: column auto-detection
# ─────────────────────────────────────────────────────────────────────────────
def find_best_column(columns: list[str], keywords: list[str]) -> str | None:
    lower_cols = [(c, c.lower().replace(" ", "")) for c in columns]
    # 1순위: 정확 일치
    for kw in keywords:
        kw_l = kw.lower().replace(" ", "")
        for col, col_l in lower_cols:
            if kw_l == col_l:
                return col
    # 2순위: 부분 일치
    for kw in keywords:
        kw_l = kw.lower().replace(" ", "")
        for col, col_l in lower_cols:
            if kw_l in col_l:
                return col
    return None

def auto_detect_columns(columns: list[str]) -> dict:
    return {
        "date":           find_best_column(columns, DATE_KEYWORDS),
        "time":           find_best_column(columns, TIME_KEYWORDS),
        "amount":         find_best_column(columns, AMOUNT_KEYWORDS),
        "merchant":       find_best_column(columns, MERCHANT_KEYWORDS),
        "category":       find_best_column(columns, CATEGORY_KEYWORDS),
        "card":           find_best_column(columns, CARD_KEYWORDS),
        "user":           find_best_column(columns, USER_KEYWORDS),
        "dept":           find_best_column(columns, DEPT_KEYWORDS),
        # 더존 iUERP 전용 컬럼
        "approval_type":  find_best_column(columns, APPROVAL_TYPE_KEYWORDS),
        "biz_reg":        find_best_column(columns, BIZ_REG_KEYWORDS),
        "supply_amt":     find_best_column(columns, SUPPLY_AMT_KEYWORDS),
        "vat":            find_best_column(columns, VAT_KEYWORDS),
        "vat_type":       find_best_column(columns, VAT_TYPE_KEYWORDS),
        "vat_yn":         find_best_column(columns, VAT_YN_KEYWORDS),
        "service_fee":    find_best_column(columns, SERVICE_FEE_KEYWORDS),
        "approval_no":    find_best_column(columns, APPROVAL_NO_KEYWORDS),
        "account_code":   find_best_column(columns, ACCOUNT_CODE_KEYWORDS),
        "account_name":   find_best_column(columns, ACCOUNT_NAME_KEYWORDS),
        "budget_control": find_best_column(columns, BUDGET_CONTROL_KEYWORDS),
        "cost_center":    find_best_column(columns, COST_CENTER_KEYWORDS),
        "budget_unit":    find_best_column(columns, BUDGET_UNIT_KEYWORDS),
        "memo":           find_best_column(columns, MEMO_KEYWORDS),
        "biz_car":        find_best_column(columns, BIZ_CAR_KEYWORDS),
        "slip_status":    find_best_column(columns, SLIP_STATUS_KEYWORDS),
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
    valid    = datetimes.notna()
    dow      = datetimes.dt.dayofweek
    date_str = datetimes.dt.date.astype(str)
    is_weekend = valid & (dow >= 5)
    is_holiday = valid & ~is_weekend & date_str.isin(kr_holidays)
    reasons = pd.Series("", index=datetimes.index, dtype=str)
    for d in (5, 6):
        m = is_weekend & (dow == d)
        if m.any():
            reasons[m] = f"주말({DOW[d]}요일)"
    reasons[is_holiday] = "공휴일"
    return (is_weekend | is_holiday).tolist(), reasons.tolist()

def detect_late_night(datetimes: pd.Series, start_h: int = 22, end_h: int = 6):
    valid   = datetimes.notna()
    h       = datetimes.dt.hour
    is_late = valid & ((h >= start_h) | (h < end_h))
    reasons = pd.Series("", index=datetimes.index, dtype=str)
    reasons[is_late] = datetimes[is_late].dt.strftime("심야/새벽(%H:%M)")
    return is_late.tolist(), reasons.tolist()

def detect_suspicious(df: pd.DataFrame, merchant_col: str | None,
                      category_col: str | None, keywords: list[str]):
    if not keywords:
        return [False] * len(df), [""] * len(df)
    pattern = "|".join(re.escape(k) for k in keywords)
    flags   = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index, dtype=str)
    for col, label in [(category_col, "업종"), (merchant_col, "가맹점")]:
        if col is None:
            continue
        matched = df[col].astype(str).str.extract(f"({pattern})", expand=False)
        new_hit = matched.notna() & ~flags
        reasons[new_hit] = label + "주의(" + matched[new_hit] + ")"
        flags |= matched.notna()
    return flags.tolist(), reasons.tolist()

def detect_high_amount(df: pd.DataFrame, amount_col: str, threshold: int):
    amt     = pd.to_numeric(df[amount_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    flags   = amt >= threshold
    reasons = pd.Series("", index=df.index, dtype=str)
    reasons[flags] = amt[flags].apply(lambda x: f"고액거래({x:,.0f}원)")
    return flags.tolist(), reasons.tolist()


def detect_slip_unprocessed(df: pd.DataFrame, slip_col: str):
    """전표처리 컬럼이 '미처리'인 행 탐지"""
    status  = df[slip_col].astype(str).str.strip()
    flags   = status == "미처리"
    reasons = pd.Series("", index=df.index, dtype=str)
    reasons[flags] = "전표미처리"
    return flags.tolist(), reasons.tolist()

def detect_monthly_limit(df: pd.DataFrame, amount_col: str, user_col: str,
                          date_col: str, monthly_limit: int):
    """인당 월별 승인금액 합계가 기준을 초과하는 거래 탐지"""
    flags   = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index, dtype=str)
    try:
        amt  = pd.to_numeric(df[amount_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
        dt   = pd.to_datetime(df[date_col], errors="coerce")
        ym   = dt.dt.to_period("M").astype(str)
        user = df[user_col].astype(str).str.strip()
        work = pd.DataFrame({"_user_": user, "_ym_": ym, "_amt_": amt})
        monthly_sum = work.groupby(["_user_", "_ym_"])["_amt_"].transform("sum")
        flags   = (monthly_sum >= monthly_limit) & dt.notna()
        reasons[flags] = (
            user[flags] + "/" + ym[flags] + " 월합계("
            + monthly_sum[flags].apply(lambda x: f"{x:,.0f}원") + ")"
        )
    except Exception:
        pass
    return flags.tolist(), reasons.tolist()

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
    st.title(f"🔍 법인카드 이상징후 스크리닝 시스템  `{APP_VERSION}`")
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
        st.subheader("🏢 추가 탐지 (iUERP 전용)")
        use_slip = st.checkbox(
            "전표 미처리 탐지",
            value=True,
            help="전표처리 컬럼이 '미처리'인 거래를 탐지합니다.",
        )
        use_monthly_limit = st.checkbox("인당 월 한도 초과 탐지", value=False)
        if use_monthly_limit:
            monthly_limit = st.number_input(
                "월 한도 기준 (원)",
                min_value=0,
                value=500000,
                step=100000,
                format="%d",
            )
            st.caption(f"인당 월 합계 **{int(monthly_limit):,}원** 초과 시 탐지")
        else:
            monthly_limit = 500000

        st.divider()
        st.subheader("🏦 더존 iUERP 옵션")
        exclude_cancel = st.checkbox(
            "취소 거래 제외 (구분='취소')",
            value=True,
            help="'구분' 컬럼이 있을 때 취소 거래를 분석에서 제외합니다.",
        )

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
        with st.expander("📋 더존 iUERP 샘플 데이터 형식"):
            sample = pd.DataFrame({
                "법인카드":       ["42890(재무회계계정)", "42890(재무회계계정)", "55100(영업팀)", "55100(영업팀)"],
                "관리부서":       ["재무회계팀",          "재무회계팀",          "영업1팀",       "영업1팀"],
                "소유자":         ["홍길동",              "홍길동",              "김영희",        "김영희"],
                "승인일자":       ["2024/01/13 10:30:00", "2024/01/14 23:15:00", "2024/01/20 14:20:00", "2024/01/20 14:45:00"],
                "가맹점":         ["스타벅스",            "강남 룸살롱",         "구내식당",      "구내식당"],
                "업종":           ["카페",                "유흥주점",            "일반음식점",    "일반음식점"],
                "승인금액":       [6500,                  350000,                15000,           15000],
                "사업자등록번호": ["123-45-67890",        "234-56-78901",        "345-67-89012",  "345-67-89012"],
                "공급가액":       [5909,                  318182,                13636,           13636],
                "부가세":         [591,                   31818,                 1364,            1364],
                "봉사료":         [0,                     0,                     0,               0],
                "승인번호":       ["11813366",            "22924477",            "33035588",      "44146699"],
                "구분":           ["승인",                "승인",                "승인",          "승인"],
                "상대계정명":     ["재무회계팀",          "재무회계팀",          "영업1팀",       "영업1팀"],
                "코스트센터명":   ["본사",                "본사",                "영업본부",      "영업본부"],
            })
            st.dataframe(sample, use_container_width=True, hide_index=True)
            st.caption("※ 더존 iUERP 내보내기 형식 기준이며, 컬럼명은 자동으로 감지됩니다.")
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

    # 더존 iU ERP 정확한 컬럼명 프리셋
    _DOOJOHN_PRESET = {
        "sel_date": "승인일자", "sel_time": "(사용 안함)",
        "sel_amount": "승인금액", "sel_merchant": "가맹점",
        "sel_category": "업종", "sel_card": "법인카드",
        "sel_user": "소유자", "sel_dept": "관리부서",
        "sel_approval_type": "구분", "sel_approval_no": "승인번호",
        "sel_biz_reg": "사업자등록번호", "sel_supply_amt": "공급가액",
        "sel_vat": "부가세", "sel_vat_type": "부가세구분", "sel_vat_yn": "부가세여부",
        "sel_service_fee": "봉사료",
        "sel_account_code": "계정코드", "sel_account_name": "계정명",
        "sel_budget_control": "통제성예산",
        "sel_cost_center": "코스트센터명", "sel_budget_unit": "예산단위",
        "sel_memo": "적요", "sel_biz_car": "업무용차량",
        "sel_slip_status": "전표처리",
    }

    # 매핑 상태 뱃지
    detected = sum(1 for v in auto.values() if v)
    total = len(auto)
    _pct = detected / total
    _badge_cols = st.columns([3, 1])
    with _badge_cols[0]:
        if _pct >= 0.8:
            st.success(f"더존 iU 형식 감지됨 · {detected}/{total} 컬럼 자동 인식")
        elif _pct >= 0.4:
            st.warning(f"일부 컬럼 미감지 · {detected}/{total} 인식 · 아래에서 확인하세요")
        else:
            st.error(f"컬럼 자동 감지 실패 · {detected}/{total} · 수동 설정 필요")
    with _badge_cols[1]:
        if st.button("더존 iU 기본형식 적용", use_container_width=True):
            for _k, _v in _DOOJOHN_PRESET.items():
                if _v in opts:
                    st.session_state[_k] = _v
            st.rerun()

    def _sel_val(key, auto_val):
        """session_state에 preset이 있으면 우선 사용, 없으면 auto_detect 값 사용"""
        return st.session_state.get(key, auto_val)

    with st.expander("컬럼 매핑 확인 / 수정", expanded=(_pct < 0.8)):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**기본 컬럼**")
            sel_date     = st.selectbox("날짜 컬럼 *",    opts, key="sel_date",          index=col_index(opts, _sel_val("sel_date",          auto["date"])))
            sel_time     = st.selectbox("시간 컬럼",      opts, key="sel_time",          index=col_index(opts, _sel_val("sel_time",          auto["time"])))
            sel_amount   = st.selectbox("승인금액 컬럼",  opts, key="sel_amount",        index=col_index(opts, _sel_val("sel_amount",        auto["amount"])))
            sel_merchant = st.selectbox("가맹점 컬럼",    opts, key="sel_merchant",      index=col_index(opts, _sel_val("sel_merchant",      auto["merchant"])))
            sel_category = st.selectbox("업종 컬럼",      opts, key="sel_category",      index=col_index(opts, _sel_val("sel_category",      auto["category"])))
            sel_card     = st.selectbox("법인카드 컬럼",  opts, key="sel_card",          index=col_index(opts, _sel_val("sel_card",          auto["card"])))
            sel_user     = st.selectbox("소유자 컬럼",    opts, key="sel_user",          index=col_index(opts, _sel_val("sel_user",          auto["user"])))
            sel_dept     = st.selectbox("관리부서 컬럼",  opts, key="sel_dept",          index=col_index(opts, _sel_val("sel_dept",          auto["dept"])))
        with c2:
            st.markdown("**더존 iUERP 전용 컬럼**")
            sel_approval_type  = st.selectbox("구분 컬럼 (승인/취소)",  opts, key="sel_approval_type",  index=col_index(opts, _sel_val("sel_approval_type",  auto["approval_type"])))
            sel_approval_no    = st.selectbox("승인번호 컬럼",           opts, key="sel_approval_no",    index=col_index(opts, _sel_val("sel_approval_no",    auto["approval_no"])))
            sel_biz_reg        = st.selectbox("사업자등록번호 컬럼",     opts, key="sel_biz_reg",         index=col_index(opts, _sel_val("sel_biz_reg",         auto["biz_reg"])))
            sel_supply_amt     = st.selectbox("공급가액 컬럼",           opts, key="sel_supply_amt",      index=col_index(opts, _sel_val("sel_supply_amt",      auto["supply_amt"])))
            sel_vat            = st.selectbox("부가세 컬럼",             opts, key="sel_vat",             index=col_index(opts, _sel_val("sel_vat",             auto["vat"])))
            sel_vat_type       = st.selectbox("부가세구분 컬럼",         opts, key="sel_vat_type",        index=col_index(opts, _sel_val("sel_vat_type",        auto["vat_type"])))
            sel_vat_yn         = st.selectbox("부가세여부 컬럼",         opts, key="sel_vat_yn",          index=col_index(opts, _sel_val("sel_vat_yn",          auto["vat_yn"])))
            sel_service_fee    = st.selectbox("봉사료 컬럼",             opts, key="sel_service_fee",     index=col_index(opts, _sel_val("sel_service_fee",     auto["service_fee"])))
            sel_account_code   = st.selectbox("계정코드 컬럼",           opts, key="sel_account_code",    index=col_index(opts, _sel_val("sel_account_code",    auto["account_code"])))
            sel_account_name   = st.selectbox("계정명 컬럼",             opts, key="sel_account_name",    index=col_index(opts, _sel_val("sel_account_name",    auto["account_name"])))
            sel_budget_control = st.selectbox("통제성예산 컬럼",         opts, key="sel_budget_control",  index=col_index(opts, _sel_val("sel_budget_control",  auto["budget_control"])))
            sel_cost_center    = st.selectbox("코스트센터명 컬럼",       opts, key="sel_cost_center",     index=col_index(opts, _sel_val("sel_cost_center",     auto["cost_center"])))
            sel_budget_unit    = st.selectbox("예산단위 컬럼",           opts, key="sel_budget_unit",     index=col_index(opts, _sel_val("sel_budget_unit",     auto["budget_unit"])))
            sel_memo           = st.selectbox("적요 컬럼",               opts, key="sel_memo",            index=col_index(opts, _sel_val("sel_memo",            auto["memo"])))
            sel_biz_car        = st.selectbox("업무용차량 컬럼",         opts, key="sel_biz_car",         index=col_index(opts, _sel_val("sel_biz_car",         auto["biz_car"])))
            sel_slip_status    = st.selectbox("전표처리 컬럼",           opts, key="sel_slip_status",     index=col_index(opts, _sel_val("sel_slip_status",     auto["slip_status"])))

    date_col           = to_none(sel_date)
    time_col           = to_none(sel_time)
    amount_col         = to_none(sel_amount)
    merchant_col       = to_none(sel_merchant)
    category_col       = to_none(sel_category)
    card_col           = to_none(sel_card)
    user_col           = to_none(sel_user)
    dept_col           = to_none(sel_dept)
    approval_type_col  = to_none(sel_approval_type)
    approval_no_col    = to_none(sel_approval_no)
    biz_reg_col        = to_none(sel_biz_reg)
    supply_amt_col     = to_none(sel_supply_amt)
    vat_col            = to_none(sel_vat)
    vat_type_col       = to_none(sel_vat_type)
    vat_yn_col         = to_none(sel_vat_yn)
    service_fee_col    = to_none(sel_service_fee)
    account_code_col   = to_none(sel_account_code)
    account_name_col   = to_none(sel_account_name)
    budget_control_col = to_none(sel_budget_control)
    cost_center_col    = to_none(sel_cost_center)
    budget_unit_col    = to_none(sel_budget_unit)
    memo_col           = to_none(sel_memo)
    biz_car_col        = to_none(sel_biz_car)
    slip_status_col    = to_none(sel_slip_status)

    if not date_col:
        st.warning("날짜 컬럼을 선택해야 분석을 진행할 수 있습니다.")
        return

    st.header("3️⃣ 스크리닝 실행")
    run_clicked = st.button("🔍 이상징후 스크리닝 시작", type="primary", use_container_width=True)

    # 파일이 바뀌면 이전 분석 결과 초기화
    _file_key = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("_file_key") != _file_key:
        st.session_state.pop("_cache", None)
        st.session_state["_file_key"] = _file_key

    if run_clicked:
        progress = st.progress(0, text="분석 준비 중...")
        _result = df.copy()

        # 취소 거래 제외 (더존 iUERP '구분' 컬럼)
        _cancelled_df = pd.DataFrame()
        if exclude_cancel and approval_type_col and approval_type_col in _result.columns:
            mask_cancel = _result[approval_type_col].astype(str).str.strip().isin(["취소", "취소(전체)", "CANCEL"])
            _cancelled_df = _result[mask_cancel].copy()
            _result = _result[~mask_cancel].reset_index(drop=True)
        if len(_cancelled_df):
            st.info(f"ℹ️ 취소 거래 **{len(_cancelled_df):,}건** 제외 후 분석합니다. (엑셀 내보내기 시 빨간색으로 표기)")
        # 취소 제외 후의 _result 기준으로 datetime 파싱
        _datetimes = parse_datetimes(_result, date_col, time_col)
        _result["_dt_"] = _datetimes
        _flag_cols: list[str] = []

        if use_weekend:
            progress.progress(10, text="공휴일 데이터 로드 중...")
            kr_hols = load_kr_holidays()
            progress.progress(25, text="주말/공휴일 탐지 중...")
            f, r = detect_weekend_holiday(_datetimes, kr_hols)
            _result["주말_공휴일"] = f
            _result["주말_공휴일_사유"] = r
            _flag_cols.append("주말_공휴일")

        if use_late_night:
            progress.progress(40, text="심야/새벽 탐지 중...")
            if series_has_time(_result, date_col, time_col):
                f, r = detect_late_night(_datetimes, late_start, late_end)
                _result["심야_새벽"] = f
                _result["심야_새벽_사유"] = r
                _flag_cols.append("심야_새벽")
            else:
                st.info("시간 정보가 없어 심야/새벽 탐지를 건너뜁니다.")

        if use_suspicious and (merchant_col or category_col):
            progress.progress(60, text="유흥·사치성 업종 탐지 중...")
            f, r = detect_suspicious(_result, merchant_col, category_col, suspicious_keywords)
            _result["유흥_사치성"] = f
            _result["유흥_사치성_사유"] = r
            _flag_cols.append("유흥_사치성")

        if use_high_amount and amount_col:
            progress.progress(85, text="고액 거래 탐지 중...")
            f, r = detect_high_amount(_result, amount_col, int(high_amount_threshold))
            _result["고액_거래"] = f
            _result["고액_거래_사유"] = r
            _flag_cols.append("고액_거래")

        if use_split and merchant_col:
            progress.progress(88, text="분할결제 탐지 중...")
            f, r = detect_split_payment(_result, merchant_col, date_col, split_min)
            _result["분할_결제"] = f
            _result["분할_결제_사유"] = r
            _flag_cols.append("분할_결제")

        if use_slip and slip_status_col:
            progress.progress(93, text="전표 미처리 탐지 중...")
            f, r = detect_slip_unprocessed(_result, slip_status_col)
            _result["전표_미처리"] = f
            _result["전표_미처리_사유"] = r
            _flag_cols.append("전표_미처리")

        if use_monthly_limit and amount_col and user_col:
            progress.progress(95, text="월 한도 초과 탐지 중...")
            f, r = detect_monthly_limit(_result, amount_col, user_col, date_col, int(monthly_limit))
            _result["월한도_초과"] = f
            _result["월한도_초과_사유"] = r
            _flag_cols.append("월한도_초과")

        progress.progress(97, text="결과 집계 중...")
        _result["위험점수"] = _result[_flag_cols].sum(axis=1).astype(int)
        _result["위험등급"] = _result["위험점수"].map(
            lambda s: "🔴 위험" if s >= 2 else ("🟡 주의" if s == 1 else "🟢 정상")
        )
        _reason_cols = [c for c in _result.columns if c.endswith("_사유")]
        _result["이상사유"] = _result[_reason_cols].apply(
            lambda row: " | ".join(v for v in row if v and str(v) not in ("", "nan")),
            axis=1,
        )
        progress.progress(100, text="완료!")
        progress.empty()

        # 분석 결과를 session_state에 저장 (필터 변경 시에도 유지)
        st.session_state["_cache"] = {
            "result": _result,
            "cancelled": _cancelled_df,
            "flag_cols": _flag_cols,
            "cols": {
                "date": date_col, "time": time_col, "amount": amount_col,
                "merchant": merchant_col, "category": category_col,
                "card": card_col, "user": user_col, "dept": dept_col,
                # 더존 iUERP 전용
                "approval_type":  approval_type_col,
                "approval_no":    approval_no_col,
                "biz_reg":        biz_reg_col,
                "supply_amt":     supply_amt_col,
                "vat":            vat_col,
                "vat_type":       vat_type_col,
                "vat_yn":         vat_yn_col,
                "service_fee":    service_fee_col,
                "account_code":   account_code_col,
                "account_name":   account_name_col,
                "budget_control": budget_control_col,
                "cost_center":    cost_center_col,
                "budget_unit":    budget_unit_col,
                "memo":           memo_col,
                "biz_car":        biz_car_col,
                "slip_status":    slip_status_col,
            },
        }

    if "_cache" not in st.session_state:
        return

    # session_state에서 결과 복원
    cache            = st.session_state["_cache"]
    result           = cache["result"]
    cancelled_df     = cache.get("cancelled", pd.DataFrame())
    flag_cols        = cache["flag_cols"]
    date_col         = cache["cols"]["date"]
    time_col         = cache["cols"]["time"]
    amount_col       = cache["cols"]["amount"]
    merchant_col     = cache["cols"]["merchant"]
    category_col     = cache["cols"]["category"]
    card_col         = cache["cols"]["card"]
    user_col         = cache["cols"]["user"]
    dept_col         = cache["cols"]["dept"]
    approval_type_col  = cache["cols"].get("approval_type")
    approval_no_col    = cache["cols"].get("approval_no")
    biz_reg_col        = cache["cols"].get("biz_reg")
    supply_amt_col     = cache["cols"].get("supply_amt")
    vat_col            = cache["cols"].get("vat")
    vat_type_col       = cache["cols"].get("vat_type")
    vat_yn_col         = cache["cols"].get("vat_yn")
    service_fee_col    = cache["cols"].get("service_fee")
    account_code_col   = cache["cols"].get("account_code")
    account_name_col   = cache["cols"].get("account_name")
    budget_control_col = cache["cols"].get("budget_control")
    cost_center_col    = cache["cols"].get("cost_center")
    budget_unit_col    = cache["cols"].get("budget_unit")
    memo_col           = cache["cols"].get("memo")
    biz_car_col        = cache["cols"].get("biz_car")
    slip_status_col    = cache["cols"].get("slip_status")
    datetimes        = result["_dt_"]

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
                mask = result["위험점수"] > 0
                user_amt = (
                    pd.DataFrame({user_col: result.loc[mask, user_col], "_amt_": amt_s[mask]})
                    .groupby(user_col)["_amt_"].sum()
                    .reset_index()
                    .rename(columns={"_amt_": "이상금액합계"})
                )
                user_stats = user_stats.merge(user_amt, on=user_col, how="left")
                user_stats["이상금액합계"] = user_stats["이상금액합계"].fillna(0).astype(int)
            except Exception:
                pass
        user_stats = user_stats.sort_values("이상건수", ascending=False)
        # 이상금액합계 콤마 포맷 (문자열 변환으로 확실하게 표시)
        disp_user = user_stats.copy()
        if "이상금액합계" in disp_user.columns:
            disp_user["이상금액합계"] = disp_user["이상금액합계"].apply(lambda x: f"{int(x):,}원")
        st.dataframe(disp_user, use_container_width=True, hide_index=True)

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
                mask = result["위험점수"] > 0
                dept_amt = (
                    pd.DataFrame({dept_col: result.loc[mask, dept_col], "_amt_": amt_s[mask]})
                    .groupby(dept_col)["_amt_"].sum()
                    .reset_index()
                    .rename(columns={"_amt_": "이상금액합계"})
                )
                dept_stats = dept_stats.merge(dept_amt, on=dept_col, how="left")
                dept_stats["이상금액합계"] = dept_stats["이상금액합계"].fillna(0).astype(int)
            except Exception:
                pass
        dept_stats = dept_stats.sort_values("이상건수", ascending=False)
        disp_dept = dept_stats.copy()
        if "이상금액합계" in disp_dept.columns:
            disp_dept["이상금액합계"] = disp_dept["이상금액합계"].apply(lambda x: f"{int(x):,}원")
        st.dataframe(disp_dept, use_container_width=True, hide_index=True)

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
    # 기본 컬럼
    for c in [date_col, time_col, user_col, dept_col, card_col,
              merchant_col, category_col, amount_col]:
        if c:
            show_cols.append(c)
    # 더존 iUERP 전용 컬럼 (존재할 때만 표시)
    for c in [approval_type_col, approval_no_col, biz_reg_col,
              supply_amt_col, vat_col, vat_type_col, vat_yn_col,
              service_fee_col, account_code_col, account_name_col,
              budget_control_col, cost_center_col, budget_unit_col,
              memo_col, biz_car_col, slip_status_col]:
        if c:
            show_cols.append(c)
    show_cols.append("위험점수")
    show_cols = [c for c in show_cols if c in display.columns]

    # ── 표시 행 수 제한 (브라우저 렌더링 부담 감소) ─────────────────────────
    MAX_ROWS = 500
    total_display = len(display)
    if total_display > MAX_ROWS:
        st.warning(
            f"⚡ 표시 건수가 많아 상위 **{MAX_ROWS:,}건**만 미리봅니다. "
            f"(전체 {total_display:,}건은 아래 엑셀로 다운로드하세요)"
        )
        display = display.head(MAX_ROWS)

    # ── 위험등급 컬럼만 배경색 적용 (행 전체 스타일링 대비 ~20× 빠름) ────────
    def _grade_bg(s: pd.Series) -> list[str]:
        return [
            "background-color:#fde8e8" if "위험" in str(v) else
            "background-color:#fef9e7" if "주의" in str(v) else ""
            for v in s
        ]

    # 금액 컬럼 콤마 포맷
    _money_fmt = lambda x: (
        f"{float(str(x).replace(',', '')):,.0f}"
        if str(x) not in ("nan", "", "0") else "-"
    )
    fmt = {}
    for _mc in [amount_col, supply_amt_col, vat_col, service_fee_col]:
        if _mc and _mc in show_cols:
            fmt[_mc] = _money_fmt

    st.caption(f"표시 건수: {min(total_display, MAX_ROWS):,}건 / 전체 {total_display:,}건")
    styled = display[show_cols].style
    if "위험등급" in show_cols:
        styled = styled.apply(_grade_bg, subset=["위험등급"])
    if fmt:
        styled = styled.format(fmt, na_rep="-")
    st.dataframe(styled, use_container_width=True, height=420, hide_index=True)

    st.subheader("📥 결과 다운로드")

    # ── 내보낼 컬럼 정리: bool 플래그·개별 사유 컬럼 제거 ──────────────────
    _drop_cols = (
        list(flag_cols) +
        [c for c in result.columns if c.endswith("_사유")] +
        ["_dt_"]
    )
    export = result.drop(columns=_drop_cols, errors="ignore")

    # ── 취소거래 준비 ─────────────────────────────────────────────────────────
    has_cancelled = len(cancelled_df) > 0
    if has_cancelled:
        _cancel_export = cancelled_df.drop(
            columns=[c for c in _drop_cols if c in cancelled_df.columns],
            errors="ignore",
        )
        for col in ["위험등급", "이상사유", "위험점수"]:
            if col not in _cancel_export.columns:
                _cancel_export[col] = "취소"
        n_cancel_rows = len(_cancel_export)
    else:
        _cancel_export = pd.DataFrame()
        n_cancel_rows = 0

    # ── 더존 iUERP 포맷 컬럼 순서 정의 ──────────────────────────────────────
    _desired_order = [
        dept_col, user_col, date_col, time_col, merchant_col,
        vat_type_col, vat_yn_col, category_col,
        supply_amt_col, vat_col, amount_col,
        approval_type_col,
        approval_no_col, biz_reg_col,
        account_code_col, account_name_col, budget_control_col,
        cost_center_col, budget_unit_col, memo_col, biz_car_col,
        service_fee_col, card_col,
        "이상사유", "위험등급",
    ]
    _seen: set = set()
    _export_cols: list = []
    for _c in _desired_order:
        if _c and _c in export.columns and _c not in _seen:
            _export_cols.append(_c)
            _seen.add(_c)
    # 위 순서에 없는 원본 컬럼도 추가 (내부 플래그 컬럼 제외)
    _skip = set(flag_cols) | {c for c in export.columns if c.endswith("_사유")} | {"위험점수", "_dt_"}
    for _c in export.columns:
        if _c not in _seen and _c not in _skip:
            _export_cols.append(_c)
            _seen.add(_c)

    # ── 그룹형 엑셀 빌드 (openpyxl 직접 사용) ─────────────────────────────
    from openpyxl import Workbook
    from openpyxl.styles import Alignment

    _grey_fill   = PatternFill(fill_type="solid", fgColor="D9D9D9")
    _sub_fill    = PatternFill(fill_type="solid", fgColor="F2F2F2")
    _red_fill    = PatternFill(fill_type="solid", fgColor="FFD7D7")
    _yellow_fill = PatternFill(fill_type="solid", fgColor="FFFFD7")
    _hdr_fill    = PatternFill(fill_type="solid", fgColor="4472C4")
    _red_font    = Font(color="FF0000")
    _bold_font   = Font(bold=True)
    _hdr_font    = Font(bold=True, color="FFFFFF")
    _num_cols    = {c for c in [supply_amt_col, vat_col, amount_col, service_fee_col] if c}

    def _row_values(row_series):
        vals = []
        for c in _export_cols:
            v = row_series.get(c, "")
            if pd.isna(v):
                v = ""
            vals.append(v)
        return vals

    def _write_header(ws):
        ws.append(_export_cols)
        r = ws.max_row
        for i, _ in enumerate(_export_cols, 1):
            cell = ws.cell(row=r, column=i)
            cell.font = _hdr_font
            cell.fill = _hdr_fill
            cell.alignment = Alignment(horizontal="center")

    def _write_nm_header(ws, owner_name):
        ws.append([f"NM_OWNER: {owner_name}"] + [""] * (len(_export_cols) - 1))
        r = ws.max_row
        if len(_export_cols) > 1:
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=len(_export_cols))
        cell = ws.cell(row=r, column=1)
        cell.fill = _grey_fill
        cell.font = _bold_font

    def _write_data_row(ws, row_series, is_cancel=False):
        ws.append(_row_values(row_series))
        r = ws.max_row
        risk = row_series.get("위험점수", 0)
        if is_cancel:
            for cell in ws[r]:
                cell.font = _red_font
        elif risk >= 2:
            for cell in ws[r]:
                cell.fill = _red_fill
        elif risk == 1:
            for cell in ws[r]:
                cell.fill = _yellow_fill

    def _write_subtotal(ws, grp_df):
        sub = [""] * len(_export_cols)
        for c in _num_cols:
            if c in _export_cols:
                idx = _export_cols.index(c)
                try:
                    sub[idx] = pd.to_numeric(
                        grp_df[c].astype(str).str.replace(",", ""), errors="coerce"
                    ).fillna(0).sum()
                except Exception:
                    pass
        ws.append(sub)
        r = ws.max_row
        for cell in ws[r]:
            cell.fill = _sub_fill
            cell.font = _bold_font

    def _write_grouped_sheet(ws, approved_df, cancel_df):
        _write_header(ws)
        if user_col and user_col in approved_df.columns:
            owners = approved_df[user_col].dropna().unique()
            for owner in owners:
                grp = approved_df[approved_df[user_col] == owner]
                c_grp = cancel_df[cancel_df[user_col] == owner] if (
                    len(cancel_df) > 0 and user_col in cancel_df.columns
                ) else pd.DataFrame()
                _write_nm_header(ws, owner)
                for _, row in grp.iterrows():
                    _write_data_row(ws, row)
                for _, row in c_grp.iterrows():
                    _write_data_row(ws, row, is_cancel=True)
                _write_subtotal(ws, grp)
        else:
            for _, row in approved_df.iterrows():
                _write_data_row(ws, row)
            if len(cancel_df) > 0:
                for _, row in cancel_df.iterrows():
                    _write_data_row(ws, row, is_cancel=True)

    wb = Workbook()
    ws_all = wb.active
    ws_all.title = "전체결과"
    _write_grouped_sheet(ws_all, export, _cancel_export)

    ws_flag = wb.create_sheet("이상의심")
    _flagged_df = export[export["위험점수"] > 0] if "위험점수" in export.columns else export
    _write_grouped_sheet(ws_flag, _flagged_df, pd.DataFrame())

    ws_high = wb.create_sheet("고위험")
    _high_df = export[export["위험점수"] >= 2] if "위험점수" in export.columns else pd.DataFrame(columns=_export_cols)
    _write_grouped_sheet(ws_high, _high_df, pd.DataFrame())

    ws_sum = wb.create_sheet("요약")
    _summary_data = pd.DataFrame({
        "구분": ["총 거래건수", "취소 거래(제외)", "이상 의심 건수", "고위험 건수", "이상 비율(%)"],
        "값":   [total + n_cancel_rows, n_cancel_rows, flagged, high_risk,
                 f"{flagged/total*100:.1f}%" if total else "0%"],
    })
    ws_sum.append(["구분", "값"])
    for _, row in _summary_data.iterrows():
        ws_sum.append([row["구분"], row["값"]])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    st.download_button(
        label="📥 분석 결과 엑셀 다운로드",
        data=buf.getvalue(),
        file_name=f"{Path(uploaded.name).stem}_스크리닝.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

if __name__ == "__main__":
    main()
