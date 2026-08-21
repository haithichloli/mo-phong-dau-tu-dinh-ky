from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter


st.set_page_config(
    page_title="Mô phỏng đầu tư định kỳ",
    page_icon="📈",
    layout="wide",
)


KENH_THI_TRUONG = {
    "Chứng khoán Việt Nam (ETF VN30)": "E1VFVN30.VN",
    "Chứng khoán Mỹ (S&P 500)": "^GSPC",
    "Vàng": "GC=F",
    "Bitcoin": "BTC-USD",
}

MAU_KENH = {
    "Tiết kiệm": "#0F766E",
    "Chứng khoán Việt Nam (ETF VN30)": "#2563EB",
    "Chứng khoán Mỹ (S&P 500)": "#7C3AED",
    "Vàng": "#D97706",
    "Bitcoin": "#EA580C",
}

TEN_COT_DU_PHONG = {
    "Chứng khoán Việt Nam (ETF VN30)": "ETF_VN30",
    "Chứng khoán Mỹ (S&P 500)": "SP500",
    "Vàng": "VANG",
    "Bitcoin": "BITCOIN",
}

KHOA_MATPLOTLIB = RLock()


@dataclass
class KetQuaMoPhong:
    ten_kenh: str
    duong_di: np.ndarray
    tong_von: np.ndarray
    gia_tri_cuoi: np.ndarray
    trung_vi: float
    phan_vi_5: float
    phan_vi_95: float
    xac_suat_lo: float
    nguon_du_lieu: str


def dinh_dang_vnd(so_tien: float) -> str:
    """Định dạng số tiền lớn theo triệu hoặc tỷ đồng."""
    if abs(so_tien) >= 1_000_000_000:
        return f"{so_tien / 1_000_000_000:,.2f} tỷ đồng"
    return f"{so_tien / 1_000_000:,.1f} triệu đồng"


@st.cache_data(ttl="6h", show_spinner=False)
def tai_loi_suat_thang(
    ten_kenh: str, ticker: str, giai_doan: str
) -> tuple[np.ndarray, str]:
    """Tải lợi suất tháng; tự dùng dữ liệu đóng gói nếu Yahoo tạm lỗi."""
    try:
        du_lieu = yf.download(
            ticker,
            period=giai_doan,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if du_lieu.empty or "Close" not in du_lieu.columns:
            raise ValueError("Không tải được dữ liệu giá.")

        gia_dong_cua = du_lieu["Close"]
        if isinstance(gia_dong_cua, pd.DataFrame):
            gia_dong_cua = gia_dong_cua.iloc[:, 0]

        gia_dong_cua = pd.to_numeric(gia_dong_cua, errors="coerce").dropna()
        gia_dong_cua.index = pd.to_datetime(gia_dong_cua.index)
        gia_thang = gia_dong_cua.resample("ME").last().dropna()
        loi_suat = gia_thang.pct_change(fill_method=None).dropna()
        loi_suat = loi_suat[np.isfinite(loi_suat) & (loi_suat > -0.999)]

        if len(loi_suat) < 24:
            raise ValueError("Dữ liệu lịch sử có ít hơn 24 tháng.")

        ngay_cuoi = gia_thang.index[-1].strftime("%d/%m/%Y")
        return (
            loi_suat.to_numpy(dtype=float),
            f"Yahoo Finance · dữ liệu đến {ngay_cuoi}",
        )
    except Exception:
        tep_du_phong = Path(__file__).parent / "data" / "historical_monthly_returns.csv"
        if ten_kenh not in TEN_COT_DU_PHONG or not tep_du_phong.exists():
            raise

        bang = pd.read_csv(tep_du_phong, parse_dates=["Ngay"], index_col="Ngay")
        loi_suat = pd.to_numeric(
            bang[TEN_COT_DU_PHONG[ten_kenh]], errors="coerce"
        ).dropna()
        if giai_doan in {"5y", "10y"}:
            so_nam = int(giai_doan.removesuffix("y"))
            moc = loi_suat.index.max() - pd.DateOffset(years=so_nam)
            loi_suat = loi_suat[loi_suat.index >= moc]
        loi_suat = loi_suat[np.isfinite(loi_suat) & (loi_suat > -0.999)]
        if len(loi_suat) < 24:
            raise ValueError("Dữ liệu dự phòng có ít hơn 24 tháng.")

        ngay_cuoi = loi_suat.index[-1].strftime("%d/%m/%Y")
        return (
            loi_suat.to_numpy(dtype=float),
            f"Bản dữ liệu Yahoo Finance dự phòng · đến {ngay_cuoi}",
        )


def mo_phong_tu_loi_suat(
    ten_kenh: str,
    loi_suat_lich_su: np.ndarray,
    von_ban_dau: float,
    dau_tu_hang_thang: float,
    so_thang: int,
    so_lan_mo_phong: int,
    hat_giong: int,
    nguon_du_lieu: str,
) -> KetQuaMoPhong:
    """Bootstrap lợi suất lịch sử để tạo nhiều kịch bản tương lai."""
    bo_tao_so = np.random.default_rng(hat_giong)
    chi_so_ngau_nhien = bo_tao_so.integers(
        0,
        len(loi_suat_lich_su),
        size=(so_lan_mo_phong, so_thang),
    )
    loi_suat_mo_phong = loi_suat_lich_su[chi_so_ngau_nhien]

    duong_di = np.zeros((so_lan_mo_phong, so_thang + 1), dtype=float)
    duong_di[:, 0] = von_ban_dau

    # Giả định khoản đầu tư định kỳ được nộp vào đầu mỗi tháng.
    for thang in range(so_thang):
        duong_di[:, thang + 1] = (
            duong_di[:, thang] + dau_tu_hang_thang
        ) * (1 + loi_suat_mo_phong[:, thang])

    tong_von = von_ban_dau + dau_tu_hang_thang * np.arange(so_thang + 1)
    gia_tri_cuoi = duong_di[:, -1]
    tong_von_cuoi = tong_von[-1]

    return KetQuaMoPhong(
        ten_kenh=ten_kenh,
        duong_di=duong_di,
        tong_von=tong_von,
        gia_tri_cuoi=gia_tri_cuoi,
        trung_vi=float(np.median(gia_tri_cuoi)),
        phan_vi_5=float(np.percentile(gia_tri_cuoi, 5)),
        phan_vi_95=float(np.percentile(gia_tri_cuoi, 95)),
        xac_suat_lo=float(np.mean(gia_tri_cuoi < tong_von_cuoi) * 100),
        nguon_du_lieu=nguon_du_lieu,
    )


def mo_phong_tiet_kiem(
    von_ban_dau: float,
    dau_tu_hang_thang: float,
    so_thang: int,
    so_lan_mo_phong: int,
    lai_suat_nam: float,
) -> KetQuaMoPhong:
    """Mô phỏng tiết kiệm với lãi kép và lãi suất năm cố định."""
    lai_suat_thang = (1 + lai_suat_nam / 100) ** (1 / 12) - 1
    loi_suat = np.full(12, lai_suat_thang, dtype=float)
    return mo_phong_tu_loi_suat(
        ten_kenh="Tiết kiệm",
        loi_suat_lich_su=loi_suat,
        von_ban_dau=von_ban_dau,
        dau_tu_hang_thang=dau_tu_hang_thang,
        so_thang=so_thang,
        so_lan_mo_phong=so_lan_mo_phong,
        hat_giong=0,
        nguon_du_lieu=f"Lãi suất giả định {lai_suat_nam:.2f}%/năm",
    )


def dinh_dang_truc_tien(so_tien: float, _vi_tri: int) -> str:
    if abs(so_tien) >= 1_000_000_000:
        return f"{so_tien / 1_000_000_000:.1f} tỷ"
    return f"{so_tien / 1_000_000:.0f} tr"


def bieu_do_dai_kich_ban(ket_qua: KetQuaMoPhong):
    thang = np.arange(ket_qua.duong_di.shape[1])
    nam = thang / 12
    p5 = np.percentile(ket_qua.duong_di, 5, axis=0)
    p50 = np.percentile(ket_qua.duong_di, 50, axis=0)
    p95 = np.percentile(ket_qua.duong_di, 95, axis=0)
    mau = MAU_KENH.get(ket_qua.ten_kenh, "#2563EB")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")
    ax.fill_between(nam, p5, p95, color="#93C5FD", alpha=0.35,
                    label="Khoảng kịch bản 5%–95%")
    ax.plot(nam, p50, color=mau, linewidth=2.6, label="Giá trị trung vị")
    ax.plot(nam, ket_qua.tong_von, color="#475569", linewidth=2,
            linestyle="--", label="Tổng vốn đã góp")
    ax.set_title(f"Dải kịch bản — {ket_qua.ten_kenh}", fontweight="bold")
    ax.set_xlabel("Số năm")
    ax.set_ylabel("Giá trị danh mục")
    ax.yaxis.set_major_formatter(FuncFormatter(dinh_dang_truc_tien))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    fig.tight_layout()
    return fig


def bieu_do_so_sanh(cac_ket_qua: dict[str, KetQuaMoPhong]):
    ten = list(cac_ket_qua)
    tong_von = [cac_ket_qua[k].tong_von[-1] for k in ten]
    trung_vi = [cac_ket_qua[k].trung_vi for k in ten]
    x = np.arange(len(ten))
    rong = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.patch.set_facecolor("white")
    cot_von = ax.bar(x - rong / 2, tong_von, rong, color="#94A3B8",
                     label="Tổng vốn góp")
    cot_trung_vi = ax.bar(
        x + rong / 2,
        trung_vi,
        rong,
        color=[MAU_KENH.get(k, "#2563EB") for k in ten],
        label="Giá trị cuối kỳ trung vị",
    )
    ax.bar_label(cot_von, labels=[dinh_dang_vnd(v) for v in tong_von],
                 fontsize=8, padding=3, rotation=15)
    ax.bar_label(cot_trung_vi, labels=[dinh_dang_vnd(v) for v in trung_vi],
                 fontsize=8, padding=3, rotation=15)
    ax.set_title("So sánh vốn góp và giá trị cuối kỳ", fontweight="bold")
    ax.set_xticks(x, ten, rotation=12, ha="right")
    ax.set_ylabel("Giá trị")
    ax.yaxis.set_major_formatter(FuncFormatter(dinh_dang_truc_tien))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.margins(y=0.18)
    fig.tight_layout()
    return fig


def bieu_do_phan_phoi(ket_qua: KetQuaMoPhong):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.patch.set_facecolor("white")
    ax.hist(
        ket_qua.gia_tri_cuoi,
        bins=50,
        color=MAU_KENH.get(ket_qua.ten_kenh, "#2563EB"),
        alpha=0.82,
    )
    ax.axvline(ket_qua.tong_von[-1], color="#DC2626", linestyle="--",
               linewidth=2, label="Tổng vốn góp")
    ax.axvline(ket_qua.trung_vi, color="#111827", linewidth=2,
               label="Trung vị")
    ax.set_title(f"Phân phối giá trị cuối kỳ — {ket_qua.ten_kenh}",
                 fontweight="bold")
    ax.set_xlabel("Giá trị cuối kỳ")
    ax.set_ylabel("Số kịch bản")
    ax.xaxis.set_major_formatter(FuncFormatter(dinh_dang_truc_tien))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


st.markdown(
    """
    <style>
        :root { color-scheme: light; }
        .stApp {
            background: linear-gradient(135deg, #F0FDFA 0%, #EFF6FF 100%);
            color: #0F172A;
        }
        .block-container { max-width: 1200px; padding-top: 2rem; }
        .hero {
            padding: 28px;
            border-radius: 20px;
            color: white;
            background: linear-gradient(135deg, #0F766E, #2563EB);
            box-shadow: 0 12px 30px rgba(15, 118, 110, 0.20);
            margin-bottom: 22px;
        }
        .hero h1, .hero p { color: white !important; margin: 0; }
        .hero p { margin-top: 10px; font-size: 17px; }
        .card {
            background: white;
            border: 1px solid #D8E2EA;
            border-radius: 16px;
            padding: 17px;
            min-height: 132px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
        }
        .card .label { color: #475569; font-weight: 650; }
        .card .value { color: #0F172A; font-size: 25px; font-weight: 850; margin-top: 9px; }
        .card .sub { color: #64748B; margin-top: 6px; }
        .disclaimer {
            background: #FFF7ED;
            color: #7C2D12;
            border-left: 5px solid #F97316;
            border-radius: 10px;
            padding: 16px 18px;
            margin-top: 20px;
            font-weight: 600;
        }
        div[data-testid="stForm"] {
            background: white;
            border: 1px solid #D8E2EA;
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
        }
        div[data-testid="stFormSubmitButton"] button {
            min-height: 48px;
            border: 0;
            border-radius: 12px;
            color: white;
            font-weight: 750;
            background: linear-gradient(135deg, #0F766E, #2563EB);
        }
    </style>
    <div class="hero">
        <h1>📈 Mô phỏng đầu tư định kỳ</h1>
        <p>
            So sánh nhiều kênh đầu tư khi góp một khoản tiền cố định mỗi tháng,
            dựa trên dữ liệu lịch sử và các kịch bản biến động ngẫu nhiên.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Phương pháp: lấy mẫu ngẫu nhiên có hoàn lại từ lợi suất tháng trong lịch sử. "
    "Đây là mô phỏng kịch bản, không phải dự báo giá."
)

with st.form("thong_so_mo_phong"):
    cot_1, cot_2, cot_3 = st.columns(3)
    with cot_1:
        dau_tu_hang_thang = st.number_input(
            "Số tiền đầu tư mỗi tháng (VND)",
            min_value=0,
            value=10_000_000,
            step=1_000_000,
        )
        von_ban_dau = st.number_input(
            "Vốn ban đầu (VND)",
            min_value=0,
            value=0,
            step=10_000_000,
        )
    with cot_2:
        so_nam = st.slider("Thời gian đầu tư (năm)", 1, 30, 10)
        so_lan_mo_phong = st.slider(
            "Số kịch bản mô phỏng", 200, 5_000, 1_000, step=200
        )
    with cot_3:
        giai_doan_nhan = st.selectbox(
            "Độ dài dữ liệu lịch sử",
            ["5 năm", "10 năm", "Tối đa"],
            index=1,
        )
        hat_giong = st.number_input(
            "Hạt giống ngẫu nhiên", min_value=0, value=42, step=1
        )

    kenh_da_chon = st.multiselect(
        "Chọn các kênh đầu tư",
        ["Tiết kiệm", *KENH_THI_TRUONG.keys()],
        default=[
            "Tiết kiệm",
            "Chứng khoán Việt Nam (ETF VN30)",
            "Vàng",
            "Bitcoin",
        ],
    )
    lai_suat_tiet_kiem = st.number_input(
        "Lãi suất tiết kiệm giả định (%/năm)",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
        step=0.1,
    )
    bat_dau = st.form_submit_button("Chạy mô phỏng", width="stretch")

if bat_dau:
    if not kenh_da_chon:
        st.error("Hãy chọn ít nhất một kênh đầu tư.")
    elif dau_tu_hang_thang == 0 and von_ban_dau == 0:
        st.error("Vốn ban đầu và số tiền đầu tư hàng tháng không thể cùng bằng 0.")
    else:
        giai_doan = {"5 năm": "5y", "10 năm": "10y", "Tối đa": "max"}[
            giai_doan_nhan
        ]
        so_thang = int(so_nam * 12)
        cac_ket_qua: dict[str, KetQuaMoPhong] = {}
        loi_du_lieu: list[str] = []

        with st.spinner("Đang tải dữ liệu và tạo các kịch bản..."):
            for vi_tri, ten_kenh in enumerate(kenh_da_chon):
                if ten_kenh == "Tiết kiệm":
                    cac_ket_qua[ten_kenh] = mo_phong_tiet_kiem(
                        von_ban_dau=float(von_ban_dau),
                        dau_tu_hang_thang=float(dau_tu_hang_thang),
                        so_thang=so_thang,
                        so_lan_mo_phong=int(so_lan_mo_phong),
                        lai_suat_nam=float(lai_suat_tiet_kiem),
                    )
                    continue

                try:
                    loi_suat, nguon = tai_loi_suat_thang(
                        ten_kenh, KENH_THI_TRUONG[ten_kenh], giai_doan
                    )
                    cac_ket_qua[ten_kenh] = mo_phong_tu_loi_suat(
                        ten_kenh=ten_kenh,
                        loi_suat_lich_su=loi_suat,
                        von_ban_dau=float(von_ban_dau),
                        dau_tu_hang_thang=float(dau_tu_hang_thang),
                        so_thang=so_thang,
                        so_lan_mo_phong=int(so_lan_mo_phong),
                        hat_giong=int(hat_giong) + vi_tri,
                        nguon_du_lieu=nguon,
                    )
                except Exception as exc:
                    loi_du_lieu.append(f"{ten_kenh}: {exc}")

        st.session_state["cac_ket_qua"] = cac_ket_qua
        st.session_state["loi_du_lieu"] = loi_du_lieu

if "cac_ket_qua" in st.session_state:
    cac_ket_qua = st.session_state["cac_ket_qua"]
    loi_du_lieu = st.session_state.get("loi_du_lieu", [])

    if loi_du_lieu:
        st.warning(
            "Một số kênh không tải được dữ liệu:\n\n- " + "\n- ".join(loi_du_lieu)
        )

    if cac_ket_qua:
        st.subheader("Kết quả tổng quan")
        bang_tom_tat = []
        for ten_kenh, ket_qua in cac_ket_qua.items():
            tong_von_cuoi = ket_qua.tong_von[-1]
            loi_nhuan = ket_qua.trung_vi - tong_von_cuoi
            ty_suat = loi_nhuan / tong_von_cuoi * 100 if tong_von_cuoi else 0
            bang_tom_tat.append(
                {
                    "Kênh đầu tư": ten_kenh,
                    "Tổng vốn góp": round(tong_von_cuoi),
                    "Kịch bản thấp 5%": round(ket_qua.phan_vi_5),
                    "Giá trị trung vị": round(ket_qua.trung_vi),
                    "Kịch bản cao 95%": round(ket_qua.phan_vi_95),
                    "Lợi nhuận trung vị": round(loi_nhuan),
                    "Tỷ suất trung vị (%)": round(ty_suat, 2),
                    "Xác suất thấp hơn vốn góp (%)": round(ket_qua.xac_suat_lo, 2),
                }
            )

        st.dataframe(
            pd.DataFrame(bang_tom_tat),
            hide_index=True,
            width="stretch",
            column_config={
                "Tổng vốn góp": st.column_config.NumberColumn(format="%d ₫"),
                "Kịch bản thấp 5%": st.column_config.NumberColumn(format="%d ₫"),
                "Giá trị trung vị": st.column_config.NumberColumn(format="%d ₫"),
                "Kịch bản cao 95%": st.column_config.NumberColumn(format="%d ₫"),
                "Lợi nhuận trung vị": st.column_config.NumberColumn(format="%d ₫"),
            },
        )

        nguon_du_phong = [
            ten
            for ten, ket_qua in cac_ket_qua.items()
            if "dự phòng" in ket_qua.nguon_du_lieu
        ]
        if nguon_du_phong:
            st.warning(
                "Yahoo Finance đang tạm thời không phản hồi cho: "
                + ", ".join(nguon_du_phong)
                + ". Ứng dụng đã tự dùng bản dữ liệu lịch sử đóng gói."
            )

        with KHOA_MATPLOTLIB:
            hinh_so_sanh = bieu_do_so_sanh(cac_ket_qua)
            st.pyplot(hinh_so_sanh, width="stretch")
            plt.close(hinh_so_sanh)

        kenh_chi_tiet = st.selectbox(
            "Chọn kênh để xem chi tiết", list(cac_ket_qua.keys())
        )
        ket_qua = cac_ket_qua[kenh_chi_tiet]

        cot_a, cot_b, cot_c, cot_d = st.columns(4)
        tong_von_cuoi = ket_qua.tong_von[-1]
        cot_a.markdown(
            f'<div class="card"><div class="label">Tổng vốn góp</div>'
            f'<div class="value">{dinh_dang_vnd(tong_von_cuoi)}</div></div>',
            unsafe_allow_html=True,
        )
        cot_b.markdown(
            f'<div class="card"><div class="label">Giá trị trung vị</div>'
            f'<div class="value">{dinh_dang_vnd(ket_qua.trung_vi)}</div></div>',
            unsafe_allow_html=True,
        )
        cot_c.markdown(
            f'<div class="card"><div class="label">Khoảng kịch bản 5%–95%</div>'
            f'<div class="value">{dinh_dang_vnd(ket_qua.phan_vi_5)}</div>'
            f'<div class="sub">đến {dinh_dang_vnd(ket_qua.phan_vi_95)}</div></div>',
            unsafe_allow_html=True,
        )
        cot_d.markdown(
            f'<div class="card"><div class="label">Xác suất thấp hơn vốn góp</div>'
            f'<div class="value">{ket_qua.xac_suat_lo:.1f}%</div></div>',
            unsafe_allow_html=True,
        )

        with KHOA_MATPLOTLIB:
            hinh_dai = bieu_do_dai_kich_ban(ket_qua)
            st.pyplot(hinh_dai, width="stretch")
            plt.close(hinh_dai)

            hinh_phan_phoi = bieu_do_phan_phoi(ket_qua)
            st.pyplot(hinh_phan_phoi, width="stretch")
            plt.close(hinh_phan_phoi)

        st.caption(f"Nguồn/giả định cho {kenh_chi_tiet}: {ket_qua.nguon_du_lieu}")

st.markdown(
    """
    <div class="disclaimer">
        <strong>Miễn trừ trách nhiệm:</strong> Công cụ này chỉ mô phỏng các kịch bản
        dựa trên dữ liệu quá khứ. Kết quả quá khứ không đảm bảo kết quả tương lai
        và nội dung không phải là khuyến nghị đầu tư. Mô hình chưa tính thuế, phí,
        trượt giá, lạm phát hoặc tác động tỷ giá đối với tài sản nước ngoài.
        ETF VN30 được dùng làm đại diện cho kênh chứng khoán Việt Nam; kết quả
        thực tế còn phụ thuộc thanh khoản và sai lệch bám chỉ số.
    </div>
    """,
    unsafe_allow_html=True,
)

