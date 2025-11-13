import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
from shapely.geometry import Point

# ----------------- 사용자 경로 -----------------
BASE = "C:/Users/USER/Desktop/공모전/충남대/python"
DATA = os.path.join(BASE, "data")

# 이미 파이썬에서 만들었던 병합 결과(청주 500m 격자 + 인구)
# 없으면 아래에서 grid_pop을 다시 만들어도 되는데 여기선 파일로 가져오는 가정
# (바로 메모리의 grid_pop을 쓰려면, 이 부분을 주석 처리하고 아래 st.session_state 사용)
GRID_POP_GPKG = os.path.join(BASE, "data", "chungju_grid_pop.gpkg")

# 체육시설 CSV (지오코딩 결과 포함)
FAC_CSV = os.path.join(DATA, "cheongju_geocoded_google.csv")
# ---------------------------------------------

st.set_page_config(page_title="청주시 인구·체육시설 지도", layout="wide")

st.title("청주시 500m 격자 인구 & 체육시설 대시보드")



# 0) OSM 베이스맵 (토큰 불필요)
tile_layer = pdk.Layer(
    "TileLayer",
    data="https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
    minZoom=0, maxZoom=19, tileSize=256
)




# ========= 데이터 로드 =========
# 1) 격자 인구 (GeoPackage 권장)
@st.cache_data(show_spinner=False)
def load_grid_pop(gpkg_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg_path)
    # 예상 컬럼: GRID_CD, POP, geometry
    # 좌표: 시각화용 deck.gl은 WGS84 필요
    if gdf.crs is None:
        # 데이터에 crs 없으면 WGS84라고 가정(필요시 수정)
        gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(epsg=4326)
    # 숫자형 보장
    gdf["POP"] = pd.to_numeric(gdf.get("POP"), errors="coerce")
    return gdf

# 2) 체육시설 포인트 (final_location → lat, lon 파싱)
@st.cache_data(show_spinner=False)
def load_facilities(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")

    # final_location: "36.xxx, 126.xxx" (lat, lon)
    latlon = df["final_location"].astype(str).str.extract(
        r'^\s*(?P<lat>-?\d+(?:\.\d+)?)\s*,\s*(?P<lon>-?\d+(?:\.\d+)?)\s*$'
    )
    df["lat"] = pd.to_numeric(latlon["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(latlon["lon"], errors="coerce")

    # 한국 범위 필터
    df = df.dropna(subset=["lat", "lon"]).copy()
    df = df[(df["lat"].between(33, 39)) & (df["lon"].between(124, 132))].copy()

    # 카테고리 표준화(공백 제거, 결측 NA 처리)
    cat = df["시설구분"].astype(str).str.strip()
    cat = cat.where(cat.isin(["생활", "전문", "직장", "NA"]), other="NA")
    df["시설구분"] = cat

    # 시설명 컬럼 이름이 다를 수 있어 대비
    # 우선순위로 찾기
    name_col_candidates = ["시설명", "name", "시설 이름", "시설명칭"]
    name_col = next((c for c in name_col_candidates if c in df.columns), None)
    if name_col is None:
        # 없으면 임시명
        df["시설명"] = "(이름없음)"
        name_col = "시설명"

    return df, name_col

grid_pop = load_grid_pop(GRID_POP_GPKG)
fac_df, FAC_NAME_COL = load_facilities(FAC_CSV)

# ========= 사이드바 =========
st.sidebar.header("레이어 컨트롤")
# 격자 시각화 스케일 선택
scale_type = st.sidebar.selectbox("인구 색상 스케일", ["Log(권장)", "Linear"], index=0)
alpha = st.sidebar.slider("격자 투명도", 0.0, 1.0, 0.6, 0.05)

# 체육시설 카테고리 토글(다중 선택)
all_cats = ["생활", "전문", "직장", "NA"]
selected_cats = st.sidebar.multiselect("시설구분 표시", all_cats, default=all_cats)

# ========= 색상 매핑(인구: 회색→빨강) =========
# 회색(#E5E7EB) → 연빨강(#FCA5A5) → 진빨강(#DC2626)
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

COLORS = [hex_to_rgb("#E5E7EB"), hex_to_rgb("#FCA5A5"), hex_to_rgb("#DC2626")]

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def color_scale_gray_red(x, vmin, vmax):
    if pd.isna(x) or vmax <= vmin:
        return [0, 0, 0, 0]
    # 0~1 정규화
    t = (x - vmin) / (vmax - vmin)
    t = np.clip(t, 0, 1)
    # 두 구간 보간 (3색)
    if t < 0.5:
        c = lerp_color(COLORS[0], COLORS[1], t / 0.5)
    else:
        c = lerp_color(COLORS[1], COLORS[2], (t - 0.5) / 0.5)
    return [c[0], c[1], c[2], int(alpha * 255)]


vals = grid_pop["POP"].dropna()
if len(vals) == 0:
    vmin, vmax = 0, 1
    scale_vals = grid_pop["POP"]
elif scale_type.startswith("Log"):
    # 로그 스케일: 0 회피 + 극단치 완화
    scale_vals = np.log1p(vals)
    vmin = max(scale_vals.min(), 1e-6)
    vmax = np.quantile(scale_vals, 0.99)
    grid_pop["_POP_SCALED"] = np.log1p(grid_pop["POP"].clip(lower=0))
else:
    # 선형: 극단치 완화
    vmin, vmax = 0, np.quantile(vals, 0.99)
    grid_pop["_POP_SCALED"] = grid_pop["POP"].clip(lower=0, upper=vmax)

grid_pop["_fill_color"] = grid_pop["_POP_SCALED"].apply(lambda x: color_scale_gray_red(x, vmin, vmax))

# ========= 중심 뷰 계산 =========
if grid_pop.geometry.is_empty.all():
    view_state = pdk.ViewState(latitude=36.64, longitude=127.49, zoom=10.5, pitch=0)
else:
    center = grid_pop.to_crs(epsg=4326).geometry.unary_union.centroid
    view_state = pdk.ViewState(latitude=center.y, longitude=center.x, zoom=11, pitch=0)

# ========= 레이어 구성 =========
# 1) 격자(GeoJsonLayer) — hover 시 POP 표시
grid_geojson = grid_pop.__geo_interface__  # FeatureCollection

grid_layer = pdk.Layer(
    "GeoJsonLayer",
    data=grid_geojson,
    stroked=False,
    filled=True,
    get_fill_color="properties._fill_color",  # ← 여기만 바꾸기
    get_line_color=[0, 0, 0, 0],
    pickable=True,
)


# 2) 체육시설 포인트(ScatterplotLayer) — 카테고리별 on/off
cat_colors = {
    "생활": [37, 99, 235, 220],   # 파랑
    "전문": [16, 185, 129, 220],  # 초록
    "직장": [234, 179, 8, 220],   # 노랑
    "NA":  [107, 114, 128, 220],  # 회색
}

fac_layers = []
if selected_cats:
    fac_sel = fac_df[fac_df["시설구분"].isin(selected_cats)].copy()
    # 포지션 컬럼
    fac_sel = fac_sel.rename(columns={FAC_NAME_COL: "시설명"})
    fac_sel["color"] = fac_sel["시설구분"].apply(lambda k: cat_colors.get(k, [107,114,128,220]))
    fac_layer = pdk.Layer(
        "ScatterplotLayer",
        data=fac_sel,
        get_position='[lon, lat]',
        get_radius=40,
        radius_min_pixels=4,
        radius_max_pixels=60,
        get_fill_color="color",
        pickable=True,
    )
    fac_layers.append(fac_layer)

# ========= 툴팁 =========
tooltip = {
    "html": """
    <div style="font-size:13px;">
      <b>인구(POP)</b>: {POP}<br/>
      <b>시설명</b>: {시설명}<br/>
      <b>시설구분</b>: {시설구분}
    </div>
    """,
    "style": {"backgroundColor": "rgba(255,255,255,0.9)", "color": "black"}
}

# ========= 지도 출력 =========
r = pdk.Deck(
    map_style=None,                       # ← Mapbox 스타일 OFF (토큰 불필요)
    initial_view_state=view_state,
    layers=[tile_layer, grid_layer] + fac_layers,   # ← 타일 → 격자 → 포인트 순서
    tooltip=tooltip,
)


st.pydeck_chart(r)

# ========= 부가정보 =========
with st.expander("데이터 요약", expanded=False):
    st.subheader("체육시설 개수(선택됨)")
    st.write(
        pd.DataFrame(
            fac_df[fac_df["시설구분"].isin(selected_cats)]["시설구분"].value_counts()
        )
    )


st.caption("팁: 좌측 체크박스로 시설 구분 레이어를 토글하세요. 격자에 마우스를 올리면 POP, 포인트에 올리면 시설명이 뜹니다.")
