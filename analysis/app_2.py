
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
from shapely.ops import unary_union
try:
    import pulp
except ImportError:
    st.error("Pulp is not installed. Please install it using: pip install pulp")
    st.stop()

# ----------------- 경로 설정 -----------------
# 현재 스크립트 파일의 디렉토리를 기준으로 상대 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")


# [입력] GTF 분석 결과 파일
GTF_RESULTS_GPKG = os.path.join(DATA_DIR, "cheongju_gtf_results.gpkg")

# [입력] 생활체육시설 원본 데이터
FACILITY_FILE = os.path.join(DATA_DIR, "cheongju_geocoded_google.csv")

# ----------------- 앱 기본 설정 -----------------
st.set_page_config(page_title="청주시 신규 체육시설 최적 입지 분석", layout="wide")
st.title("청주시 신규 체육시설 최적 입지 분석")

# ----------------- 데이터 로딩 (캐싱) -----------------
@st.cache_data(show_spinner="결과 데이터 로딩 중...")
def load_gtf_results(gpkg_path):
    """GTF 분석 결과가 저장된 GeoPackage 파일을 로드합니다."""
    if not os.path.exists(gpkg_path):
        st.error(f"오류: GTF 결과 파일({gpkg_path})을 찾을 수 없습니다. 데이터 생성 스크립트를 먼저 실행하세요.")
        st.stop()
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    return gdf

@st.cache_data(show_spinner="기존 시설 데이터 로딩 중...")
def load_existing_facilities(csv_path):
    """기존 체육시설 데이터를 로드하고 좌표를 파싱합니다."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")

    latlon = df["final_location"].astype(str).str.extract(
        r'^\s*(?P<lat>-?\d+(?:\.\d+)?)\s*,\s*(?P<lon>-?\d+(?:\.\d+)?)\s*$'
    )
    df["lat"] = pd.to_numeric(latlon["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(latlon["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"].between(33, 39)) & (df["lon"].between(124, 132))].copy()
    
    name_col = next((c for c in ["시설명", "name"] if c in df.columns), "시설명")
    df = df.rename(columns={name_col: "name"})
    if "name" not in df.columns:
        df["name"] = "(이름없음)"
        
    
    return df

# ----------------- 최적화 함수 (251030_modified_3.py에서 가져옴) -----------------
def run_location_optimization(grid_gdf, n_facilities, candidate_col, threshold_percentile, service_radius_m):
    """MCLP 모델을 사용하여 최적의 신규 시설 입지를 찾습니다."""
    with st.spinner(f"{n_facilities}개 입지 최적화 중..."):
        PROJ_CRS = "EPSG:5179"
        
        # 1) 수요지: 인구>0인 격자만
        demand_points = grid_gdf[grid_gdf['total_pop'] > 0].copy().to_crs(PROJ_CRS)
        demand_points['demand_id'] = range(len(demand_points))

        # 2) 후보지: 잠재력 상위 percentile 이상
        threshold_value = np.percentile(grid_gdf[candidate_col], threshold_percentile)
        candidate_sites = grid_gdf[grid_gdf[candidate_col] >= threshold_value].copy().to_crs(PROJ_CRS)
        candidate_sites['candidate_id'] = range(len(candidate_sites))

        if candidate_sites.empty:
            st.warning("후보지가 없습니다. 후보지 선정 기준(상위 %)을 낮춰보세요.")
            return gpd.GeoDataFrame(), 0

        # ---- (기존 gid_to_candidate_id 제거) ----
        # gid_to_candidate_id = ...

        # 3) 후보지 버퍼 생성
        candidate_buffers_gdf = candidate_sites.copy()
        candidate_buffers_gdf.geometry = candidate_sites.geometry.buffer(service_radius_m)
        candidate_buffers_gdf.crs = PROJ_CRS

        # 4) 수요지-후보지 커버 관계 (공간조인)
        coverage = gpd.sjoin(
            demand_points,
            candidate_buffers_gdf,
            how='inner',
            predicate='intersects'
        )

        if coverage.empty:
            st.warning("후보지와 수요지 간의 커버리지가 없습니다. 서비스 반경을 늘려보세요.")
            return gpd.GeoDataFrame(), 0

        # 🔴 핵심 수정: demand_id → candidate_id 매핑
        coverage_dict = coverage.groupby('demand_id')['candidate_id'].apply(list).to_dict()

        # 5) 최적화 모형 설정
        prob = pulp.LpProblem("Facility_Location_MCLP", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", candidate_sites['candidate_id'].to_list(), cat='Binary')
        y = pulp.LpVariable.dicts("y", demand_points['demand_id'].tolist(), cat='Binary')

        demand_pop_dict = pd.Series(demand_points.total_pop.values, index=demand_points.demand_id).to_dict()
        prob += pulp.lpSum([demand_pop_dict[i] * y[i] for i in y]), "Total_Covered_Population"
        prob += pulp.lpSum([x[j] for j in x]) == n_facilities, "Num_Facilities_Constraint"

        # 6) 커버 제약: 시설 중 하나라도 커버하면 y[i] = 1 허용
        for i in y:
            candidate_ids = coverage_dict.get(i, [])
            if candidate_ids:
                prob += y[i] <= pulp.lpSum([x[j] for j in candidate_ids]), f"Coverage_Constraint_{i}"
            else:
                prob += y[i] == 0

        # 7) 풀이
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status == pulp.LpStatusOptimal:
            optimal_sites_indices = [j for j in x if x[j].varValue > 0.9]
            optimal_sites_proj = candidate_sites[candidate_sites['candidate_id'].isin(optimal_sites_indices)]
            optimal_sites = optimal_sites_proj.to_crs(grid_gdf.crs)
            covered_pop = pulp.value(prob.objective)
            return optimal_sites, covered_pop
        else:
            st.error("최적화에 실패했습니다. 다른 파라미터를 시도해보세요.")
            return gpd.GeoDataFrame(), 0


# ----------------- 사이드바 UI -----------------
st.sidebar.header("⚙️ 최적화 파라미터")

n_new = st.sidebar.slider("1. 신규 시설 개수 (N)", min_value=1, max_value=20, value=5, step=1)

st.sidebar.markdown("---")
st.sidebar.header("🗺️ 지도 시각화 설정")

potential_col = st.sidebar.selectbox(
    "2. 잠재력 기준 (배경)",
    ['gtf_smoothed', 'gtf_residuals', 'y_imbalance', 'total_pop'],
    index=0,
    help="`gtf_smoothed`: 수요-공급 불균형의 공간적 패턴(핫스팟) / `gtf_residuals`: 주변과 다른 특이점(포켓)"
)

candidate_percentile = st.sidebar.slider(
    "3. 후보지 선정 기준 (상위 %)", 50, 100, 90, 1,
    help="잠재력 점수가 상위 몇 %인 지역을 후보지로 사용할지 결정합니다."
)

service_radius = st.sidebar.slider(
    "4. 서비스 반경 (미터)", 500, 5000, 2000, 100,
    help="신규 시설이 커버할 수 있는 최대 거리를 설정합니다."
)

alpha = st.sidebar.slider("5. 배경 투명도", 0.0, 1.0, 0.5, 0.05)

show_existing = st.sidebar.checkbox("기존 체육시설 표시", value=True)


# ----------------- 메인 로직 -----------------
# 데이터 로드
grid_pop = load_gtf_results(GTF_RESULTS_GPKG)
existing_fac_df = load_existing_facilities(FACILITY_FILE)

# 최적화 실행
optimal_sites_gdf, covered_pop = run_location_optimization(
    grid_gdf=grid_pop,
    n_facilities=n_new,
    candidate_col=potential_col,
    threshold_percentile=candidate_percentile,
    service_radius_m=service_radius
)

# ----------------- 지도 시각화 (Pydeck) -----------------
# 뷰포트 설정
center = grid_pop.unary_union.centroid
view_state = pdk.ViewState(latitude=center.y, longitude=center.x, zoom=11, pitch=45)

# OSM 베이스맵 레이어
tile_layer = pdk.Layer(
    "TileLayer",
    data="https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
    minZoom=0, maxZoom=19, tileSize=256
)

# 1. 배경 잠재력 레이어 (GeoJsonLayer)
# 색상 스케일 계산
vals = grid_pop[potential_col].dropna()
vmin = vals.min()
vmax = np.percentile(vals, 99) if len(vals) > 0 else vmin + 1

def color_scale(x, vmin, vmax, cmap):
    if pd.isna(x) or vmax <= vmin: return [0, 0, 0, 0]
    t = (x - vmin) / (vmax - vmin)
    t = np.clip(t, 0, 1)
    color = cmap(t)
    return [int(c * 255) for c in color[:3]] + [int(alpha * 255)]

from matplotlib.cm import get_cmap
cmap = get_cmap('OrRd')
grid_pop["_fill_color"] = grid_pop[potential_col].apply(lambda x: color_scale(x, vmin, vmax, cmap))

potential_layer = pdk.Layer(
    "GeoJsonLayer",
    data=grid_pop.__geo_interface__,
    stroked=False,
    filled=True,
    get_fill_color="properties._fill_color",
    pickable=True,
)

# 2. 기존 체육시설 레이어 (ScatterplotLayer)
layers = [tile_layer, potential_layer]
if show_existing:
    existing_fac_layer = pdk.Layer(
        "ScatterplotLayer",
        data=existing_fac_df,
        get_position='[lon, lat]',
        get_radius=50,
        get_fill_color=[37, 99, 235, 200], # Blue
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
        pickable=True,
    )
    layers.append(existing_fac_layer)

# 3. 신규 추천 입지 레이어 (ScatterplotLayer)
if not optimal_sites_gdf.empty:
    optimal_sites_df = pd.DataFrame({
        'lon': optimal_sites_gdf.geometry.centroid.x,
        'lat': optimal_sites_gdf.geometry.centroid.y,
        'gid': optimal_sites_gdf.index,
        'score': optimal_sites_gdf[potential_col]
    })
    
    new_sites_layer = pdk.Layer(
        "ScatterplotLayer",
        data=optimal_sites_df,
        get_position='[lon, lat]',
        get_radius=150,
        get_fill_color=[220, 38, 38, 220], # Red
        get_line_color=[0, 0, 0],
        line_width_min_pixels=2,
        pickable=True,
    )
    layers.append(new_sites_layer)

# 툴팁 설정
tooltip = {
    "html": """
    <div style="font-size: 13px; color: black; background-color: rgba(255,255,255,0.9); padding: 8px; border-radius: 4px;">
      <b>{properties.gid}</b><br/>
      인구: {properties.total_pop}<br/>
      수요-공급 불균형: {properties.y_imbalance:.2f}<br/>
      GTF(핫스팟): {properties.gtf_smoothed:.2f}<br/>
      GTF(특이점): {properties.gtf_residuals:.2f}<br/>
      <hr style="margin: 4px 0;"/>
      <b>{name}</b><br/>
      <b style="color:red;">추천 입지 점수: {score:.2f}</b>
    </div>
    """,
    "style": {"backgroundColor": None, "border": None}
}


# 지도 렌더링
r = pdk.Deck(
    map_style=None,
    initial_view_state=view_state,
    layers=layers,
    tooltip=tooltip,
)
st.pydeck_chart(r)

# ----------------- 요약 정보 -----------------
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 최적화 결과 요약")
    total_pop = grid_pop['total_pop'].sum()
    if not optimal_sites_gdf.empty:
        coverage_percent = (covered_pop / total_pop) * 100 if total_pop > 0 else 0
        st.metric(f"선택된 {n_new}개 신규 입지의 예상 커버 인구", f"{int(covered_pop):,} 명", f"{coverage_percent:.1f}% of Total")
    else:
        st.info("선택된 입지가 없습니다.")

with col2:
    st.subheader("📍 추천 입지 목록 (Top 5)")
    if not optimal_sites_gdf.empty:
        st.dataframe(
            optimal_sites_gdf[[potential_col, 'total_pop', 'y_imbalance']]
            .sort_values(by=potential_col, ascending=False)
            .head(5)
            .style.format('{:.2f}')
        )
    else:
        st.info("선택된 입지가 없습니다.")

st.caption("지도 범례: 붉은색 배경(잠재력 높음), 파란 점(기존 시설), 붉은 큰 점(신규 추천 시설)")
