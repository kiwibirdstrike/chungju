
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import geopandas as gpd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import contextily as cx
from shapely.ops import unary_union
try:
    import pulp
except ImportError:
    print("Pulp is not installed. Please install it using: pip install pulp")
    exit()

# --- 1. 경로 및 설정 (수정됨) ---

ROOT_DIR = "/Users/dongyounglee/Library/CloudStorage/GoogleDrive-dtsy5891@gmail.com/내 드라이브/Coursework/2025 충청권의 미래"

# [입력] 격자 및 인구 데이터 경로
SIG_SHAPE = os.path.join(ROOT_DIR, "250923/bnd_sigungu_00_2024_2Q/bnd_sigungu_00_2024_2Q.shp")
GRID_A = os.path.join(ROOT_DIR, "250923/_grid_border_grid_2024_grid_다바_grid_다바/grid_다바_500M.shp")
GRID_B = os.path.join(ROOT_DIR, "250923/_grid_border_grid_2024_grid_라바_grid_라바/grid_라바_500M.shp")
POP_A  = os.path.join(ROOT_DIR, "250923/_census_reqdoc_1758606603346/2023년_인구_다바_500M.txt")
POP_B  = os.path.join(ROOT_DIR, "250923/_census_reqdoc_1758606603346/2023년_인구_라바_500M.txt")

# [입력] 생활체육시설 공급 데이터 (활성화)
FACILITY_FILE = os.path.join(ROOT_DIR, "251019/python/analysis/cheongju_community_facilities.csv")

# [출력] 결과 맵이 저장될 폴더
FIG_DIR = "output_maps_v3"
os.makedirs(FIG_DIR, exist_ok=True)

# [설정] GTF 하이퍼파라미터
LAMBDA = 1.0
STEPS = 500
LEARNING_RATE = 0.01

# [설정] 공급(Supply) 정의 (활성화)
BUFFER_RADIUS_METERS = 2000 # 서비스 권역 반경 2km
SUPPLY_EPSILON = 0.01       # 0으로 나누기 방지

# [설정] 최적화 모델 파라미터 (신규 추가)
N_NEW_FACILITIES = 5  # 신규로 건설할 시설의 수
CANDIDATE_THRESHOLD_PERCENTILE = 90  # 후보지 선정 기준 (상위 %)
OPTIMIZATION_TARGET_COLUMN = 'gtf_smoothed' # 최적화에 사용할 GTF 결과 (gtf_smoothed 또는 gtf_residuals)


# -------------------------------------------------------------------------

def setup_korean_font():
    """Matplotlib에서 한글 폰트를 설정합니다."""
    try:
        font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
        print(f"Korean font '{font_name}' set up for plotting.")
    except:
        print("Korean font not found. Skipping font setup.")

def load_demand_and_grid(sig_path, grid_paths, pop_paths):
    """
    Sigungu, Grid, Population 데이터를 로드하고 Cheongju 지역으로 클리핑합니다.
    """
    print("Loading sigungu shapefile...")
    sig = gpd.read_file(sig_path)
    cheongju_codes = ["33041", "33042", "33043", "33044"]
    cheongju_one = sig[sig["SIGUNGU_CD"].isin(cheongju_codes)].dissolve()

    print("Loading grid shapefiles...")
    gdfs = [gpd.read_file(path) for path in grid_paths]
    grid_all = pd.concat(gdfs, ignore_index=True)

    if grid_all.crs != cheongju_one.crs:
        grid_all = grid_all.to_crs(cheongju_one.crs)

    if "GRID_500M_" in grid_all.columns:
        grid_all = grid_all.rename(columns={"GRID_500M_": "gid"})
    elif "GRID_500M" in grid_all.columns:
        grid_all = grid_all.rename(columns={"GRID_500M": "gid"})

    grid_all["gid"] = grid_all["gid"].astype(str).str.strip()
    grid_all = grid_all.dropna(subset=["gid"])
    grid_all_unique = grid_all.drop_duplicates(subset='gid')

    print("Clipping grid to Cheongju area...")
    grid_cj = gpd.clip(grid_all_unique, cheongju_one)
    grid_cj = grid_cj.set_index('gid')

    print("Loading population text files...")
    pop_dfs = []
    for path in pop_paths:
        try:
            df = pd.read_csv(path, sep='^', header=None, names=['year', 'gid', 'type', 'pop'], dtype={'gid': str})
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep='^', header=None, names=['year', 'gid', 'type', 'pop'], encoding='cp949', dtype={'gid': str})
        pop_dfs.append(df)
    pop_df = pd.concat(pop_dfs, ignore_index=True)
    pop_df['gid'] = pop_df['gid'].str.strip()

    total_pop_df = pop_df[pop_df['type'] == 'to_in_001'].pivot(index='gid', columns='type', values='pop').fillna(0)
    total_pop_df.columns = ['total_pop']
    total_pop_df['total_pop'] = pd.to_numeric(total_pop_df['total_pop'], errors='coerce').fillna(0)

    grid_pop = grid_cj.join(total_pop_df, how='left').fillna(0)
    print(f"Loaded and processed {len(grid_pop)} grid cells for Cheongju.")
    return grid_pop

def calculate_supply(grid_gdf, facility_path, buffer_radius_m):
    """
    생활체육시설 위치를 기준으로 각 격자(grid)의 공급 비율(0~1)을 계산합니다.
    """
    print(f"Calculating supply based on '{facility_path}'...")
    facilities_df = pd.read_csv(facility_path)
    facilities_gdf = gpd.GeoDataFrame(
        facilities_df, 
        geometry=gpd.points_from_xy(facilities_df.lon, facilities_df.lat),
        crs="EPSG:4326"
    )
    
    facilities_gdf = facilities_gdf.to_crs(grid_gdf.crs)
    
    print(f"Creating {buffer_radius_m}m buffers for {len(facilities_gdf)} facilities...")
    buffers = facilities_gdf.geometry.buffer(buffer_radius_m)
    all_buffers_union = unary_union(buffers)
    
    grid_gdf['grid_area'] = grid_gdf.geometry.area
    
    print("Calculating intersection area (this may take a moment)...")
    intersection_area = grid_gdf.geometry.intersection(all_buffers_union).area
    
    grid_gdf['supply_ratio'] = (intersection_area / grid_gdf['grid_area']).fillna(0)
    grid_gdf['supply_ratio'] = grid_gdf['supply_ratio'].clip(0, 10)
    
    print("Supply calculation complete.")
    return grid_gdf

def build_graph(grid_gdf):
    """
    GeoDataFrame에서 인접성(Queen)을 기반으로 그래프 엣지를 구축합니다. (수정된 버전)
    """
    print("Building graph (spatial join)...")
    grid_gdf_reset = grid_gdf.reset_index()
    N = len(grid_gdf_reset)
    neighbors = gpd.sjoin(grid_gdf_reset, grid_gdf_reset, how="inner", predicate="touches")
    neighbors = neighbors[neighbors.index != neighbors['index_right']]
    senders = neighbors.index.values
    receivers = neighbors['index_right'].values
    E = len(senders)
    print(f"Graph built: {N} nodes, {E} edges.")
    return N, E, senders, receivers

def run_gtf_gpu(y_signal_tf, N, E, senders, receivers, lambda_val, steps, learning_rate):
    """
    TensorFlow (GPU)를 사용하여 GTF 최적화를 수행합니다.
    """
    print("Running GTF optimization on GPU...")
    beta = tf.Variable(tf.zeros(N, dtype=tf.float32), name="beta")
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss_fidelity = tf.reduce_sum(tf.square(y_signal_tf - beta)) * 0.5
            beta_i = tf.gather(beta, senders)
            beta_j = tf.gather(beta, receivers)
            loss_penalty = lambda_val * tf.reduce_sum(tf.abs(beta_i - beta_j))
            total_loss = loss_fidelity + loss_penalty
        gradients = tape.gradient(total_loss, [beta])
        optimizer.apply_gradients(zip(gradients, [beta]))
        return total_loss

    for step in range(steps):
        loss = train_step()
        if (step + 1) % (steps // 10) == 0:
            print(f"Step {step+1}/{steps}, Loss: {loss.numpy():.2f}")
    print("GTF optimization complete.")
    return beta.numpy()

def plot_and_save(gdf, column, cmap, filename, vmin=None, vmax=None): # vmin, vmax 추가
    """결과 맵을 플로팅하고 저장합니다. (수정됨: vmin, vmax 지원)"""
    print(f"Plotting and saving: {filename}")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf_3857 = gdf.to_crs(epsg=3857)
    
    gdf_3857.plot(
        column=column, 
        cmap=cmap, 
        legend=True, 
        ax=ax, 
        alpha=0.7,
        legend_kwds={'shrink': 0.8},
        vmin=vmin,  # ❗️ 스케일 고정을 위해 추가
        vmax=vmax   # ❗️ 스케일 고정을 위해 추가
    )
    
    ax.set_axis_off()
    cx.add_basemap(ax, crs=gdf_3857.crs.to_string(), source=cx.providers.CartoDB.Positron)
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_location_optimization(grid_gdf, n_facilities, candidate_col, threshold_percentile, service_radius_m):
    """
    MCLP (Maximal Covering Location Problem) 모델을 사용하여 최적의 신규 시설 입지를 찾습니다.
    (이전 LSCP 주석을 MCLP로 수정)
    """
    print("\n--- Running Location Optimization (MCLP) ---")
    
    # 💡 [해결책 1] 사용할 '미터(m) 기반' 투영 좌표계를 정의합니다. (예: EPSG:5179)
    PROJ_CRS = "EPSG:5179" 
    
    # 1. 수요지(Demand Points) 정의
    demand_points = grid_gdf[grid_gdf['total_pop'] > 0].copy()
    # 💡 [해결책 2] 수요지를 미터 기반 좌표계로 변환합니다.
    demand_points = demand_points.to_crs(PROJ_CRS)
    demand_points['demand_id'] = range(len(demand_points))
    print(f"Defined {len(demand_points)} demand points (grids with population > 0).")

    # 2. 후보지(Candidate Sites) 선정
    threshold_value = np.percentile(grid_gdf[candidate_col], threshold_percentile)
    candidate_sites = grid_gdf[grid_gdf[candidate_col] >= threshold_value].copy()
    # 💡 [해결책 3] 후보지를 미터 기반 좌표계로 변환합니다.
    candidate_sites = candidate_sites.to_crs(PROJ_CRS)
    candidate_sites['candidate_id'] = range(len(candidate_sites))
    print(f"Selected {len(candidate_sites)} candidate sites (top {100-threshold_percentile}% of '{candidate_col}').")

    # 후보지 gid를 candidate_id로 매핑하는 딕셔너리 생성
    # (이제 candidate_sites의 인덱스(gid)를 사용합니다)
    gid_to_candidate_id = pd.Series(candidate_sites.candidate_id.values, index=candidate_sites.index).to_dict()

    # 3. 커버리지 매트릭스 생성
    print(f"Creating coverage matrix (service radius: {service_radius_m}m)...")
    
    # 후보지의 버퍼를 생성 (이제 PROJ_CRS 상에서 2000 '미터' 버퍼가 정확히 생성됨)
    candidate_buffers_gdf = candidate_sites.copy()
    candidate_buffers_gdf.geometry = candidate_sites.geometry.buffer(service_radius_m)
    # (버퍼 후 CRS가 손실될 수 있으므로 명시적으로 재할당)
    candidate_buffers_gdf.crs = PROJ_CRS

    # sjoin을 사용 (이제 demand_points와 candidate_buffers_gdf 모두 동일한 PROJ_CRS를 가짐)
    coverage = gpd.sjoin(demand_points, candidate_buffers_gdf, how='inner', predicate='intersects')
    
    if coverage.empty:
        print("\n[!!!] CRITICAL ERROR: sjoin returned an empty result even with CRS projection.")
        print("Please check if candidate/demand points are in the same region.")
        return gpd.GeoDataFrame() 

    # 각 수요지(demand_id)를 커버하는 후보지(gid) 리스트 생성
    # (coverage의 index_right에는 candidate_buffers_gdf의 인덱스(gid)가 저장됩니다)
    coverage_dict = coverage.groupby('demand_id')['gid_right'].apply(list).to_dict()

    # 4. 최적화 모델 수립 (PuLP)
    print("Setting up PuLP optimization model...")
    prob = pulp.LpProblem("Facility_Location_MCLP", pulp.LpMaximize)

    # 결정 변수
    x = pulp.LpVariable.dicts("x", candidate_sites['candidate_id'].to_list(), cat='Binary')
    y = pulp.LpVariable.dicts("y", [d['demand_id'] for _, d in demand_points.iterrows()], cat='Binary')

    # 목적 함수: 커버되는 총 인구수 최대화
    # (demand_points가 projected되었으므로 .loc[]를 사용하여 정확한 인구 참조)
    demand_pop_dict = pd.Series(demand_points.total_pop.values, index=demand_points.demand_id).to_dict()
    prob += pulp.lpSum([demand_pop_dict[i] * y[i] for i in y]), "Total_Covered_Population"

    # 제약 조건
    # 1) 신규 시설은 정확히 N개만 건설
    prob += pulp.lpSum([x[j] for j in x]) == n_facilities, "Num_Facilities_Constraint"

    # 2) 수요지 i가 커버되려면, 그 수요지를 커버하는 후보지 중 적어도 하나에 시설이 건설되어야 함
    for i in y:
        candidate_gids_for_demand_i = coverage_dict.get(i, [])
        if candidate_gids_for_demand_i:
            # gid를 candidate_id로 변환
            candidate_ids_for_demand_i = [gid_to_candidate_id[gid] for gid in candidate_gids_for_demand_i if gid in gid_to_candidate_id]
            
            if candidate_ids_for_demand_i: 
                 prob += y[i] <= pulp.lpSum([x[j] for j in candidate_ids_for_demand_i]), f"Coverage_Constraint_{i}"
            else:
                 prob += y[i] == 0 
        else:
            prob += y[i] == 0

    # 5. 모델 실행
    print("Solving optimization problem...")
    prob.solve()
    print(f"Solver status: {pulp.LpStatus[prob.status]}")
    
    if prob.status != pulp.LpStatusOptimal:
        print("[!!!] Optimization FAILED or was not optimal.")
        return gpd.GeoDataFrame()

    # 6. 결과 추출
    optimal_sites_indices = [j for j in x if x[j].varValue > 0.9]
    optimal_sites_proj = candidate_sites[candidate_sites['candidate_id'].isin(optimal_sites_indices)]
    
    # 💡 [해결책 4] 최종 결과를 원래 좌표계로 되돌려 시각화에 사용합니다.
    optimal_sites = optimal_sites_proj.to_crs(grid_gdf.crs)
    
    total_pop = demand_points['total_pop'].sum()
    covered_pop = pulp.value(prob.objective)
    coverage_percentage = (covered_pop / total_pop) * 100 if total_pop > 0 else 0
    
    print(f"\nOptimization Results:")
    print(f" - Selected {len(optimal_sites)} new facility locations.")
    print(f" - Total population in Cheongju grids: {total_pop:,.0f}")
    print(f" - Population covered by new facilities: {covered_pop:,.0f}")
    print(f" - Coverage percentage: {coverage_percentage:.2f}%")
    
    return optimal_sites

def plot_optimization_results(base_gdf, existing_facilities_path, new_facility_sites, target_col, filename):
    """최적화 결과를 시각화합니다. (수정됨: 마커 일치, 제목 제거)"""
    print(f"Plotting and saving optimization results: {filename}")
    
    # 기존 시설 로드
    existing_fac_df = pd.read_csv(existing_facilities_path)
    existing_fac_gdf = gpd.GeoDataFrame(
        existing_fac_df, 
        geometry=gpd.points_from_xy(existing_fac_df.lon, existing_fac_df.lat),
        crs="EPSG:4326"
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    base_gdf_3857 = base_gdf.to_crs(epsg=3857)
    existing_fac_3857 = existing_fac_gdf.to_crs(epsg=3857)
    
    # ❗️ new_facility_sites가 비어있는 경우(최적화 실패 등)를 대비
    if not new_facility_sites.empty:
        new_sites_3857 = new_facility_sites.to_crs(epsg=3857)
    else:
        # 비어있는 GeoDataFrame을 생성하여 오류 방지
        new_sites_3857 = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")


    # 1. 배경 지도 (GTF 결과)
    base_gdf_3857.plot(
        column=target_col, cmap='OrRd', legend=True, ax=ax, alpha=0.7,
        legend_kwds={'shrink': 0.8, 'label': f"Potential Score ({target_col})"}
    )
    
    # 2. 기존 시설 위치
    existing_fac_3857.plot(ax=ax, marker='o', color='blue', markersize=30, label='Existing Facilities', alpha=0.8, edgecolor='white', zorder=5)

    # 3. 최적 신규 입지 (❗️수정된 부분)
    # .centroid를 사용하여 폴리곤이 아닌 중심점에 마커를 찍도록 수정
    if not new_sites_3857.empty:
        new_sites_3857.centroid.plot(
            ax=ax, 
            marker='*', 
            color='red', 
            markersize=200, 
            label='Optimal New Locations', 
            edgecolor='black',
            zorder=10  # ❗️ zorder를 추가하여 항상 위에 보이도록 함
        )

    # ax.set_title(f'Optimal Locations for {len(new_facility_sites)} New Facilities') # ❗️ 요청에 따라 제목 제거
    ax.set_axis_off()
    cx.add_basemap(ax, crs=base_gdf_3857.crs.to_string(), source=cx.providers.CartoDB.Positron)
    
    # 범례 핸들 생성
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Existing Facilities', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Optimal New Locations', markerfacecolor='red', markersize=15, markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12)

    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    
    
# --- 4. Main 실행 로직 ---
def main():
    setup_korean_font()
    
    # 1. 수요(Demand) 및 격자(Grid) 로드
    grid_pop = load_demand_and_grid(SIG_SHAPE, [GRID_A, GRID_B], [POP_A, POP_B])
    
    # 2. 공급(Supply) 계산 (활성화)
    grid_pop = calculate_supply(grid_pop, FACILITY_FILE, BUFFER_RADIUS_METERS)
    
    # 3. 그래프(Graph) 구축
    (N, E, senders, receivers) = build_graph(grid_pop)
    
    # 4. 최종 불균형 신호(Y) 정의 (수요/공급 모두 사용)
    demand = grid_pop['total_pop'].values
    supply = grid_pop['supply_ratio'].values
    y_signal = np.log1p(demand) - np.log(supply + SUPPLY_EPSILON)
    y_signal[demand == 0] = 0 
    grid_pop['y_imbalance'] = y_signal
    
    # 5. GTF 모델 실행 (GPU)
    y_signal_tf = tf.constant(y_signal, dtype=tf.float32)
    beta_smoothed = run_gtf_gpu(
        y_signal_tf, N, E, senders, receivers, 
        LAMBDA, STEPS, LEARNING_RATE
    )
    
    # 6. 결과 저장
    grid_pop['gtf_smoothed'] = beta_smoothed
    grid_pop['gtf_residuals'] = y_signal - beta_smoothed
    
    # --- 7. 결과 시각화 및 저장 (❗️ 스케일 통일 작업) ---
    
    print("Saving sequential maps (1, 2, 4) with 'OrRd' colormap...")
    # (1, 2, 4번은 붉은 계열 유지)
    plot_and_save(grid_pop, 'total_pop', 'OrRd', '01_original_demand.png') 
    plot_and_save(grid_pop, 'supply_ratio', 'OrRd', '02_original_supply.png')
    plot_and_save(grid_pop, 'gtf_smoothed', 'OrRd', '04_GTF_Beta_Hotspots.png')

    # ❗️ [신규] 3번과 5번의 스케일 통일을 위한 최대 절대값 계산
    print("Calculating unified scale for diverging maps (3, 5)...")
    all_diverging_values = pd.concat([grid_pop['y_imbalance'], grid_pop['gtf_residuals']])
    v_abs_max = all_diverging_values.abs().max()
    print(f"Unified scale set to: vmin={-v_abs_max:.2f}, vmax={v_abs_max:.2f}")

    # ❗️ [수정] 3번과 5번 플롯 저장 (cmap 복원 및 vmin/vmax 적용)
    
    # ❗️ 3번: 'coolwarm' 복원, 스케일 고정
    plot_and_save(
        grid_pop, 'y_imbalance', 'coolwarm', '03_original_imbalance_Y.png',
        vmin=-v_abs_max, 
        vmax=v_abs_max
    )
    
    # ❗️ 5번: 'coolwarm' 복원, 스케일 고정
    plot_and_save(
        grid_pop, 'gtf_residuals', 'coolwarm', '05_GTF_Residuals_Pockets.png',
        vmin=-v_abs_max, 
        vmax=v_abs_max
    )
    # (참고: 기존 5번 'PuOr' 대신 'coolwarm'으로 통일)

    # --- 8. 신규: 최적 입지 분석 및 시각화 ---
    optimal_locations = run_location_optimization(
        grid_gdf=grid_pop,
        n_facilities=N_NEW_FACILITIES,
        candidate_col=OPTIMIZATION_TARGET_COLUMN,
        threshold_percentile=CANDIDATE_THRESHOLD_PERCENTILE,
        service_radius_m=BUFFER_RADIUS_METERS
    )

    print("\n\n--- 📍 최종 선정된 최적 입지 (수동 확인) 📍 ---")
    if optimal_locations.empty:
        print(" [결과] 최적화에 실패했거나 선정된 입지가 없습니다.")
    else:
        print(f" [결과] 총 {len(optimal_locations)}개의 최적 입지를 선정했습니다.")
        
        # 1. 선정된 격자의 GID (고유 ID)
        print("\n [1. 선정된 격자 GID]")
        print(list(optimal_locations.index))
        
        # 2. 선정된 격자의 주요 데이터 (GTF 점수, 인구 등)
        print("\n [2. 선정된 격자 상세 정보]")
        print(optimal_locations[[OPTIMIZATION_TARGET_COLUMN, 'total_pop', 'y_imbalance', 'supply_ratio']])
        
        # 3. 구글맵 등에서 확인할 수 있는 위도/경도 좌표 (중심점)
        try:
            print("\n [3. 선정된 격자 중심 위도/경도 (EPSG:4326)]")
            optimal_locations_4326 = optimal_locations.to_crs(epsg=4326)
            for gid, row in optimal_locations_4326.iterrows():
                print(f" - GID {gid}: (Lat: {row.geometry.centroid.y:.6f}, Lon: {row.geometry.centroid.x:.6f})")
            print("   (위 주소를 복사하여 구글맵에 '위도, 경도' 형식으로 붙여넣기)")
                
        except Exception as e:
            print(f"   (위도/경도 변환 중 오류 발생: {e})")
            
    print("------------------------------------------------------\n\n")
    # ================================================================

    plot_optimization_results(
        base_gdf=grid_pop,
        existing_facilities_path=FACILITY_FILE,
        new_facility_sites=optimal_locations,
        target_col=OPTIMIZATION_TARGET_COLUMN,
        filename=f'06_Optimized_{N_NEW_FACILITIES}_New_Locations.png'
    )

    print(f"\n--- 모든 분석이 완료되었습니다. '{FIG_DIR}' 폴더를 확인하세요. ---")
    
    
    

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(e)
    else:
        print("Running on CPU.")
    main()
