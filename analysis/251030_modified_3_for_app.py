import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import geopandas as gpd
import numpy as np
import tensorflow as tf
from shapely.ops import unary_union

# --- 1. 경로 및 설정 ---

ROOT_DIR = "C:/Users/USER/Desktop/공모전/충남대/python"
DATA_DIR = os.path.join(ROOT_DIR, "/data")
ANALYSIS_DIR = os.path.join(ROOT_DIR, "/analysis")


# [입력] 격자 및 인구 데이터 경로
SIG_SHAPE = os.path.join(ROOT_DIR, "/bnd_sigungu_00_2024_2Q/bnd_sigungu_00_2024_2Q.shp")
GRID_A = os.path.join(ROOT_DIR, "/_grid_border_grid_2024_grid_다바_grid_다바/grid_다바_500M.shp")
GRID_B = os.path.join(ROOT_DIR, "/_grid_border_grid_2024_grid_라바_grid_라바/grid_라바_500M.shp")
POP_A  = os.path.join(ROOT_DIR, "/_census_reqdoc_1758606603346/2023년_인구_다바_500M.txt")
POP_B  = os.path.join(ROOT_DIR, "/_census_reqdoc_1758606603346/2023년_인구_라바_500M.txt")

# [입력] 생활체육시설 공급 데이터
FACILITY_FILE = os.path.join(ANALYSIS_DIR, "cheongju_community_facilities.csv")

# [출력] GTF 결과 GeoPackage 파일
OUTPUT_GPKG = os.path.join(DATA_DIR, "cheongju_gtf_results.gpkg")
os.makedirs(DATA_DIR, exist_ok=True)


# [설정] GTF 하이퍼파라미터
LAMBDA = 1.0
STEPS = 500
LEARNING_RATE = 0.01

# [설정] 공급(Supply) 정의
BUFFER_RADIUS_METERS = 2000 # 서비스 권역 반경 2km
SUPPLY_EPSILON = 0.01       # 0으로 나누기 방지


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
    grid_gdf['supply_ratio'] = grid_gdf['supply_ratio'].clip(0, 1)
    
    print("Supply calculation complete.")
    return grid_gdf

def build_graph(grid_gdf):
    """
    GeoDataFrame에서 인접성(Queen)을 기반으로 그래프 엣지를 구축합니다.
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

# --- Main 실행 로직 ---
def main():
    # 1. 수요(Demand) 및 격자(Grid) 로드
    grid_pop = load_demand_and_grid(SIG_SHAPE, [GRID_A, GRID_B], [POP_A, POP_B])
    
    # 2. 공급(Supply) 계산
    grid_pop = calculate_supply(grid_pop, FACILITY_FILE, BUFFER_RADIUS_METERS)
    
    # 3. 그래프(Graph) 구축
    (N, E, senders, receivers) = build_graph(grid_pop)
    
    # 4. 최종 불균형 신호(Y) 정의
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
    
    # 7. GeoPackage 파일로 저장
    print(f"Saving GTF results to '{OUTPUT_GPKG}'...")
    # For gpkg, need to convert column names to string
    grid_pop.columns = grid_pop.columns.astype(str)
    grid_pop.to_file(OUTPUT_GPKG, driver='GPKG')
    
    print(f"\n--- Analysis complete. Results saved to '{OUTPUT_GPKG}' ---")

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


