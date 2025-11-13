

import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# 0) 입력 ------------------------------
# 관심 좌표 (위도, 경도)
lat, lon = 36.64210, 127.48900
# 네트워크 종류: 보행 "walk" / 차량 "drive" / 자전거 "bike"
NETWORK_TYPE = "drive"

# 등거리/등시간 임계값
DIST_MAX_M = 2000      # 네트워크 거리 2 km
TIME_MAX_MIN = 10      # 네트워크 시간 10 분

# 1) 관심 영역 그래프 내려받기 (포인트 반경으로 가볍게)
G = ox.graph_from_point((lat, lon), dist=4000, network_type=NETWORK_TYPE, simplify=True)

# 2) 투영(미터 단위로 계산 안정화) ------------------------------
Gp = ox.project_graph(G)                 # 그래프 투영
crs_proj = Gp.graph["crs"]

# 3) 출발 노드 스냅 --------------------------------------------
src = ox.nearest_nodes(Gp, X=gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
                                .to_crs(crs_proj).iloc[0].x,
                          Y=gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
                                .to_crs(crs_proj).iloc[0].y)

# 4A) 네트워크 '거리' 기반 등거리 버퍼 -------------------------
#    weight='length' 는 투영 그래프에서 미터 단위
distances = nx.single_source_dijkstra_path_length(Gp, source=src, weight="length")
nodes_within_dist = {n for n, d in distances.items() if d <= DIST_MAX_M}

# 4B) 네트워크 '시간' 기반 등시간 버퍼 -------------------------
#    차량/자전거는 도로속도 태그 기반 travel_time(초) 생성
Gp = ox.add_edge_speeds(Gp)          # 'speed_kph'
Gp = ox.add_edge_travel_times(Gp)     # 'travel_time' (초)
times = nx.single_source_dijkstra_path_length(Gp, source=src, weight="travel_time")
nodes_within_time = {n for n, t in times.items() if t <= TIME_MAX_MIN * 60.0}

# 5) 서브그래프 & GeoDataFrame 추출 ----------------------------
def subgraph_by_nodes(Gproj, keep_nodes):
    # 노드 집합으로 유 induced subgraph 만들고 고립 제거
    H = Gproj.subgraph(keep_nodes).copy()
    # 가장 큰 연결요소만 남기면 경계가 깔끔
    if not nx.is_empty(H):
        largest_cc = max(nx.connected_components(H.to_undirected()), key=len)
        H = H.subgraph(largest_cc).copy()
    return H

Gdist = subgraph_by_nodes(Gp, nodes_within_dist)
Gtime = subgraph_by_nodes(Gp, nodes_within_time)

nodes_d, edges_d = ox.graph_to_gdfs(Gdist)
nodes_t, edges_t = ox.graph_to_gdfs(Gtime)

# 6) 폴리곤(isochrone) 만들기 ----------------------------------
# 방법 1: 엣지 라인들의 union → 약간 버퍼 → 외곽 폴리곤
def isochrone_polygon(edges_gdf, buffer_m=25):
    if len(edges_gdf) == 0:
        return None
    # 라인 유니온 후 얇게 버퍼 → 폴리곤화
    merged = edges_gdf.geometry.unary_union
    poly = gpd.GeoSeries([merged], crs=edges_gdf.crs).buffer(buffer_m).unary_union
    return poly

iso_poly_dist = isochrone_polygon(edges_d, buffer_m=25)
iso_poly_time = isochrone_polygon(edges_t, buffer_m=25)

# 7) 시각화 -----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# A) 등거리
if iso_poly_dist:
    gpd.GeoSeries([iso_poly_dist], crs=crs_proj).plot(ax=ax[0], facecolor="#c6e2ff", edgecolor="#3399ff", alpha=0.6)
edges_d.plot(ax=ax[0], linewidth=0.6, color="#555555")
gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_proj).plot(ax=ax[0], color="gold", markersize=50, edgecolor="black")
ax[0].set_title(f"Network Distance Isochrone ≤ {DIST_MAX_M/1000:.1f} km")
ax[0].set_axis_off()

# B) 등시간
if iso_poly_time:
    gpd.GeoSeries([iso_poly_time], crs=crs_proj).plot(ax=ax[1], facecolor="#d6f5d6", edgecolor="#33aa33", alpha=0.6)
edges_t.plot(ax=ax[1], linewidth=0.6, color="#555555")
gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs_proj).plot(ax=ax[1], color="gold", markersize=50, edgecolor="black")
ax[1].set_title(f"Network Time Isochrone ≤ {TIME_MAX_MIN} min")
ax[1].set_axis_off()

plt.tight_layout()
plt.show()




import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

pt_wgs = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
pt_proj = pt_wgs.to_crs(Gp.graph["crs"])
circle2km = pt_proj.buffer(2000)  # 직선 2 km 원(지도 비교용)

# 등거리 폴리곤(iso_poly_dist)과 같이 그림
ax = edges_d.plot(figsize=(8,8), linewidth=0.6, color="#666")
gpd.GeoSeries([iso_poly_dist], crs=Gp.graph["crs"]).plot(ax=ax, facecolor="#c6e2ff", alpha=0.5, edgecolor="#3399ff")
circle2km.plot(ax=ax, facecolor="none", edgecolor="crimson", linewidth=1.2, linestyle="--")
pt_proj.plot(ax=ax, color="gold", markersize=50, edgecolor="black")
ax.set_axis_off(); plt.show()














# pip install osmnx networkx geopandas shapely matplotlib

import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# ----------------------- 입력 -----------------------
lat, lon = 36.64210, 127.48900   # 관심 좌표 (위도, 경도)
NETWORK_TYPE = "drive"           # "walk" / "drive" / "bike"
DIST_MAX_M = 2000                # 등거리 임계(네트워크 거리) 2 km
GRAPH_FETCH_DIST = 4000          # 그래프 다운로드 반경(여유 있게)
# ---------------------------------------------------

# 1) 관심점 반경으로 그래프 받기 (가볍고 빠름)
G = ox.graph_from_point((lat, lon), dist=GRAPH_FETCH_DIST, network_type=NETWORK_TYPE, simplify=True)

# 2) 투영(미터 단위 계산 안정화)
Gp = ox.project_graph(G)                 # projected graph (CRS in meters)
crs_proj = Gp.graph["crs"]

# 3) 출발 노드 스냅
pt_wgs = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
pt_proj = pt_wgs.to_crs(crs_proj)
src = ox.nearest_nodes(Gp, X=pt_proj.iloc[0].x, Y=pt_proj.iloc[0].y)

# 4) 단일 소스 → 모든 노드 최단 '거리(미터)'
distances = nx.single_source_dijkstra_path_length(Gp, source=src, weight="length")
nodes_within = {n for n, d in distances.items() if d <= DIST_MAX_M}

# 5) 2 km 이내 노드들로 서브그래프 만들기(가장 큰 연결요소만 사용하면 경계가 깔끔)
def subgraph_by_nodes(Gproj, keep_nodes):
    H = Gproj.subgraph(keep_nodes).copy()
    if not nx.is_empty(H):
        largest_cc = max(nx.connected_components(H.to_undirected()), key=len)
        H = H.subgraph(largest_cc).copy()
    return H

G_iso = subgraph_by_nodes(Gp, nodes_within)
nodes_all, edges_all = ox.graph_to_gdfs(Gp)
nodes_iso, edges_iso = ox.graph_to_gdfs(G_iso)

# 6) 이소크론 폴리곤(선 유니온 → 얇은 버퍼로 면화)
def isochrone_polygon(edges_gdf, buffer_m=25):
    if len(edges_gdf) == 0:
        return None
    merged = edges_gdf.geometry.unary_union
    poly = gpd.GeoSeries([merged], crs=edges_gdf.crs).buffer(buffer_m).unary_union
    return poly

iso_poly = isochrone_polygon(edges_iso, buffer_m=20)

# 7) 시각화
fig, ax = plt.subplots(figsize=(9, 9))

# 전체 도로(옅은 회색)
edges_all.plot(ax=ax, linewidth=0.5, color="#BBBBBB", alpha=0.7)

# 이소크론 폴리곤(하늘색 반투명)
if iso_poly:
    gpd.GeoSeries([iso_poly], crs=crs_proj).plot(ax=ax, facecolor="#c6e2ff", edgecolor="#3399ff", alpha=0.5)

# 2km 이내 도로(진한 회색)
edges_iso.plot(ax=ax, linewidth=1.0, color="#555555")

# 전체 노드(연한 점)
nodes_all.plot(ax=ax, markersize=2, color="#CCCCCC", alpha=0.6)

# 2km 이내 노드(파란 점)
if len(nodes_iso) > 0:
    nodes_iso.plot(ax=ax, markersize=6, color="#1976D2", alpha=0.9)

# 출발점(금색 마커)
pt_proj.plot(ax=ax, color="gold", edgecolor="black", markersize=80, zorder=5)

# 참고용: 직선 2km 원도 같이 보고 싶으면 주석 해제
# circle2km = pt_proj.buffer(DIST_MAX_M)
# circle2km.plot(ax=ax, facecolor="none", edgecolor="crimson", linewidth=1.2, linestyle="--")

ax.set_title(f"Network Isochrone ≤ {DIST_MAX_M/1000:.1f} km (type={NETWORK_TYPE})", fontsize=12)
ax.set_axis_off()
plt.tight_layout()
plt.show()

# (선택) 결과 저장
# nodes_iso.to_file("iso_nodes.gpkg", layer="nodes", driver="GPKG")
# edges_iso.to_file("iso_edges.gpkg", layer="edges", driver="GPKG")



