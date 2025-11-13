# -*- coding: utf-8 -*-
import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib import font_manager, rc
from matplotlib.colors import LinearSegmentedColormap
import platform

# ─────────────────────────────────────────────
# 0) 기본 세팅 (한글 폰트 & 마이너스)
# ─────────────────────────────────────────────
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은고딕
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


# ─────────────────────────────────────────────
# 1) 경로
# ─────────────────────────────────────────────
SIG_SHAPE = "C:/Users/USER/Desktop/공모전/충남대/python/data/bnd_sigungu_00_2024_2Q.shp"
GRID_A    = "C:/Users/USER/Desktop/공모전/충남대/python/data/grid_다바_500M.shp"
GRID_B    = "C:/Users/USER/Desktop/공모전/충남대/python/data/grid_라바_500M.shp"
POP_A     = "C:/Users/USER/Desktop/공모전/충남대/python/data/2023년_인구_다바_500M.txt"
POP_B     = "C:/Users/USER/Desktop/공모전/충남대/python/data/2023년_인구_라바_500M.txt"

OUT_DIR   = "C:/Users/USER/Desktop/공모전/충남대/python/out"
FIG_DIR   = os.path.join(OUT_DIR, "fig")
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2) 시군구 경계 → 청주(4개 코드) → 단일 폴리곤
# ─────────────────────────────────────────────
sig = gpd.read_file(SIG_SHAPE)
codes = ["33041", "33042", "33043", "33044"]
cheongju_one = sig[sig["SIGUNGU_CD"].isin(codes)].dissolve()  # 단일 경계

# ─────────────────────────────────────────────
# 3) 500m 격자 두 장 → 합치고 → GRID_CD로 유일화(dissolve)
#    (여기서 유일화가 핵심! 이후 clip해도 중복 안 생김)
# ─────────────────────────────────────────────
gA = gpd.read_file(GRID_A)
gB = gpd.read_file(GRID_B)

# CRS 정렬
if gA.crs != cheongju_one.crs: gA = gA.to_crs(cheongju_one.crs)
if gB.crs != cheongju_one.crs: gB = gB.to_crs(cheongju_one.crs)

grid_all = gpd.GeoDataFrame(pd.concat([gA, gB], ignore_index=True),
                            geometry="geometry", crs=gA.crs)

# GRID 코드 컬럼명 표준화 (데이터에 따라 'GRID_500M_' 또는 'GRID_500M'일 수 있음)
if "GRID_500M_" in grid_all.columns:
    grid_all = grid_all.rename(columns={"GRID_500M_": "GRID_CD"})
elif "GRID_500M" in grid_all.columns:
    grid_all = grid_all.rename(columns={"GRID_500M": "GRID_CD"})
else:
    raise KeyError("격자 코드 컬럼을 찾지 못했습니다. ('GRID_500M_' 또는 'GRID_500M')")

# 키 정리
grid_all["GRID_CD"] = grid_all["GRID_CD"].astype(str).str.strip()
grid_all = grid_all.dropna(subset=["GRID_CD"])

# (중요) 동일 GRID_CD를 하나의 폴리곤으로 유일화
grid_all_unique = grid_all.dissolve(by="GRID_CD", as_index=False)

print("※ 격자 중복 제거 전:", grid_all.duplicated(subset="GRID_CD").sum())
print("※ 격자 중복 제거 후:", grid_all_unique.duplicated(subset="GRID_CD").sum())  # 0 기대

# ─────────────────────────────────────────────
# 4) 단일 경계로 clip (overlay X)
# ─────────────────────────────────────────────
grid_cj = gpd.clip(grid_all_unique, cheongju_one)
print("※ clip 후 grid_cj 중복 GRID_CD:",
      grid_cj.duplicated(subset="GRID_CD").sum())  # 0 기대

# ─────────────────────────────────────────────
# 5) 인구 텍스트 두 장 → 합치기 → 총인구(to_in_001)만 → GRID별 합계
# ─────────────────────────────────────────────
# 인코딩은 utf-8-sig / cp949 중 하나로 맞춰 읽히는 쪽 사용
try:
    popA_df = pd.read_csv(POP_A, sep="^", engine="python", header=None, encoding="utf-8-sig")
    popB_df = pd.read_csv(POP_B, sep="^", engine="python", header=None, encoding="utf-8-sig")
except UnicodeDecodeError:
    popA_df = pd.read_csv(POP_A, sep="^", engine="python", header=None, encoding="cp949")
    popB_df = pd.read_csv(POP_B, sep="^", engine="python", header=None, encoding="cp949")

pop_all = pd.concat([popA_df, popB_df], ignore_index=True)
pop_all.columns = ["year", "GRID_CD", "POP_TYPE", "POP"]

# 총인구 코드만
pop_all = pop_all[pop_all["POP_TYPE"] == "to_in_001"].copy()

# 숫자형 보장(천단위 콤마 방지)
pop_all["POP"] = (pop_all["POP"].astype(str).str.replace(",", "", regex=False))
pop_all["POP"] = pd.to_numeric(pop_all["POP"], errors="coerce")

# 키 정리 + GRID별 합계(안전빵)
pop_all["GRID_CD"] = pop_all["GRID_CD"].astype(str).str.strip()
pop_grp = pop_all.groupby("GRID_CD", as_index=False)["POP"].sum()

# ─────────────────────────────────────────────
# 6) 병합 + 합계 검증
# ─────────────────────────────────────────────
grid_pop = grid_cj.merge(pop_grp, on="GRID_CD", how="left")
grid_pop["POP"] = pd.to_numeric(grid_pop["POP"], errors="coerce").fillna(0)

total_pop = int(grid_pop["POP"].sum())
print(f"※ 최종 격자 총인구 합계: {total_pop:,} 명")

# sanity check: 중복 여부 최종 점검
print("※ 최종 grid_pop 중복 GRID_CD:",
      grid_pop.duplicated(subset="GRID_CD").sum())  # 0 기대

# ─────────────────────────────────────────────
# 7) (선택) 시각화: 회색→빨강, 로그스케일 권장
# ─────────────────────────────────────────────
# 컬러맵 (연회색→연빨강→진빨강)
gray_red = LinearSegmentedColormap.from_list(
    "gray_red", ["#E5E7EB", "#FCA5A5", "#DC2626"], N=256
)

g3857  = grid_pop.to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(9, 9))

# 데이터 먼저 그려 축 범위 확보
vals = g3857["POP"].replace([np.inf, -np.inf], np.nan).dropna()
if len(vals):
    vmin = max(vals.min(), 1)               # 로그용 0 회피
    vmax = np.quantile(vals, 0.99)          # 극값 완화
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=vmin, vmax=vmax)
else:
    norm = None

g3857.plot(
    ax=ax,
    column="POP",
    cmap=gray_red,
    norm=norm,
    alpha=0.75,
    linewidth=0,
    legend=True,
    missing_kwds={"color": "none", "edgecolor": "none"}
)

# 베이스맵
cx.add_basemap(ax, crs=g3857.crs, source=cx.providers.CartoDB.Positron, zoom=11)
ax.set_title("청주시 500m 격자별 인구 (저=회색, 고=빨강, 로그스케일)", fontsize=13)
ax.set_axis_off()
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, "cheongju_500m_population_fixed.png")
plt.savefig(fig_path, dpi=220, bbox_inches="tight")
plt.show()
print("※ 저장:", fig_path)


out_gpkg = os.path.join(OUT_DIR, "cheongju_500m_population_fixed.gpkg")
grid_pop.to_file(out_gpkg, driver="GPKG")
print("※ GPKG 저장:", out_gpkg)








