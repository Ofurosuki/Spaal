# Depth Map Architecture Alternatives

## 現状のアーキテクチャ

### Design
```python
depth_map: Dict[(azimuth_key, altitude_key)] -> (distance, intensity)
```

### 特徴
- **単一点保持**: 各グリッドセルに1点のみ
- **最近傍選択**: 同じセルに複数点 → 最も近い点のみ保持
- **固定解像度**: 全距離で同じグリッド解像度

### 問題点
1. 情報損失: 複数点が同じセルに投影される場合、N-1点が失われる
2. 遠距離で顕著: 0.1°でも100mでは17.5cmの間隔
3. チャンネル間隔不均一: 大きい間隔のチャンネルで点が集中 → 損失増加

---

## Alternative 1: Multi-Point Depth Map ⭐⭐⭐

### Design
```python
depth_map: Dict[(azimuth_key, altitude_key)] -> List[(distance, intensity)]
```

### 実装
```python
def _create_depth_map_multi(self, max_points_per_cell: int = 5):
    """Each grid cell stores up to K nearest points."""

    depth_map = {}

    for point_idx, point in enumerate(points):
        key = (azimuth_key, altitude_key)
        distance = np.linalg.norm(point)

        if key not in depth_map:
            depth_map[key] = []

        depth_map[key].append((distance, intensity, point_idx))

        # Keep only K nearest points
        depth_map[key].sort(key=lambda x: x[0])
        if len(depth_map[key]) > max_points_per_cell:
            depth_map[key] = depth_map[key][:max_points_per_cell]

    return depth_map
```

### 長所
- 情報損失が大幅に削減（K倍の点を保持）
- 実装が簡単（既存コードからの変更が少ない）
- メモリ使用量が予測可能（max K倍）

### 短所
- メモリ使用量増加: K倍（K=5なら5倍）
- データ構造の変更が必要
- 復元時にK個の点すべてを出力

### 適用シーン
- 遠距離の復元率向上
- チャンネル間隔が大きい箇所の改善

### 期待される改善
- 40-60m復元率: 34% → **60-70%**
- 60-100m復元率: 2% → **30-40%**

---

## Alternative 2: Adaptive Resolution Grid ⭐⭐⭐

### Design
```python
# 距離に応じて解像度を変える
depth_map: Dict[(azimuth_key, altitude_key, distance_band)] -> (distance, intensity)
```

### 実装
```python
def _get_adaptive_resolution(self, distance_m: float) -> tuple[float, float]:
    """Return (h_resolution, v_resolution) based on distance."""
    if distance_m < 20:
        return 0.1, None  # Current resolution
    elif distance_m < 50:
        return 0.05, None  # 2x finer
    else:
        return 0.025, None  # 4x finer
```

### 長所
- 遠距離の解像度が大幅に向上
- 物理的に均一な点密度を維持
- 近距離のメモリは増えない

### 短所
- 実装が複雑（距離バンドごとの管理）
- メモリ使用量が遠距離で増加
- グリッドサイズが可変

### 期待される改善
- 40-60m復元率: 34% → **80-90%**
- 60-100m復元率: 2% → **60-70%**

---

## Alternative 3: Range Image with Point Lists ⭐⭐

### Design
```python
# 2D配列、各セルにリスト
range_image: np.ndarray[channels, h_steps] of List[PointData]
```

### 実装
```python
def _create_range_image(self):
    """Create a 2D range image where each pixel contains a list of points."""

    # Initialize 2D array of lists
    range_image = np.empty((self.channels, self.h_steps), dtype=object)
    for i in range(self.channels):
        for j in range(self.h_steps):
            range_image[i, j] = []

    # Populate
    for point in points:
        ch_idx = self._find_channel(elevation)
        h_idx = int(azimuth / self.h_resolution)
        range_image[ch_idx, h_idx].append(PointData(distance, intensity))

    return range_image
```

### 長所
- 構造が直感的（LiDARの物理的な配置に対応）
- アクセスが高速（2Dインデックス）
- 可視化しやすい

### 短所
- メモリ効率が悪い（空のセルも配列を確保）
- 疎なデータに不適
- リストのオーバーヘッド

---

## Alternative 4: Octree / KD-Tree ⭐

### Design
```python
# 空間を階層的に分割
octree: Octree[Point3D]
```

### 長所
- 空間クエリが高速
- メモリ効率が良い（疎なデータに強い）
- 適応的な解像度

### 短所
- LiDARの構造（円筒座標）に不適合
- 実装が複雑
- 既存コードからの変更が大きい

---

## Alternative 5: Continuous Point Cloud (No Grid) ⭐⭐

### Design
```python
# グリッド化せず、元の点群を直接保持
# スキャン時に最も近い点を検索
points: np.ndarray[N, 3]
kdtree: KDTree(points)

def scan(azimuth, altitude):
    # Calculate ray direction
    direction = spherical_to_cartesian(azimuth, altitude)

    # Find points within cone
    candidates = kdtree.query_radius(origin, direction, cone_angle)

    # Return nearest
    return min(candidates, key=lambda p: distance(p))
```

### 長所
- 情報損失ゼロ（すべての点を保持）
- グリッド化の問題を回避
- 元のデータに忠実

### 短所
- スキャン時の計算コストが高い
- リアルタイム性能が悪い
- メモリ使用量が大きい

---

## Alternative 6: Probabilistic Occupancy Grid

### Design
```python
# 各セルに確率分布を保持
occupancy_grid: Dict[(az, alt)] -> GaussianDistribution(mean_dist, variance)
```

### 長所
- 不確実性を表現できる
- 複数点を統計的に統合

### 短所
- LiDARシミュレーションには過剰
- 計算コストが高い
- 実装が複雑

---

## 推奨される組み合わせ

### Phase 1: 即座に実装可能 ⭐⭐⭐
**Multi-Point Depth Map** (Alternative 1)

```python
depth_map: Dict[(az_key, alt_key)] -> List[(distance, intensity), ...]  # Up to K=5
```

**理由**:
- 実装が簡単（既存コードの小さな変更）
- 効果が高い（遠距離とチャンネル間隔の問題を両方解決）
- メモリ増加が許容範囲（5倍程度）

**期待される効果**:
- 全体復元率: 88% → **93-95%**
- 40-60m: 34% → **60-70%**

---

### Phase 2: 中長期的改善 ⭐⭐⭐
**Adaptive Resolution Grid** (Alternative 2)

```python
def get_resolution(distance):
    if distance < 20: return 0.1
    elif distance < 50: return 0.05
    else: return 0.025
```

**組み合わせ効果**:
- Multi-Point + Adaptive Resolution
- 全体復元率: 88% → **95-98%**
- 60-100m: 2% → **60-80%**

---

## 比較表

| アーキテクチャ | 実装難易度 | メモリ増加 | 復元率改善 | 推奨度 |
|--------------|----------|----------|----------|--------|
| **Multi-Point Depth Map** | 低 | 5x | +5-7% | ⭐⭐⭐ |
| **Adaptive Resolution** | 中 | 3-5x | +10-15% | ⭐⭐⭐ |
| **Range Image** | 低 | 10-20x | +3-5% | ⭐⭐ |
| **Octree** | 高 | 1-2x | +5-10% | ⭐ |
| **Continuous (No Grid)** | 高 | 変わらず | 0% (損失なし) | ⭐ |
| **Probabilistic** | 高 | 5-10x | +5-10% | ⭐ |

---

## 次のステップ

1. ✅ 問題の根本原因を特定完了（チャンネル間隔の不均一性）
2. ⏭️ Multi-Point Depth Map の実装（プロトタイプ）
3. ⏭️ 効果測定（復元率の改善を確認）
4. ⏭️ Adaptive Resolution の検討

---

## 結論

**最も効果的で実装しやすいアーキテクチャ変更**:

1. **Multi-Point Depth Map** (各セルに最大5点保持)
   - 簡単に実装可能
   - 大幅な改善が期待できる
   - メモリコストは許容範囲

2. その後、必要に応じて **Adaptive Resolution** を追加
   - さらなる遠距離の改善
   - 物理的に均一な点密度

この2つの組み合わせで、**95%以上の復元率**を達成できると期待されます。
