# 料盘边界误检候选与 2×4 网格恢复施工方案

## 1. 文档目的

本文给出 `plate_empty` / `plate` 普通粗定位模式中，料盘边界附近的相邻料盘误检候选进入当前料盘后，污染方向估计和 2×4 网格拟合的修复方案。

本方案针对 `00741`～`00745` 的稳定失败，同时保持以下既有能力：

- `PLATE_CELL` 或 `LUNZI` 外框存在但中心 `hole` 缺失时，真实目标仍有机会输出；
- `00038`、`00039` 的真实小框不能因尺寸略小被误删；
- ROI 覆盖三排时，继续使用现有 `rect + 上下边缘` 行对裁决；
- 证据不足时安全失败，不通过“取前 4 个”或“删最低置信度”猜测结果；
- 不修改公共 C API、应用调用参数、JSON 字段和坐标系。

本文是施工方案，不包含本次代码修改。

---

## 2. 复现基线

测试图片：

```text
00741.jpg
00742.jpg
00743.jpg
00744.jpg
00745.jpg
```

调用模式：

```json
{
  "xialiao_type": "plate_empty",
  "rect": [94, 91, 1197, 91, 1197, 667, 94, 667]
}
```

业务真值：

```text
8 个空位，0 个轮子
```

当前五张图结果一致：

```text
empty_count = 0
wheel_count = 0
unknown_count = 8
tray_complete = false
ret-msg = warning: grid fit failed:
          selected tray row must contain two to four candidates
```

模型并非漏检。结果图和断点均确认：目标料盘的 8 个 `PLATE_CELL` 和 8 个 `PLATE_HOLE` 已正确检出，另外还出现了 1 个来自左上相邻料盘的低质量 `PLATE_CELL`。

---

## 3. 00741 精确诊断数据

### 3.1 检测到的料盘区域

```text
tray_roi:
x      = 188.52
y      = 150.70
width  = 880.47
height = 459.62
```

### 3.2 正常 8 个候选

正常候选中心位于：

```text
上排 Y = 269.16 ～ 277.26
下排 Y = 480.03 ～ 490.27
```

正常外框尺寸约为：

```text
width  = 206 ～ 210 px
height = 205 ～ 211 px
confidence ≈ 0.53
has_center_hole = true
```

### 3.3 多余候选

```text
center     = (200.82, 152.88)
bbox       = (125.99, 78.68, 149.67, 148.40)
confidence = 0.31
has_center_hole = false
```

该候选的异常证据：

| 证据 | 数值 | 说明 |
|---|---:|---|
| 中心距料盘上边界 | 2.18 px | 只擦进当前料盘边界 |
| 中心距料盘左边界 | 12.30 px | 位于左上角边缘带 |
| 外框与料盘交叠率 | 29.97% | 主体大部分位于料盘外 |
| 宽度/正常中位数 | 约 72.3% | 明显偏小 |
| 高度/正常中位数 | 约 71.0% | 明显偏小 |
| 中心孔 | 缺失 | 缺少配对增强证据 |
| 置信度 | 0.31 | 明显低于正常候选约 0.53 |

这些证据单独使用都可能误伤真实目标，但组合后足以判断它不能作为当前料盘的网格锚点。

---

## 4. 当前失败链路

### 4.1 边界过滤把“中心在框内”当成无条件通过

当前 `shouldKeepTrayPair()` 的首要规则是：

```cpp
if (obj_center_in)
    return true;
```

多余候选中心的 `Y=152.88`，料盘上边界为 `Y=150.70`。它只进入约 2.18 px，但仍被直接保留，后续 30% 的交叠率、无孔、尺寸偏小都没有机会阻止它进入网格拟合。

涉及位置：

```text
det-src/det.cpp
isBoxInsideTray()
shouldKeepTrayPair()
filterObjectsOnlyByTray()
filterPairsByTray()
```

### 4.2 所有候选共同参与方向估计

`det_to_pts_plate_empty()` 当前使用全部候选中心执行：

```cpp
cv::RotatedRect rc = cv::minAreaRect(list_holes);
```

8 个正常点接近水平，但加入左上角第 9 点后，估计角度变成：

```text
rc.angle = 9.37°
```

第 9 点因此不仅增加候选数量，还直接带偏料盘局部坐标系。

### 4.3 错误方向导致 1+4+4 被合并成 5+4

当前行聚类阈值为：

```cpp
cell_size * 0.5f
```

并采用按排序后相邻点间距连接的单链聚类。方向带偏后，第 9 点与目标上排在局部 Y 方向被串成一行，最终只得到：

```text
grid rows = 2
row 0 = 5 candidates
row 1 = 4 candidates
```

### 4.4 rect 没有机会参与

`fitOrderedGrid2x4()` 仅在 `all_rows.size() > 2` 时执行 `rect` 行对评分。当前错误聚类已经只剩两行，因此直接进入：

```text
selected_pair = 0
source = only_pair
```

即使应用传入 rect，也无法处理“两行中的一行多了一个候选”。

### 4.5 安全门槛正确地阻止了错误输出

当前代码要求选中行每行只能有 2～4 个候选。发现 5 个候选后返回失败是合理的，不能删除这个安全门槛。

需要修复的是进入网格拟合的候选质量和方向估计，不是把 `>4` 改成允许，或在 5 个中随意截取 4 个。

---

## 5. 设计目标

1. `00741`～`00745` 使用现有 rect 稳定输出 8 空位、0 轮子。
2. 明显擦边的相邻料盘候选不得参与方向和初始网格估计。
3. 无 hole 的真实 `PLATE_CELL` / `LUNZI` 只要位于料盘内部且尺寸正常，仍可作为可靠候选。
4. 靠近边界但不能确定真假的候选不立即硬删，降级为补位候选。
5. 可靠候选先拟合网格；补位候选只能填补空槽，不能抢占可靠候选或改变网格。
6. 不用固定置信度阈值决定目标生死。
7. 不依赖候选容器顺序。
8. 不改变 `pts`、`pts_cell`、`pts_cell_rect`、`pts_lunzi` 的含义、排序和坐标系。
9. `top_capture == 1` 的单格精拍分支不纳入本次改造。

---

## 6. 非目标

- 不重新训练或替换 `xialiao.m`；
- 不修改 TensorRT 推理阈值；
- 不调整深度计算、手眼矩阵或抓取深度；
- 不把应用 rect 改成强制图像裁剪；
- 不改变三排场景现有的 `rect + edge_tiebreak` 规则；
- 不把“最低置信度候选”直接当成噪声删除；
- 不引入跨帧缓存或跨帧跟踪；
- 不修改公共导出函数 `AI_det_init`、`AI_det`、`AI_det_close`。

---

## 7. 总体方案

后处理调整为以下流程：

```text
原始检测和 object/hole 配对
        │
        ▼
计算候选边界证据、尺寸证据和孔证据
        │
        ├─ rejected：明确位于料盘外
        ├─ primary：可靠网格锚点
        └─ supplemental：可补位但不可建网格
        │
        ▼
只用 primary 候选估计方向和初始 2×4 网格
        │
        ├─ primary 足以建立网格：继续
        └─ primary 证据不足：受控回退或安全失败
        │
        ▼
primary 先一对一归位
        │
        ▼
supplemental 仅填补仍为 unknown 的格位
        │
        ▼
完整性校验、JSON 输出和最终绘图
```

核心原则：

> 候选是否有资格“建立网格”和它是否有资格“补一个缺失格位”是两件事。

本次第 9 个候选可以暂时保留为 `supplemental`，但不得参与 `minAreaRect()` 和初始行聚类。正常 8 个候选将建立正确网格，第 9 个候选因没有空槽可填而最终被拒绝。

---

## 8. 内部数据结构调整

### 8.1 候选角色

在 `det.cpp` 内部新增：

```cpp
enum class TrayCandidateRole {
    Primary,
    Supplemental,
    Rejected
};
```

语义：

| 角色 | 可以估计方向 | 可以拟合网格 | 可以填补空槽 | 可以抢占已匹配格位 |
|---|---|---|---|---|
| `Primary` | 是 | 是 | 是 | 不适用，优先匹配 |
| `Supplemental` | 否 | 否 | 是 | 否 |
| `Rejected` | 否 | 否 | 否 | 否 |

### 8.2 扩展 TrayCandidate

建议扩展为：

```cpp
struct TrayCandidate {
    TraySlotState state = TraySlotState::Unknown;
    int class_id = -1;
    float confidence = 0.0f;
    cv::Rect2f bbox;
    cv::Point2f center;
    int source_index = -1;
    bool has_center_hole = false;
    bool size_outlier = false;

    float tray_overlap_ratio = 0.0f;
    float tray_edge_distance = -1.0f;
    float width_ratio = 0.0f;
    float height_ratio = 0.0f;
    bool object_center_in_tray = false;
    bool object_center_in_safe_tray = false;
    bool hole_center_in_safe_tray = false;

    TrayCandidateRole role = TrayCandidateRole::Rejected;
    std::string reject_reason;
};
```

这些字段保持内部使用，不加入业务 JSON。

### 8.3 固定拒绝原因

至少定义：

```text
outside_tray
boundary_low_overlap
invalid_bbox
primary_grid_unavailable
supplemental_no_open_slot
supplemental_too_far
supplemental_conflict
duplicate_slot
```

拒绝原因应使用枚举或固定常量，不要在不同函数中拼接不同字符串。

---

## 9. 候选证据计算

### 9.1 参考尺寸使用中位数

当前 `avg_side` 使用所有外框宽度的平均值，容易被额外小框拉低。改造后区分：

```text
raw_average_side：仅保留兼容或调试，不用于核心阈值
reference_width：候选宽度中位数
reference_height：候选高度中位数
reference_side：reference_width 和 reference_height 的稳健组合
```

推荐：

```cpp
reference_side = 0.5f * (median_width + median_height);
```

`PLATE_CELL` 与 `LUNZI` 外框尺寸若长期存在系统性差异，应分别计算类别中位数；样本太少时再回退到所有主体框中位数。

尺寸参考必须使用两遍式计算，禁止先把 rect 内所有候选混入中位数：

```text
第一遍：只计算中心是否在 detected tray 内、bbox 与 tray 的交叠率
第二遍：仅使用“主体中心在 tray 内且交叠率 >= 0.60”的候选计算中位尺寸
第三遍：使用干净的中位尺寸计算全部候选的 width_ratio/height_ratio
```

该目标 tray 参考池仅在以下条件同时成立时启用：

```cpp
hasValidTraySelectionRect() &&
tray_source == "detected" &&
g_top_capture != 1
```

无 rect、fitted tray、top 和校准路径不启用目标 tray 参考池。参考池为空时安全失败，禁止回退到目标 tray 外的候选凑尺寸基准。

### 9.2 安全内区

在检测到的 `tray_roi` 内构造收缩后的 `safe_tray_roi`：

```cpp
edge_margin = clamp(reference_side * 0.08f, 8.0f, 24.0f);
safe_tray_roi = inset(tray_roi, edge_margin);
```

初始值 `0.08` 需要通过回归样本标定，不作为永久不可调硬编码。

对 00741：

```text
reference_side ≈ 208 px
edge_margin ≈ 16.6 px
```

第 9 个候选距上边缘仅 2.18 px，因此不在安全内区；正常候选距料盘边缘约 100 px 以上，不受影响。

### 9.3 外框交叠率

统一定义：

```cpp
overlap = area(tray_roi & candidate_bbox) / area(candidate_bbox);
```

初始阈值沿用：

```text
min_tray_overlap_ratio = 0.60
```

第 9 个候选仅约 0.30。

### 9.4 边缘距离

当 object 中心位于 tray 内时，计算中心到四条边的最小距离：

```cpp
edge_distance = min(
    center.x - tray.left,
    tray.right - center.x,
    center.y - tray.top,
    tray.bottom - center.y);
```

不得只计算上边界，因为相邻料盘也可能出现在左、右或下方。

### 9.5 尺寸比

```cpp
width_ratio  = bbox.width  / median_width;
height_ratio = bbox.height / median_height;
```

尺寸分两级：

```text
primary_size 范围：0.75 ～ 1.30
acceptable_size 范围：0.65 ～ 1.40
```

- `primary_size` 用于判断是否能作为建网格锚点；
- `acceptable_size` 用于判断是否仍可作为补位候选；
- 不能仅因不满足 `primary_size` 就删除候选。

00741 第 9 个候选尺寸比约 0.71～0.72，不满足 `primary_size`，但接近现有宽松范围，因此降级比硬删更稳妥。

### 9.6 hole 证据

hole 是增强证据，不是主体存在的必要条件：

- hole 有效且位于安全内区：增强候选可信度；
- hole 缺失：允许依靠主体外框、位置和尺寸成为 `Primary`；
- 单独 hole 没有主体 object：不得创建候选；
- hole 略微越界：坐标回退到主体中心，保持现有兼容逻辑。

### 9.7 置信度只用于同级排序

本次不设置诸如 `confidence < 0.4` 就删除的规则。

置信度仅在两个同角色候选竞争同一空槽、且几何代价接近时作为次级排序依据。位置和网格一致性优先于置信度。

---

## 10. 候选角色判定

### 10.1 明确拒绝 Rejected

满足以下任一条件可直接拒绝：

1. bbox 非法或面积为 0；
2. object 中心不在 tray 内；
3. object 交叠率低于 0.60；
4. 同时没有安全有效的配对 hole；
5. 且不满足现有“主体有效交叠 + hole 在盘内”的边缘保留规则。

这里的 2～4 应作为组合条件，不是任意一项单独成立就删除。

### 10.2 可靠候选 Primary

推荐初始规则：

```text
object 中心位于 safe_tray_roi
AND 尺寸满足 primary_size
AND 主体与 tray 有有效交叠
```

满足以上条件即可成为 `Primary`，不强制要求 hole。

因此以下生产场景仍然兼容：

```text
检测到 LUNZI 外框
没有 HOLE
轮子位于料盘正常格位
外框尺寸正常
```

它仍会作为 `Primary` 参与网格和输出，只是中心回退到外框中心。

### 10.3 补位候选 Supplemental

以下候选不用于建立网格，但暂不删除：

- 中心进入 tray，但位于边缘带；
- 中心在 tray 内，但外框交叠率偏低；
- 尺寸超出 `primary_size`，但仍在 `acceptable_size`；
- hole 缺失且还存在其他弱证据；
- 由 fitted tray 得到、边界证据本身不够可靠的候选。

00741 第 9 个候选应被判为：

```text
role = Supplemental
reason/evidence = boundary_low_overlap
```

而目标 8 个候选应全部为 `Primary`。

### 10.4 tray_source=fitted 的特殊规则

当没有 `PLATE_TRAY`，`tray_roi` 是由候选自身拟合得到时，不能再把该边界当成强证据循环证明候选有效。

此时：

- 不根据 fitted tray 的轻微越界直接硬拒绝；
- 主要依赖尺寸、hole、rect 和网格几何决定 Primary/Supplemental；
- 若无法从 Primary 建立两行网格，保持安全失败；
- 不允许为了凑 8 个点把所有 Supplemental 无条件提升。

---

## 11. Primary 网格拟合

### 11.1 拟合输入

构造两个候选集合：

```cpp
primary_candidates
supplemental_candidates
```

只有 `primary_candidates` 的中心进入：

```text
minAreaRect / 方向估计
行列聚类
相邻行对评分
2×4 有序网格拟合
```

### 11.2 方向估计

将当前：

```cpp
cv::minAreaRect(all_candidate_centers)
```

改为：

```cpp
cv::minAreaRect(primary_candidate_centers)
```

并使用 Primary 的中位尺寸设置 `TrayFitting::cell_size`。

00741 中 Primary 为正常 8 点，估计方向会回到接近水平，不再被左上角候选拉到 9.37°。

### 11.3 Primary 最低建网格条件

沿用现有安全条件：

- 总 Primary 数量不少于 4；
- 至少能形成两行；
- 每个选中行至少有 2 个 Primary；
- 每个选中行最多有 4 个 Primary；
- 三行以上继续使用现有 rect / edge_tiebreak；
- 无法唯一确定目标行对时安全失败。

不要因为总 Primary 少于 8 就直接失败。现有网格能够在每行 2～4 个点时补齐理论格位，随后再由 Supplemental 尝试填补实际状态。

### 11.4 Primary 证据不足时的受控回退

推荐只允许一次受控回退：

```text
若 Primary 无法建立两行网格：
    从 Supplemental 中选择满足 acceptable_size、
    且不属于低交叠边缘带的候选，临时升级为 FallbackPrimary；
    重新拟合一次；
若仍失败：
    安全失败。
```

禁止循环逐个加入候选直到“碰巧成功”，否则结果依赖候选顺序。

`boundary_low_overlap` 类型不得在该回退中升级；它只能在已有网格后补位。

---

## 12. 两阶段格位匹配

### 12.1 第一阶段：Primary 归位

对 Primary 与 8 个理论格位建立距离边，沿用当前一对一匹配原则：

```text
一个候选最多匹配一个格位
一个格位最多接收一个候选
超过距离阈值不得匹配
```

Primary 匹配完成后锁定已占用格位。

### 12.2 第二阶段：Supplemental 补位

Supplemental 仅针对 `Unknown` 格位建立匹配边：

- 不得抢占 Primary 已匹配格位；
- 不得重新估计方向；
- 不得改变 ordered_grid；
- 不得改变已确定格位状态；
- 距离超阈值时拒绝，原因 `supplemental_too_far`；
- 所有格位已占用时拒绝，原因 `supplemental_no_open_slot`。

00741 中正常 8 个 Primary 已占满网格，第 9 个 Supplemental 最终应被拒绝，不影响成功输出。

### 12.3 Supplemental 冲突

若两个 Supplemental 同时竞争一个 Unknown 格位：

1. 几何距离明显更优者胜出；
2. 距离接近时比较是否有中心孔；
3. 再比较尺寸一致性；
4. 最后才比较置信度；
5. 仍无法明显区分时保持 Unknown，返回不完整，不猜测。

建议使用明确领先差，例如：

```text
best_cost + conflict_margin < second_cost
```

阈值需通过异常样本标定。

---

## 13. rect 的职责边界

本次保持 rect 的既有主要语义：

```text
rect 用于多行候选时选择业务目标行对
```

不把 rect 直接改成图像硬裁剪，原因包括：

- 应用 rect 可能刻意覆盖完整托盘外框；
- 部分真实检测框可能略微越过 rect；
- 现有 edge_tiebreak 依赖 rect 外附近的真实边缘信息；
- 00741 的误检中心本身也位于当前 rect 内，单纯硬裁剪不能解决。

可以记录 `candidate_center_in_rect` 作为诊断或弱证据，但不能替代 tray 边界与网格几何。

---

## 14. 函数拆分建议

### 14.1 det.cpp 新增内部函数

建议新增：

```cpp
static cv::Rect2f insetRect(
    const cv::Rect2f& rect,
    float margin);

static float distanceToTrayEdge(
    const cv::Rect2f& tray,
    const cv::Point2f& center);

static void computeCandidateEvidence(
    std::vector<TrayCandidate>& candidates,
    const cv::Rect2f& tray,
    const std::string& tray_source,
    const TrayPostprocessConfig& cfg);

static void classifyCandidateRoles(
    std::vector<TrayCandidate>& candidates,
    const TrayPostprocessConfig& cfg);

static bool buildPrimaryGrid(
    const std::vector<TrayCandidate>& candidates,
    const cv::Rect2f& tray_roi,
    const std::vector<cv::Point2f>* selection_polygon,
    const cv::Mat* edge_image,
    std::array<cv::Point2f, 8>& ordered_grid,
    RowPairSelectionResult& selection_result,
    std::string& failure_reason);

static TrayPostprocessResult assignTraySlotsTwoPass(
    const std::vector<TrayCandidate>& candidates,
    const std::array<cv::Point2f, 8>& grid,
    float reference_side,
    const std::string& tray_source);
```

### 14.2 现有函数调整

| 文件 | 函数 | 调整 |
|---|---|---|
| `det-src/det.cpp` | `isBoxInsideTray()` | 不再让“中心刚进入边界”自动成为建网格强证据 |
| `det-src/det.cpp` | `shouldKeepTrayPair()` | 只删除明确外部目标；边界不确定目标保留并降级 |
| `det-src/det.cpp` | `filterPairsByTray()` | 保持 object/hole 索引同步，并保留拒绝原因 |
| `det-src/det.cpp` | `markSizeOutliers()` | 输出尺寸比，不只保存 bool |
| `det-src/det.cpp` | `buildTrayCandidates()` | 填充边界、hole、尺寸和角色字段 |
| `det-src/det.cpp` | `assignTraySlots()` | 改为 Primary 优先、Supplemental 补位的两阶段匹配 |
| `det-src/det.cpp` | `det_to_pts_plate_empty()` | 只用 Primary 估计方向和拟合网格 |
| `det-src/GridFitter.hpp` | `fitOrderedGrid2x4()` | 保留每行 2～4 的安全门槛；本次不允许静默截断 |

### 14.3 修改范围

主要修改：

```text
det-src/det.cpp
```

只有在需要补充内部诊断字段时才小幅调整：

```text
det-src/GridFitter.hpp
```

不修改：

```text
det-src/det.h
模型文件
公共 API
业务 JSON 协议
```

---

## 15. 配置参数

建议在 `TrayPostprocessConfig` 中集中定义：

```cpp
struct TrayPostprocessConfig {
    float min_tray_overlap_ratio = 0.60f;
    float safe_edge_margin_ratio = 0.08f;
    float safe_edge_margin_min_px = 8.0f;
    float safe_edge_margin_max_px = 24.0f;

    float primary_min_width_ratio = 0.75f;
    float primary_max_width_ratio = 1.30f;
    float primary_min_height_ratio = 0.75f;
    float primary_max_height_ratio = 1.30f;

    float acceptable_min_width_ratio = 0.65f;
    float acceptable_max_width_ratio = 1.40f;
    float acceptable_min_height_ratio = 0.65f;
    float acceptable_max_height_ratio = 1.40f;

    float grid_match_ratio = 0.40f;
    float supplemental_conflict_margin = 0.15f;
};
```

初始参数来自当前实现和 00741 诊断，只是施工起点。必须用回归集验证后再固化，不允许为了单张图片无限调参。

---

## 16. 主流程伪代码

```cpp
int det_to_pts_plate_empty(...)
{
    if (g_top_capture == 1)
        return det_to_pts_plate_top(...);

    // 1. 配对后的主体候选，保留 hole 缺失回退能力。
    auto candidates = buildTrayCandidates(...);

    // 2. 解析 detected/fitted tray。
    resolveTrayRoi(..., tray_roi, tray_source);

    // 3. 只剔除明确位于 tray 外的候选。
    filterClearlyOutsideCandidates(...);

    // 4. 使用中位数计算参考尺寸和安全内区。
    computeCandidateEvidence(candidates, tray_roi, tray_source, cfg);

    // 5. 分成 Primary / Supplemental / Rejected。
    classifyCandidateRoles(candidates, cfg);

    // 6. 只用 Primary 估计方向、聚类并拟合 2x4 网格。
    if (!buildPrimaryGrid(..., ordered_grid, ...)) {
        fillIncompletePlateEmptyJson(...);
        return 0;
    }

    // 7. Primary 先归位，Supplemental 只补 Unknown。
    auto result = assignTraySlotsTwoPass(
        candidates, ordered_grid, reference_side, tray_source);

    // 8. 根据最终匹配结果绘图和输出。
    fillPlateEmptyJson(...);
    drawFinalCandidates(...);
    return 0;
}
```

---

## 17. 调试日志与绘图

### 17.1 REL_DBG 日志

每个候选至少输出：

```text
candidate idx=8
class=plate_cell
center=(200.82,152.88)
overlap=0.300
edge_distance=2.18
size_ratio=(0.723,0.710)
has_hole=0
confidence=0.311
role=supplemental
reason=boundary_low_overlap
```

网格阶段输出：

```text
primary_count=8
supplemental_count=1
rejected_count=0
grid_input_count=8
grid_angle=...
grid rows=2 sizes=[4,4]
supplemental idx=8 result=no_open_slot
```

### 17.2 调试绘图

建议颜色：

| 类型 | 颜色 |
|---|---|
| Primary | 绿色 |
| Supplemental | 黄色 |
| Rejected | 灰色或红色 |
| 最终格位中心 | 蓝色 |
| safe_tray_roi | 青色细框 |
| tray_roi | 紫色细框 |

候选标签应包含角色和拒绝原因，避免再次出现“图上看见9个框，但不知道哪一个参与了JSON”的问题。

Debug 图必须实际保存或返回，不能只画在未落盘的局部 `img_dbg` 中。

---

## 18. JSON 与应用兼容性

本次不增加应用必须解析的新字段，也不修改已有字段：

```text
ret-code
ret-msg
pts
pts_cell
pts_cell_rect
pts_lunzi
slots
tray_complete
tray_source
empty_count
wheel_count
unknown_count
```

成功时仍按格位 index 升序输出：

```text
0 1 2 3
4 5 6 7
```

`pts_cell_rect` 与对应 `pts_cell` 保持相同世界坐标系、相同 Z 和一一对应关系。

应用端不需要修改调用方式或返回解析。

---

## 19. 实施步骤

### Step 1：增加证据字段和诊断，不改变行为

1. 扩展内部 `TrayCandidate`；
2. 计算交叠率、边缘距离、尺寸比和 hole 状态；
3. 增加受控日志；
4. 对 00741 确认第 9 个候选的证据与本文基线一致；
5. 该提交不改变现有输出。

### Step 2：候选角色分类

1. 增加 `Primary/Supplemental/Rejected`；
2. detected tray 使用安全内区；
3. fitted tray 使用保守规则；
4. hole 缺失但深处、尺寸正常的主体仍为 Primary；
5. 00741 应得到 `8 Primary + 1 Supplemental`。

### Step 3：Primary-only 网格拟合

1. 方向估计只使用 Primary；
2. `cell_size` 改用 Primary 中位尺寸；
3. 行聚类和 rect/edge 行对选择只使用 Primary；
4. 保留每行 2～4 和证据不足失败门槛；
5. 00741 网格应恢复为 `4 + 4`。

### Step 4：两阶段归位

1. Primary 先匹配并锁定格位；
2. Supplemental 仅匹配 Unknown；
3. 增加冲突领先差；
4. 未匹配 Supplemental 写入拒绝原因；
5. 00741 第 9 个候选应得到 `supplemental_no_open_slot`。

### Step 5：最终绘图和完整回归

1. 结果图区分三个角色；
2. 图中只把最终匹配框视为输出目标；
3. Debug、Release 均构建；
4. 执行基准、历史、异常和稳定性测试；
5. 确认公共导出和 JSON 协议无变化。

每一步单独提交，便于定位回归和回滚。

---

## 20. 回归测试矩阵

### 20.1 当前最小复现

| 图片 | rect | 期望 |
|---|---|---|
| 00741 | `[94,91,...,94,667]` | 8 空、0 轮、complete=true |
| 00742 | 同上 | 8 空、0 轮、complete=true |
| 00743 | 同上 | 8 空、0 轮、complete=true |
| 00744 | 同上 | 8 空、0 轮、complete=true |
| 00745 | 同上 | 8 空、0 轮、complete=true |

同时验证：

```text
Primary = 8
Supplemental = 1
网格输入 = 8
第 9 个候选未输出
```

### 20.2 plate_empty 历史基准

| 图片 | 期望 |
|---|---|
| 00038 | 6 空 + 2 轮 |
| 00039 | 6 空 + 2 轮 |
| 00044 | 8 空 + 0 轮 |

重点确认 00038/00039 的真实小框不会因 `primary_size` 降级后丢失。即使被降级为 Supplemental，也应正确填补对应格位。

### 20.3 三排选行基准

| 图片 | 原 rect | 期望 |
|---|---|---|
| 00156 | 是 | 8 个目标格位 |
| 00157 | 是 | 8 个目标格位，edge_tiebreak 稳定 |
| 00212 | 是 | 8 个目标格位，不能选上方轮子行 |

该组用于确认 Primary-only 不破坏现有 rect + 边缘二级裁决。

### 20.4 hole 缺失兼容

至少构造并验证：

1. `LUNZI` 有外框、无 `HOLE`，位于正常格位；
2. `PLATE_CELL` 有外框、无 `PLATE_HOLE`，位于正常格位；
3. hole 略微越界，主体正常；
4. 单独 hole、无主体外框；
5. 边界弱候选无 hole，但恰好对应一个真实 Unknown 格位。

期望：

- 场景 1、2 可以输出，中心回退到主体外框中心；
- 场景 3 保留主体；
- 场景 4 不创建候选；
- 场景 5 只有在已有网格且几何匹配唯一时才补位。

### 20.5 异常场景

1. 目标上边、下边、左边、右边分别出现相邻料盘候选；
2. 同一边出现 1～4 个弱候选；
3. 弱候选带有误检 hole；
4. 正常候选尺寸略小；
5. 料盘部分超出图像；
6. 无 `PLATE_TRAY`，使用 fitted tray；
7. Primary 只能形成一行；
8. 两个 Supplemental 竞争同一格位；
9. Primary 行仍超过 4 个；
10. rect 缺失、错误或只覆盖部分目标。

任何无法唯一确定 2×4 网格的场景都应保持 `tray_complete=false`，不得凑数成功。

### 20.6 透明料盘分类基准

`00132`～`00136` 的真值为 8 空、0 轮。该组此前存在“6 空 + 2 轮”的类别误判风险。

本次边界候选改造不得让结果进一步退化，但该语义误判属于类别裁决问题，应单独记录，不能用“网格完整”冒充识别正确。

### 20.7 稳定性

对 00741、00745、00157、00212 各连续调用至少 20 次：

```text
角色数量一致
网格角度一致
selected_pair 一致
8 格状态一致
无跨帧候选或图像残留
```

### 20.8 尺寸参考池污染基准

| 图片 | rect | 期望 |
|---|---|---|
| 01535～01539 | `[79,111,1214,111,1214,715,79,715]` | 8 空、0 轮、complete=true |

该组画面同时包含：

```text
目标 tray 的 8 个正常空位
上方相邻 tray 的 4 个轮子
左侧和下方远处 tray 的 8 个小尺寸空位框
```

验收日志应满足：

```text
reference_pool_count = 8
excluded_from_reference_count = 12
reference_width/reference_height 约为 182 px
```

不得通过缩小业务 rect 使测试通过。

---

## 21. 验收标准

### 21.1 功能

- 00741～00745 全部输出 8 空、0 轮；
- 第 9 个候选不参与方向和网格；
- 00038/00039/00044 保持历史正确结果；
- 00156/00157/00212 保持正确选行；
- 无 hole 的正常主体仍可参与并输出；
- 无法唯一裁决时仍安全失败。

### 21.2 协议

- `AI_det()` 参数不变；
- 公共导出符号不变；
- JSON 字段不删除、不改名；
- `pts_cell` 与 `pts_cell_rect` 一一对应；
- 应用端无需修改。

### 21.3 工程质量

- Debug 和 Release 编译通过；
- `git diff --check` 通过；
- 无新的编译警告；
- 候选角色和拒绝原因可从日志定位；
- Debug 图实际落盘；
- 无静态跨请求图像或候选缓存；
- 关键角色分类和两阶段匹配至少增加可自动执行的单元测试或离线回归脚本。

---

## 22. 风险与控制

### 风险 1：真实边缘格位被降级

降级不等于删除。Supplemental 可以在已有网格后填补 Unknown，因此真实边缘格位仍可输出。

### 风险 2：hole 缺失目标被误删

Primary 不强制要求 hole。深处、尺寸正常的主体外框可直接成为 Primary。

### 风险 3：Primary 太少导致成功率下降

保留一次确定性的 FallbackPrimary 回退，但禁止把 `boundary_low_overlap` 无条件升级。

### 风险 4：fitted tray 自证循环

fitted tray 不作为强硬边界删除依据，主要依赖网格、尺寸和 rect。

### 风险 5：Supplemental 抢占正确结果

使用两阶段匹配，Primary 格位先锁定，Supplemental 只能填 Unknown。

### 风险 6：参数过度针对 00741

参数必须通过历史基准、四方向边界、hole 缺失和透明料盘样本共同标定，不能只看当前五张图。

### 风险 7：完整但分类错误被误认为通过

验收同时检查 `empty_count/wheel_count` 真值，不只检查 `tray_complete`。

---

## 23. 不建议方案

### 23.1 直接允许一行 5 个候选

会把相邻料盘候选带入格位，不满足 2×4 物理约束。

### 23.2 发现 5 个就删除最低置信度

置信度受模型、材质和光照影响；真实目标可能恰好最低。

### 23.3 把置信度阈值提高到 0.4

虽然能删除当前 0.31 候选，但属于样本特定阈值，换模型或现场后风险高。

### 23.4 只提高尺寸下限

会重新引入 00038/00039 一类真实小框被删除的问题。

### 23.5 只缩小行聚类距离

第 9 点已经先污染方向估计，单调聚类阈值无法解决根因，并可能把倾斜真实行拆开。

### 23.6 把 rect 改成硬裁剪

当前第 9 个候选中心本身就在 rect 内，硬裁剪不能解决；还可能破坏边缘二级裁决。

### 23.7 成功网格后仍让所有候选重新拟合

会使 Supplemental 再次污染方向，违背分层设计。

---

## 24. 回滚策略

按实施步骤分提交后，可以分层回滚：

1. 关闭 Primary-only 网格开关，恢复所有候选拟合；
2. 保留证据日志和角色绘图，便于继续采样；
3. 关闭 Supplemental 补位，恢复安全失败；
4. 不回滚公共协议，因为本方案不修改公共协议；
5. 应用端无需同步回滚。

如需要短期灰度，可增加内部配置开关：

```text
enable_candidate_role_grid = true/false
```

该开关仅用于上线验证和回滚，不作为长期双逻辑维护。

---

## 25. 最终施工结论

本问题应按“候选角色分层”修复，而不是继续堆叠单一阈值：

```text
Primary 负责建立可信 2×4 网格
Supplemental 负责在网格已知时补缺
Rejected 负责隔离明确外部噪声
```

对 00741～00745：

```text
8 个正常候选 → Primary → 建立 4×2 网格
1 个左上擦边候选 → Supplemental → 无空槽可填 → 拒绝
最终输出 → 8 空、0 轮
```

这条路径同时保留了“有外框、无 hole”的生产兼容性，并继续遵守证据不足时安全失败的原则。
