# `plate_empty` 料盘后处理改造方案

## 1. 背景与问题复现

`plate_empty` 模式用于识别固定的 2 行 × 4 列料盘，并输出：

- `pts_cell`：空位；
- `pts_lunzi`：已放置轮子的位置；
- `pts`：建议抓取位置。

对 `build/plate/00038.jpg` 和 `00039.jpg` 的实际测试结果应为 **6 个空位 + 2 个轮子**，当前却输出 **5 个空位 + 2 个轮子**。

断点确认原始模型结果为：

```text
PLATE_TRAY = 1
PLATE_CELL = 6
PLATE_HOLE = 6
LUNZI      = 2
HOLE       = 2
```

模型已经正确检测出 6 个空位。丢失发生在 `det_to_pts_plate_empty()` 内调用 `filter_hole_by_plate_bbox()` 之后。

被删除的左下空位框面积约为 `56082 px²`，最大空位框面积约为 `70674 px²`：

```text
56082 / 70674 = 79.35%
```

当前规则要求目标面积不小于最大框面积的 80%，因此真实空位仅差约 0.65% 被误删：

```cpp
float min_area = max_lunzi.bbox.area() * 0.8f;
if (d.bbox.area() < min_area) {
    // 删除目标及其配对孔
}
```

此外，结果框在 `handle_ret_xialiao()` 中先于后处理绘制，所以结果图显示 6 个原始空位框，而 JSON 使用过滤后的数组，只输出 5 个。

### 1.1 检测类别、内部用途与 JSON 语义

本文统一使用以下术语，避免将外框类别、中心孔类别和 JSON 字段混为一谈：

| 检测类/字段 | 当前用途 | 改造后用途 |
|---|---|---|
| `PLATE_CELL` | 空位外框；参与过滤和网格拟合 | `Empty` 状态的主体候选，不直接作为 `pts_cell` 坐标来源 |
| `PLATE_HOLE` | 与 `PLATE_CELL` 配对；当前中心坐标输出到 `pts_cell` | 空位孔中心；作为 `pts_cell` 坐标来源和候选可信度依据 |
| `LUNZI` | 轮子外框；当前中心坐标输出到 `pts_lunzi` | `Wheel` 状态的主体候选和 `pts_lunzi` 坐标来源 |
| `HOLE` | 与 `LUNZI` 配对，用于计算中心 | 轮子中心孔；用于中心修正和候选可信度依据 |
| `pts_cell` | `PLATE_HOLE` 中心转换得到的世界坐标 | 最终归位为 `Empty` 的格位坐标；优先使用其 `PLATE_HOLE` 中心 |
| `pts_lunzi` | `LUNZI` 中心转换得到的世界坐标 | 最终归位为 `Wheel` 的格位坐标 |
| `pts` | `TrayFitting::list_pts` 中的 2 个建议抓取点 | 本次改造保持 2 个建议抓取点的既有语义 |

“空位候选”通常表示一对 `PLATE_CELL + PLATE_HOLE`。主体外框是候选存在的必要条件，中心孔用于提供输出中心和提升可信度；孔轻微越界或被判为无效时，可以回退到主体外框中心，但单独的 `PLATE_HOLE` 不能成为空位候选。

## 2. 改造目标

1. `00038`、`00039` 稳定输出 6 个空位和 2 个轮子。
2. `00044` 稳定输出 8 个空位和 0 个轮子。
3. 料盘外检测不得进入最终结果。
4. 最终每个目标只能归属一个网格位置，每个网格位置最多一个状态。
5. 位置不完整时必须通过 `tray_complete=false`、计数及明确消息暴露；Phase 2 再切换为非零错误码。
6. 结果图应能区分原始检测、最终保留目标和被拒绝目标。

## 3. 总体策略

将当前“料盘框过滤 + 最大面积硬过滤”调整为四层后处理：

```text
原始检测
  → 中心点/交叠率料盘区域过滤
  → 基于中位数的宽松尺寸异常检查
  → 固定 2×4 网格拟合与一对一归位
  → 8 格完整性校验及结构化输出
```

核心原则：**网格位置是主判据，框尺寸是辅助判据**。尺寸略小但位置正确的目标不应被删除。

## 4. 修改范围

主要修改以下文件：

| 文件 | 修改内容 |
|---|---|
| `det-src/det.cpp` | 替换料盘候选过滤、增加网格归位、状态裁决、完整性校验和调试绘图 |
| `det-src/GridFitter.hpp` | 清理并明确 2×4 标准网格输出，保证稳定返回 8 个有序点 |
| `det-src/det.h` | 如需对外暴露新增状态字段，再补充接口说明 |

不修改模型、TensorRT 推理和坐标转换逻辑。

### 4.1 本次范围边界

- `g_top_capture == 1` 时，`det_to_pts_plate_empty()` 会转入 `det_to_pts_plate_top()`；该分支检测单个中心目标，语义与 2×4 料盘不同，**不纳入本次改造**。
- 新的 2×4 后处理仅在普通 `plate_empty` 分支启用。
- 本次保持 `pts` 为 2 个建议抓取点，不把它改成 8 个格位点。
- per-slot 深度采样和基于深度的类别冲突裁决放到 Phase 2，不进入首版 MVP。

## 5. 数据结构设计

建议在 `det.cpp` 内部增加以下类型，先保持为内部实现，不扩大公共接口：

```cpp
enum class TraySlotState {
    Unknown,
    Empty,
    Wheel
};

struct TrayCandidate {
    int class_id = -1;
    float confidence = 0.0f;
    cv::Rect2f bbox;
    cv::Point2f center;
    int source_index = -1;
    bool has_center_hole = false;
    bool size_outlier = false;
    std::string reject_reason;
};

struct TraySlot {
    int index = -1;                 // 固定为 0..7
    cv::Point2f grid_center;
    TraySlotState state = TraySlotState::Unknown;
    int candidate_index = -1;
    float match_distance = FLT_MAX;
};

struct TrayPostprocessResult {
    std::array<TraySlot, 8> slots;
    std::vector<TrayCandidate> rejected;
    int empty_count = 0;
    int wheel_count = 0;
    int unknown_count = 8;
};
```

`reject_reason` 使用固定枚举值或常量字符串，至少包括：`outside_tray`、`unpaired_hole`、`grid_distance_too_large`、`duplicate_slot`、`size_outlier_unmatched` 和 `class_conflict`。

网格编号建议固定为按行排序：

```text
0  1  2  3
4  5  6  7
```

这样 JSON、机器人逻辑和调试图始终使用同一位置语义。

## 6. 第一阶段：替换料盘区域过滤

### 6.1 当前问题

`filter_hole_by_plate_bbox()` 同时承担：

1. 料盘范围过滤；
2. 目标与中心孔同步删除；
3. 基于最大框面积的尺寸过滤。

职责混合导致难以判断目标为什么被删除，而且函数中相同的面积过滤循环重复了两次。

### 6.2 新函数

将其拆成候选收集和区域判断：

```cpp
static bool isBoxInsideTray(
    const cv::Rect2f& tray,
    const cv::Rect2f& candidate,
    float min_overlap_ratio = 0.60f)
{
    if (candidate.area() <= 0.0f)
        return false;

    cv::Point2f center(
        candidate.x + candidate.width * 0.5f,
        candidate.y + candidate.height * 0.5f);

    if (tray.contains(center))
        return true;

    cv::Rect2f intersection = tray & candidate;
    float overlap = intersection.area() / candidate.area();
    return overlap >= min_overlap_ratio;
}
```

规则为：

- 中心点在料盘内时直接保留；
- 中心点略微越界时，交叠比例不低于 60% 仍可保留；
- 不再在这个阶段做面积删除。

`min_overlap_ratio` 不应散落为硬编码值。建议先在内部配置结构中定义，初始值使用 `0.60f`，用 `00038/00039/00044` 标定后再固化：

```cpp
struct TrayPostprocessConfig {
    float min_tray_overlap_ratio = 0.60f;
    float min_width_ratio = 0.65f;
    float max_width_ratio = 1.40f;
    float min_height_ratio = 0.65f;
    float max_height_ratio = 1.40f;
    float grid_match_ratio = 0.40f;
};
```

### 6.3 配对一致性

`LUNZI/HOLE` 和 `PLATE_CELL/PLATE_HOLE` 在 `align_lunzi_hole()` 后仍保持索引一一对应。过滤时必须同步处理一对对象，不能分别过滤两个数组。

区域有效性以**主体 object 中心**为主，以 object/hole 与料盘的交叠为辅助，具体规则固定为：

1. object 中心在 tray 内：保留，不因外框轻微越界而删除；
2. object 中心不在 tray 内，但 object 与 tray 交叠率不低于配置阈值，且 hole 中心在 tray 内：保留；
3. object 中心在 tray 内而 hole 略微越界：保留 object，但标记 `has_center_hole=false`，最终坐标回退到 object 中心；
4. object 和 hole 均明显位于 tray 外：拒绝，原因 `outside_tray`；
5. hole 在 tray 内但 object 与 tray 无有效交叠：拒绝，防止孤立小孔把料盘外目标带入结果。

也就是说，不能继续仅用 hole 是否位于 tray 内决定整对目标的去留。

建议增加：

```cpp
static bool filterPairsByTray(
    const Detection_pan& tray,
    std::vector<Detection_pan>& objects,
    std::vector<Detection_pan>& holes,
    std::vector<std::string>* reject_reasons = nullptr);
```

在入口处校验：

```cpp
if (objects.size() != holes.size()) {
    // 返回明确错误或先重新执行配对，禁止按越界索引继续处理
}
```

### 6.4 未检测到 `PLATE_TRAY` 时的 fallback

首版不得因为缺少 `PLATE_TRAY` 就跳过所有区域约束。固定采用以下行为：

1. 若候选主体总数不少于 4，使用所有已配对 `PLATE_CELL/LUNZI` 中心计算 `cv::minAreaRect`；
2. 将拟合矩形按估计单元格尺寸向外扩张半格，作为临时 tray ROI；
3. 标记 `tray_source="fitted"`，后续仍执行尺寸检查和 2×4 网格完整性校验；
4. 候选不足 4 个或无法稳定分成两行时，返回 `tray_not_found`，不得在无区域约束的情况下输出成功；
5. 检测到模型 `PLATE_TRAY` 时标记 `tray_source="detected"`。

fallback 只用于维持既有无 tray 场景能力，其结果可信度低于模型直接检测到 tray 的情况，并需要在 JSON 中显式体现。

### 6.5 `holes.size()==0` 时的行为

`align_lunzi_hole()` 在 `xialiao` 模型下不会为缺失孔生成占位 hole；`feilun` 模型才会补占位。因此 `plate_empty` 可能出现 `objects` 非空而 `holes` 为空，或配对后尺寸不一致。

固定规则：

1. **`objects.size()==holes.size()` 且 `holes` 非空**：按 §6.3 配对规则同步过滤，两数组索引始终一一对应；
2. **`holes.empty()` 且 `objects` 非空**：仅对 `objects` 执行 `isBoxInsideTray()` 单框过滤，保留料盘内主体；输出时全部标记 `has_center_hole=false`，坐标回退到 object 中心（Empty 用 `PLATE_CELL` 中心，Wheel 用 `LUNZI` 中心）；
3. **尺寸不一致且无法重新配对**：记录警告，退化为仅过滤 `objects` 并清空 `holes`，不得按越界索引继续读写配对数组；
4. **两者均为空**：跳过区域过滤，由后续网格完整性校验暴露 `tray_not_found` 或不完整状态。

该规则保证“仅有外框、无中心孔”的样本仍可输出，同时禁止孤立小孔在无有效主体时进入结果。

## 7. 第二阶段：基于中位数的尺寸异常检查

### 7.1 统计基准

分别统计所有料盘内候选的宽、高中位数：

```cpp
struct BoxStats {
    float median_width = 0.0f;
    float median_height = 0.0f;
};
```

空位和轮子外框可以分别统计，避免不同类别尺寸互相影响。

初始阈值必须先用 `00038/00039/00044` 及后续现场样本统计宽高分布后确认；`0.65～1.40` 是 MVP 起始值，不视为无需标定的最终常量。

### 7.2 建议阈值

```cpp
bool width_ok = width >= median_width * 0.65f &&
                width <= median_width * 1.40f;
bool height_ok = height >= median_height * 0.65f &&
                 height <= median_height * 1.40f;
```

处理规则：

- 宽高均正常：普通候选；
- 单项异常：设置 `size_outlier = true`，但暂不删除；
- 中心位置能匹配标准网格时，允许异常候选参与竞争；
- 只有无法匹配网格且尺寸异常时，才最终拒绝。

这样可避免把 `00038/00039` 左下角真实空位提前删除。

## 8. 第三阶段：生成稳定的 2×4 标准网格

### 8.1 输入点

拟合输入包括：

- `PLATE_CELL` 的中心点；
- `LUNZI` 的中心点；
- 中心孔仅用于修正目标中心和提升匹配可信度。

中心位置优先使用对应孔中心；孔不可用时使用外框中心。

### 8.2 修改 `TrayFitting`

必须区分两个现有输出：

- `list_pts`：固定为 2 个建议抓取点，继续输出到 JSON `pts`；
- `list_pts_sort`：固定为 8 个有序网格参考点，只用于格位归位及 `sort_cells()`。

本节所称“标准网格输出”专指 `list_pts_sort`，不能把 `list_pts` 改成 8 点，否则会改变现有抓取接口语义。

`TrayFitting::fitCenterPoints()` 应保证：

1. 成功时 `list_pts_sort.size() == 8`；
2. 顺序固定为上排从左到右、下排从左到右；
3. 每次调用前清空 `clusters`、`list_pts` 和 `list_pts_sort`；
4. 拟合失败时返回 `false`，而不是留下部分结果。

为降低修改风险，MVP 建议新增旁路函数，保留旧函数用于 A/B 对比：

```cpp
bool fitOrderedGrid2x4(std::array<cv::Point2f, 8>& ordered_grid);
```

验证稳定后，再决定是否将旧接口调整为：

```cpp
bool fitCenterPoints();
```

入口清理：

```cpp
bool TrayFitting::fitCenterPoints()
{
    clusters.clear();
    list_pts.clear();
    list_pts_sort.clear();

    // 原拟合逻辑……

    if (list_pts_sort.size() != 8) {
        list_pts.clear();
        list_pts_sort.clear();
        return false;
    }

    return true;
}
```

还需修正或绕开当前 `estimateFourPoints()` 在收集每个点时重复调用的问题。新旁路函数应先完整收集该行的 `det_pts`，再调用一次估计器，避免结果依赖循环中的中间状态。

### 8.3 排序方式

不要继续依赖当前 `dp0/dp1` 的手工交叉下标排列。建议：

1. 将 8 个点旋转到料盘自身坐标系；
2. 按局部 `y` 聚成两行；
3. 每行按局部 `x` 从小到大排序；
4. 生成固定的 `0..7` 顺序。

伪代码：

```cpp
std::vector<cv::Point2f> orderTrayGrid(
    const std::vector<cv::Point2f>& points,
    const cv::RotatedRect& tray_rect);
```

必须校验两行各有 4 个点。

### 8.4 `fitOrderedGrid2x4` 拟合输入点规则

拟合阶段每个候选**只贡献一个点**，不得把主体外框与中心孔重复计入：

| 候选类型 | 拟合点来源 | 说明 |
|---|---|---|
| Empty | 有合法 `PLATE_HOLE` 时用孔中心，否则用 `PLATE_CELL` 中心 | 与 §6.3 规则 3 的 `has_center_hole` 一致 |
| Wheel | 有合法 `HOLE` 时用孔中心，否则用 `LUNZI` 中心 | 同上 |

**禁止**在 `detections_holes` / `list_holes` 中额外追加 `PLATE_HOLE` 作为独立拟合点。当前实现若同时 push `PLATE_CELL`、`LUNZI` 和 `PLATE_HOLE`，会把同一空位计两次，导致网格偏移，必须在 MVP 中修正。

`HOLE` 仅用于 Wheel 候选的中心修正，同样不得与 `LUNZI` 重复计入拟合输入。

## 9. 第四阶段：候选与网格一对一匹配

### 9.1 匹配距离

根据拟合网格估算相邻列间距和行间距，使用较小值作为基准：

```cpp
float match_threshold = std::min(column_spacing, row_spacing) * 0.40f;
```

候选中心到网格点距离超过阈值时，不参与该位置匹配。

### 9.2 一对一分配

候选与网格点形成所有合法边，按代价由小到大分配，保证：

- 一个候选最多匹配一个网格；
- 一个网格最多接收一个候选。

代价建议：

```cpp
cost = normalized_distance;
cost += size_outlier ? 0.20f : 0.0f;
cost -= has_center_hole ? 0.10f : 0.0f;
cost -= confidence * 0.05f;
```

候选数量固定且很少，使用贪心即可；如果后续场景复杂，可替换为匈牙利算法。

### 9.3 空位与轮子冲突裁决

同一网格同时出现 `PLATE_CELL` 和 `LUNZI` 时按以下顺序处理：

1. 匹配距离明显更小者；
2. 有合法对应中心孔者；
3. 置信度更高者；
4. 差异不足以裁决时设为 `Unknown`，禁止猜测。

不建议简单规定轮子永远覆盖空位，因为模型重叠误检可能导致错误占用状态。

MVP 不使用深度参与冲突裁决。当前 `plate_empty` 对各输出共用 `avg_depth`，没有 per-slot 深度证据；若 Phase 2 引入深度裁决，必须先增加每格独立深度采样、有效像素比例和类别深度阈值，不能直接复用公共 `avg_depth`。

## 10. `det_to_pts_plate_empty()` 改造

建议将当前函数拆分为：

```cpp
static bool buildTrayCandidates(...);
static bool fitTrayGrid(...);
static TrayPostprocessResult assignTraySlots(...);
static void convertSlotsToWorldPoints(...);
static void drawTrayPostprocessResult(...);
```

### 10.1 主流程步骤（修订）

主流程调整为：

```cpp
int det_to_pts_plate_empty(...)
{
    // 1. resolve tray（detected 或 fitted），设置 tray_source
    // 2. 收集并过滤料盘内候选（filterPairsByTray）
    // 3. 统计中位数尺寸并标记异常框
    // 4. 拟合并排序 2×4 网格（每候选单点，见 §8.4）
    // 5. 一对一归位并裁决每格状态
    // 6. 生成 pts_cell / pts_lunzi（按格位 index 升序，见 §11.4）
    // 7. 校验总数和 Unknown 数量
    // 8. 计算建议抓取点 pts（见 §10.2）
    // 9. 绘制最终后处理结果
}
```

步骤 1 说明：若检测到 `PLATE_TRAY` 则 `tray_source="detected"`；否则按 §6.4 从候选中心拟合临时 tray 并标记 `tray_source="fitted"`。不得在无 tray ROI 的情况下跳过区域过滤。

### 10.2 建议抓取点 `pts`（MVP）

MVP 采用**方案 A**：由 8 格有序参考点 `list_pts_sort` 计算 2 个抓取点，不再依赖当前 `dp0/dp1` 交叉下标逻辑。

固定规则：

1. 将 `list_pts_sort` 按 §8.3 排成上排 0–3、下排 4–7；
2. 左抓取点 = 左侧 2×2 区域（格位 0、1、4、5）的中心；
3. 右抓取点 = 右侧 2×2 区域（格位 2、3、6、7）的中心；
4. 两点经 `get_world_xy_by_z` 转换后写入 JSON `pts`，保持 `[x, y, z, radius]` 结构不变。

对应公式：

```cpp
left  = (grid[0] + grid[1] + grid[4] + grid[5]) * 0.25f;
right = (grid[2] + grid[3] + grid[6] + grid[7]) * 0.25f;
```

两个抓取点必须左右排列在上下两行之间，不能落在上排和下排各自的中间。

若 `list_pts_sort.size()!=8`，不得回退到旧 `dp0/dp1` 逻辑，应返回拟合失败或不完整状态。

原有世界坐标转换可保留：

```cpp
g_depth_info.get_world_xy_by_z(pixel.x, pixel.y, x, y, avg_depth);
```

但 `pts_cell` 和 `pts_lunzi` 应按照网格编号输出，而不是依赖检测器返回顺序。

## 11. 返回协议

### 11.1 成功条件

```text
empty_count + wheel_count == 8
unknown_count == 0
```

满足时：

```json
{
  "ret-code": 0,
  "ret-msg": "ok",
  "tray_complete": true,
  "empty_count": 6,
  "wheel_count": 2,
  "unknown_count": 0
}
```

### 11.2 不完整料盘

如果只有 7 个位置能够确定，Phase 2 严格模式返回：

```json
{
  "ret-code": -1,
  "ret-msg": "incomplete tray: resolved 7/8 positions",
  "tray_complete": false,
  "empty_count": 5,
  "wheel_count": 2,
  "unknown_count": 1
}
```

Phase 1 兼容模式则保持 `ret-code=0`，但必须返回：

```json
{
  "ret-code": 0,
  "ret-msg": "warning: incomplete tray, resolved 7/8 positions",
  "tray_complete": false,
  "empty_count": 5,
  "wheel_count": 2,
  "unknown_count": 1
}
```

为了排查问题，可增加：

```json
"slots": [
  {"index": 0, "state": "empty"},
  {"index": 1, "state": "wheel"},
  {"index": 2, "state": "unknown"}
]
```

如果现有上位机不能立即接受新增错误码，可分两阶段上线：第一阶段保留 `ret-code`，但增加 `tray_complete=false`；上位机适配完成后再启用严格错误码。

### 11.3 API 兼容策略

`pts_cell`、`pts_lunzi` 和 `pts` 的基本结构在 Phase 1 保持不变。新增字段只做追加，不要求旧上位机立即消费。

| 阶段 | JSON 行为 | `ret-code` | 上位机要求 |
|---|---|---|---|
| Phase 1 / MVP | 保留现有数组；新增 `tray_complete`、`slots`、计数字段和 `tray_source` | 暂时保持现有行为，不完整时仍为 `0`，但 `tray_complete=false` | 旧上位机可忽略新字段；新版本应记录告警并禁止危险动作 |
| Phase 2 / 严格模式 | 字段保持不变；启用严格完整性判定 | 不完整、冲突或无 tray 时返回非零 | 上位机必须处理失败状态并停止抓取/请求重拍 |

Phase 1 上线前仍需确认机器人控制链路是否会忽略 `tray_complete=false`。如果旧上位机只看 `ret-code` 并立即执行，必须先完成上位机联调，不能假设新增告警字段天然安全。

数组顺序在 Phase 1 改为稳定网格顺序，但数组仍只包含已确定的相应类别，不使用 8 个占位元素填充。格位索引通过新增 `slots` 表达，避免破坏旧数组解析逻辑。

### 11.4 网格顺序输出规则

`pts_cell`、`pts_lunzi` 与新增 `slots` 必须共享同一格位编号（§5 中 0–7 按行排序）：

1. **排序**：`pts_cell` 内各点按对应 Empty 格位 `index` 升序；`pts_lunzi` 按 Wheel 格位 `index` 升序；
2. **稀疏数组**：`pts_cell` / `pts_lunzi` 只包含已裁决为 Empty / Wheel 的格位，长度分别为 `empty_count` / `wheel_count`，**不**用 8 个占位元素填充 Unknown 格；
3. **完整映射**：通过 `slots[]` 表达全部 8 格，例如 `{"index":6,"state":"empty"}` 表示左下格为空位；上位机若需按格位索引取坐标，应优先读 `slots` 再结合对应数组下标；
4. **稳定性**：同一输入多次运行，`slots` 顺序固定为 `index` 0→7，`pts_*` 内部顺序稳定，便于 diff 与日志对齐。

Phase 1 在尚未完成格位归位（MVP-1）时，可暂时保持检测器顺序输出 `pts_cell` / `pts_lunzi`；MVP-2 启用 `assignTraySlots()` 后必须切换为本节规则。

## 12. 调试绘图修改

当前绘图位于后处理之前。建议保留原始框，同时在后处理完成后叠加最终状态：

| 标记 | 建议样式 |
|---|---|
| 原始检测框 | 1 px 细线 |
| 最终空位 | 绿色粗框，标注 `P0 empty` |
| 最终轮子 | 蓝色粗框，标注 `P1 wheel` |
| 未确定位置 | 黄色圆，标注 `P2 unknown` |
| 被拒绝候选 | 灰色框，标注拒绝原因 |

拒绝原因至少包括：

- `outside_tray`；
- `grid_distance_too_large`；
- `duplicate_slot`；
- `size_outlier_unmatched`；
- `class_conflict`。

这样结果图和 JSON 使用同一批最终数据，不再出现“画了 6 个、输出 5 个但看不出原因”的情况。

## 13. 实施步骤

建议分四个小提交实施：

1. **过滤职责拆分**
   - 删除重复面积过滤循环；
   - 增加中心点/交叠率区域过滤；
   - 增加候选拒绝原因。

2. **网格输出稳定化**
   - `fitCenterPoints()` 返回成功状态；
   - 固定输出 8 个有序网格点；
   - 增加两行各 4 点校验。

3. **一对一归位及状态输出**
   - 完成候选匹配；
   - 输出 `slots`、计数和完整性状态；
   - `pts_cell/pts_lunzi` 改为网格顺序。

4. **绘图与回归测试**
   - 区分原始框和最终框；
   - 执行完整数据集测试；
   - 记录每张图片的格位状态。

### 13.1 推荐交付阶段

| 阶段 | 内容 | 目的 |
|---|---|---|
| Hotfix 验证 | 删除重复循环；临时用 `0.75` 或中位数规则做 A/B，不作为最终上线实现 | 快速证明误删根因 |
| MVP | 新区域过滤、旁路 2×4 有序网格、位置归位、最终结果绘图、兼容字段 | 在不破坏旧返回码的前提下稳定输出 8 格状态 |
| Phase 2 | 严格错误码、上位机联调、可选 per-slot 深度裁决 | 完成安全闭环 |

Hotfix 的代码如果用于验证，合入 MVP 时必须删除，避免临时阈值与正式规则同时生效。

## 14. 回归测试

### 14.1 基础样本

| 图片 | 期望空位 | 期望轮子 | 期望总数 |
|---|---:|---:|---:|
| `build/plate/00038.jpg` | 6 | 2 | 8 |
| `build/plate/00039.jpg` | 6 | 2 | 8 |
| `build/plate/00044.jpg` | 8 | 0 | 8 |

测试命令：

```bash
cd build
./det-tool-lib plate/00038.jpg plate_empty test
./det-tool-lib plate/00039.jpg plate_empty test
./det-tool-lib plate/00044.jpg plate_empty test
```

推荐单张运行。批量测试时只枚举彩色 JPG，并跳过 `*-gray.jpg`；同名 PNG 仅作为深度图输入，不能作为独立彩色图再次推理：

```bash
cd build
for f in plate/*.jpg; do
    case "$f" in
        *-gray.jpg) continue ;;
    esac
    ./det-tool-lib "$f" plate_empty test
done
```

### 14.2 异常场景

还需构造或收集以下用例：

1. 一个真实框比中位数小 20%～30%，但中心位置正确；
2. 料盘外存在 `plate_cell` 或 `lunzi` 误检；
3. 同一格存在两个重叠候选；
4. 模型漏检一个位置；
5. 料盘部分越出图像边界；
6. `plate_tray` 未检测到；
7. 深度图局部无效；
8. 料盘有明显旋转或透视变化。

### 14.3 验收条件

- 三张基础样本全部满足预期数量；
- 所有 `tray_complete=true` 的逻辑成功结果满足 `empty + wheel == 8`；
- Phase 1 中“不完整但兼容返回”的结果必须满足 `tray_complete=false`，不得伪装成完整料盘；
- 漏检场景返回 `unknown` 或不完整状态，不得猜测为空位；
- 料盘外误检不得进入 `pts_cell/pts_lunzi`；
- 调试图显示的最终状态数量与 JSON 完全一致；
- 同一输入连续运行结果顺序稳定。

## 15. 不建议采用的方案

### 15.1 仅将 `0.8` 改为 `0.75`

可以修复当前样本，但仍以最大异常框为基准，不能解决后续尺度变化、重复检测和网格位置冲突，只适合作为临时验证。

### 15.2 直接截取前 8 个检测结果

检测顺序不稳定，而且可能包含料盘外误检，会丢弃真正目标。

### 15.3 不足 8 个时自动补为空位

这会把模型漏检转换为机器人可执行的错误抓取点，风险较高。不确定位置必须输出 `Unknown`。

## 16. 预期结果

完成改造后，`00038` 和 `00039` 的左下空位不会再因为面积为最大框的 79.35% 而被误删。最终输出由固定 8 格状态决定：

```text
00038: 6 empty + 2 wheel = 8
00039: 6 empty + 2 wheel = 8
00044: 8 empty + 0 wheel = 8
```

同时，任何不足 8 个确定位置的输入都会明确报告不完整状态，便于上位机拦截和现场调试。
