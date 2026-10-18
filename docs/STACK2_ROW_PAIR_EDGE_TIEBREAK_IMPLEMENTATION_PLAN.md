# 码垛 2 多行候选选行：`rect + 上下边缘` 二级裁决施工方案

## 1. 文档目的

本文给出 `plate`/`stack_slots` 粗定位在 ROI 同时覆盖三排候选圆槽时的代码改造方案。

目标是在保持现有应用调用和返回协议不变的前提下，让算法稳定选择目标托盘的 `2 行 × 4 列` 网格：

- `rect` 仍是应用端传入的目标范围先验；
- `rect` 能唯一选择一对相邻行时，继续采用现有结果；
- `rect` 对多组相邻行打分并列时，使用托盘上下外边缘做二级裁决；
- 两级证据都不足时继续安全失败，不猜测目标行；
- 边缘判断必须使用 ROI 灰色遮罩前的真实图像。

本次仅修改库内部选行和图像传递逻辑，不修改模型、手眼矩阵、深度计算、公共 C API 或业务 JSON 字段。

---

## 2. 问题与复现基线

业务调用：

```text
task: stack_slots
type: plate
top: 0
image size: 1280 × 800
stack2_rect: [79,111,1214,111,1214,715,79,715]
```

应用配置中的 `stack2_rect` 最终作为算法库输入 JSON 的 `rect` 传入。

该 ROI 同时覆盖三排圆槽：

```text
第 1 排：上方相邻托盘的一排
第 2 排：目标托盘上排
第 3 排：目标托盘下排

相邻行对 0：第 1 排 + 第 2 排
相邻行对 1：第 2 排 + 第 3 排（业务目标）
```

已确认基线：

| 图片 | 传入原 `rect` | 当前结果 |
|---|---|---|
| `00156` | 是 | 8 空位，成功 |
| `00157` | 是 | `rect does not uniquely select one adjacent row pair` |
| `00212` | 是 | 同上 |

不传 `rect` 时，现有上下边缘算法会运行，三张图片均能正确选择行对 1：

| 图片 | 行对 0 边缘综合分 | 行对 1 边缘综合分 | 结果 |
|---|---:|---:|---|
| `00156` | 37.70 | 85.01 | 8 空位 |
| `00157` | 23.94 | 81.32 | 8 空位 |
| `00212` | 18.71 | 83.82 | 8 空位 |

这组对照证明：边缘算法本身能够区分两块托盘；问题是传入 `rect` 后该算法没有执行。

---

## 3. 当前根因

### 3.1 `rect` 分支短路边缘分支

当前 `TrayFitting::fitOrderedGrid2x4()` 的核心结构是：

```cpp
if (selection_polygon != nullptr) {
    // 只按 rect 内候选点数量选择相邻行对
    // 并列时直接 return false
}
else if (edge_image != nullptr) {
    // 只有未传 rect 时才计算上下边缘分数
}
```

因此，只要应用传入合法的 8 点 `rect`：

1. 代码进入 `selection_polygon` 分支；
2. ROI 覆盖三排时，行对 0 和行对 1 可能得到相同点数；
3. `score_tied=true`；
4. 函数直接返回失败；
5. `else if (edge_image...)` 永远没有机会执行。

这不是阈值不合适，而是裁决流程的优先级设计问题。

### 3.2 当前边缘图不是严格意义上的原图

`AI_det_xialiao()` 在推理前会调用：

```cpp
fillOutsideQuadrilateralFast(g_bgrImage, g_rect, 1.1f, grayColor);
```

它将扩张后 ROI 外部填成灰色。后处理阶段 `handle_ret_xialiao()` 再从当前 `image` 克隆 `img_edge`。

因此，即使简单把 `else if` 改成第二个独立的 `if`，边缘算法拿到的仍可能是已遮罩图像，存在两个风险：

- ROI 灰色填充边界可能形成强烈的人工边缘；
- 真正托盘外框若在扩张 ROI 外，会被遮掉。

施工时必须同时解决短路和边缘图来源两个问题。

---

## 4. 设计目标与非目标

### 4.1 设计目标

1. 原 `stack2_rect` 保持不变时，`00156`、`00157`、`00212` 均稳定输出目标托盘 8 个格位。
2. `rect` 是一级约束，边缘只在 `rect` 无法唯一裁决时介入。
3. 不硬编码“选最下面两排”，避免影响相机旋转、其他工位和其他托盘方向。
4. 边缘计算使用本次请求的、遮罩前的、与推理图坐标严格对齐的彩色图。
5. 证据不足时返回明确警告，不输出可能属于相邻托盘的 8 点。
6. 保持现有 `AI_det()` 调用方式、`rect` 输入格式、`ret-code`、`pts`、`pts_cell`、`pts_cell_rect`、`pts_lunzi` 等字段不变。
7. 不引入跨帧图像缓存，避免再次出现旧图/旧深度沿用问题。

### 4.2 非目标

- 不重新训练或替换 `xialiao.m`；
- 不修改检测类别；
- 不改变候选与 2×4 网格的一对一归位逻辑；
- 不修改世界坐标转换或抓取深度；
- 不调整 `top=1` 的单格精拍流程；
- 不在本次引入托盘上下堆叠关系的跨帧跟踪。

### 4.3 修改文件和函数边界

| 文件 | 函数/区域 | 修改内容 |
|---|---|---|
| `det-src/GridFitter.hpp` | `TrayFitting::fitOrderedGrid2x4()` | 拆分 rect/边缘评分；增加 rect 并列时的边缘二级裁决；返回内部诊断结果 |
| `det-src/det.cpp` | `AI_det_xialiao()` | 在 ROI 遮罩前保存本次请求的边缘源图 |
| `det-src/det.cpp` | `ai_det_one_image()` | 对推理图和边缘源图执行相同翻转，并继续向后传递 |
| `det-src/det.cpp` | `handle_ret_xialiao()` | 使用调用链传入的原始边缘源图，不再从已遮罩图 clone |
| `det-src/det.cpp` | `det_to_pts_plate_empty()` / `det_to_pts_plate()` | 保持主体逻辑不变，仅继续将正确的 `edge_image` 传入网格拟合器 |
| `det-src/det.h` | 公共声明 | 不修改 |

不得修改 `AI_det()` 的参数列表或导出符号。

---

## 5. 推荐的裁决流程

多于两排候选时，统一执行以下状态机：

```text
构造所有相邻行对
        │
        ├─ 有 rect
        │    │
        │    ├─ rect 最高分唯一且最高分 >= 4
        │    │      └─ 直接采用该行对，source=rect
        │    │
        │    ├─ rect 最高分并列
        │    │      └─ 仅对并列行对计算上下边缘分数
        │    │             ├─ 分数及领先差满足阈值：采用，source=edge_tiebreak
        │    │             └─ 证据不足：安全失败
        │    │
        │    └─ rect 最高分 < 4
        │           └─ 安全失败，不允许边缘越过业务 ROI 自行选盘
        │
        └─ 无 rect
             └─ 对全部相邻行对计算上下边缘分数
                    ├─ 唯一可信：采用，source=edge
                    └─ 证据不足：安全失败
```

关键原则：

- 边缘裁决是 `rect` 并列时的 tie-break，不是对 `rect` 的替代；
- 只对 `rect` 最高分并列集合计算和比较边缘，避免边缘选中一个明显位于 ROI 外的行对；
- `rect` 内总证据不足时不降级到边缘，因为这通常表示 ROI、检测或调用参数异常。

---

## 6. 内部数据结构

建议在 `GridFitter.hpp` 内新增内部结构，不对外暴露：

```cpp
enum class RowPairSelectionSource {
    None,
    OnlyPair,
    Rect,
    Edge,
    EdgeTieBreak
};

struct RowPairScore {
    int pair_index = -1;
    int rect_score = -1;
    float edge_top = -1.0f;
    float edge_bottom = -1.0f;
    float edge_combined = -1.0f;
};

struct RowPairSelectionResult {
    bool success = false;
    int selected_pair = -1;
    RowPairSelectionSource source = RowPairSelectionSource::None;
    std::string failure_reason;
    std::vector<float> row_centers;
    std::vector<RowPairScore> pair_scores;
};
```

说明：

- `pair_index=i` 表示 `all_rows[i] + all_rows[i+1]`；
- `row_centers` 使用料盘局部坐标，主要用于日志；
- 首版不把该结构加入业务 JSON，避免输出协议变化；
- `fitOrderedGrid2x4()` 可通过可选指针返回诊断结果。

建议接口调整为：

```cpp
bool fitOrderedGrid2x4(
    std::array<cv::Point2f, 8>* ordered_grid = nullptr,
    const std::vector<cv::Point2f>* selection_polygon = nullptr,
    std::string* failure_reason = nullptr,
    const cv::Mat* edge_image = nullptr,
    RowPairSelectionResult* selection_result = nullptr);
```

这是内部 C++ 接口变化，不影响公共 `AI_det()` ABI。

---

## 7. 代码拆分方案

### 7.1 提取 rect 行对评分

新增私有辅助函数：

```cpp
static std::vector<RowPairScore> scoreRowPairsByPolygon(
    const std::vector<LocalRow>& rows,
    const std::vector<cv::Point2f>& polygon);
```

规则保持现状：统计每个相邻行对中落在 polygon 内的原始候选点数量。

处理结果：

```cpp
best_rect_score = max(pair.rect_score);
tied_pairs = 所有 rect_score == best_rect_score 的 pair；
```

- `best_rect_score < 4`：失败，原因 `rect contains too few tray candidates`；
- `tied_pairs.size()==1`：直接采用；
- `tied_pairs.size()>1`：进入边缘二级裁决。

### 7.2 提取边缘行对评分

把当前 `else if (edge_image...)` 内的边缘实现提取为可复用函数：

```cpp
static bool scoreRowPairsByTrayEdges(
    const std::vector<LocalRow>& rows,
    const std::vector<int>& allowed_pairs,
    const cv::Mat& edge_image,
    std::vector<RowPairScore>* pair_scores,
    int* selected_pair,
    std::string* failure_reason);
```

`allowed_pairs` 的语义：

- 有 rect 且 rect 并列：只传并列的行对 index；
- 无 rect：传全部合法相邻行对。

首版沿用已验证阈值：

```cpp
constexpr float min_edge_score = 24.0f;
constexpr float min_score_margin = 6.0f;
```

成功条件：

```text
最高综合分 >= 24
且
最高综合分 - 第二名综合分 >= 6
```

只有一个可评分行对时，只校验最低分，不要求领先差。

本次不改变现有边缘综合分公式：

```cpp
combined = strongest + secondary * 0.35f;
```

先用现有公式完成流程修复，再基于更多现场样本决定是否调参，避免同时改变流程和评分模型导致回归归因困难。

### 7.3 重写多行选择主逻辑

`fitOrderedGrid2x4()` 中 `all_rows.size()>2` 的逻辑改为：

```cpp
if (selection_polygon 可用) {
    计算 rect_scores；

    if (最高分不足) {
        失败；
    } else if (最高分唯一) {
        selected_row = 唯一最高分行对；
        source = Rect；
    } else if (edge_image 可用) {
        只对并列行对执行边缘评分；
        成功则 source = EdgeTieBreak；
        失败则返回明确原因；
    } else {
        失败：rect 并列且没有可用边缘图；
    }
} else if (edge_image 可用) {
    对所有行对执行边缘评分；
    成功则 source = Edge；
} else {
    失败；
}
```

`all_rows.size()==2` 时不做额外选择，保持现有直接拟合行为，诊断来源记为 `OnlyPair`。

### 7.4 失败消息细分

建议替换当前笼统消息，便于现场定位：

| 场景 | `failure_reason` |
|---|---|
| rect 内有效候选不足 | `rect contains too few tray candidates` |
| rect 并列但没有边缘图 | `rect row-pair tie and edge image is unavailable` |
| 边缘最高分不足 | `tray outer edge score is too low` |
| 边缘第一、第二名差距不足 | `tray outer edge does not resolve rect row-pair tie` |
| 无 rect、无边缘图且多行 | `multiple tray row pairs; rect or outer edge is required` |

对外仍由现有代码统一拼接：

```text
warning: grid fit failed: <failure_reason>
```

---

## 8. 遮罩前原图的传递方案

### 8.1 采集时机

在完成以下步骤之后保存边缘源图：

1. 测试图片读取或相机采图；
2. 多曝光融合（如启用）；
3. 确认图像非空。

必须在 `fillOutsideQuadrilateralFast()` 之前执行：

```cpp
cv::Mat tray_edge_source;
if (needsTrayRowEdgeSelection(g_det_type) &&
    g_top_capture != 1 &&
    !g_bgrImage.empty())
    tray_edge_source = g_bgrImage.clone();
```

`needsTrayRowEdgeSelection()` 首版包含：

```text
plate
plate_empty
plate_zero
plate_empty_zero
```

不为其他模式额外复制 1280×800 彩色图。

### 8.2 内部调用链调整

推荐使用本次请求的局部 `cv::Mat` 逐层传递，不新增全局图像缓存：

```text
AI_det_xialiao
  └─ ai_det_one_image(..., cv::Mat* tray_edge_source)
       └─ handle_ret_xialiao(..., const cv::Mat* tray_edge_source)
            └─ det_to_pts_plate[_empty](..., p_img_edge)
                 └─ fitOrderedGrid2x4(..., edge_image)
```

对应内部签名建议：

```cpp
int ai_det_one_image(
    std::vector<Detection_pan>& detections_pan,
    json& j,
    char** ret_json,
    int* json_len,
    cv::Mat* tray_edge_source = nullptr);

int handle_ret_xialiao(
    cv::Mat& image,
    std::vector<std::vector<Detection_pan>>& det_outs,
    json& j,
    const cv::Mat* tray_edge_source = nullptr);
```

`handle_ret_xialiao()` 不再从已遮罩的 `image` 重新 clone 边缘图。

### 8.3 翻转坐标一致性

当前水平/垂直翻转发生在 `ai_det_one_image()` 内。边缘源图必须执行与推理图完全相同的翻转，否则候选点坐标与边缘图不一致。

建议提取：

```cpp
static void applyConfiguredFlip(cv::Mat& image);
```

并在同一位置调用：

```cpp
applyConfiguredFlip(g_bgrImage);
if (tray_edge_source != nullptr && !tray_edge_source->empty())
    applyConfiguredFlip(*tray_edge_source);
```

禁止只翻转检测图或重复翻转边缘图。

### 8.4 生命周期与缓存要求

- `tray_edge_source` 每次 `AI_det_xialiao()` 请求重新创建；
- 请求结束后自动释放；
- 测试图片路径无效或读图失败时必须为空；
- 不允许沿用上一请求的图；
- 不建议增加 `g_tray_edge_image` 全局变量。

这样可以避免图像、检测结果和深度来自不同帧。

---

## 9. 日志与调试信息

首版不增加生产 JSON 字段，避免应用端严格解析受到影响。通过 `REL_DBG`、现有调试开关或统一日志输出以下内容：

```text
grid rows=3 centers=[...]
grid rect pair=0 score=8
grid rect pair=1 score=8
grid rect tied_pairs=[0,1]
grid edge pair=0 top=... bottom=... combined=...
grid edge pair=1 top=... bottom=... combined=...
grid selected_pair=1 source=edge_tiebreak
```

失败时必须输出：

```text
grid selection failed source=edge_tiebreak reason=...
```

调试图建议用不同颜色标注：

- 全部行中心：灰色水平线；
- rect 并列行对：黄色；
- 最终行对：绿色；
- 采样到的上/下边缘：青色；
- 被排除的相邻托盘行：红色。

调试绘图不得改变生产 JSON 坐标和检测图数据。

---

## 10. 公共接口与应用端影响

### 10.1 输入

应用端继续传入原参数：

```json
{
  "xialiao_type": "plate",
  "top_capture": 0,
  "rect": [79,111,1214,111,1214,715,79,715]
}
```

应用配置中的 `stack2_rect` 无需修改。

### 10.2 输出

保持以下字段及语义不变：

```text
ret-code
ret-msg
pts
pts_cell
pts_cell_rect
pts_lunzi
slots
tray_complete
empty_count
wheel_count
unknown_count
```

成功后仍是正常的 `ret-code=0`、`ret-msg=ok`。本次不新增应用必须解析的字段。

### 10.3 深度与坐标

本次仅决定采用哪两排图像候选：

- 不修改深度图；
- 不修改 `avg_depth`/格位深度计算；
- 不修改像素转世界坐标；
- 不修改 `pts_cell` 与 `pts_cell_rect` 的坐标系；
- 不修改抓取高度和角度。

因此应用端不需要重新标定或修改抓取深度补偿。

---

## 11. 施工步骤

### Step 1：只做代码提取，不改变行为

1. 增加 `RowPairScore` 和 `RowPairSelectionResult`；
2. 将 rect 评分提取成函数；
3. 将边缘评分提取成函数；
4. 保持原优先级，确认现有批量结果完全一致。

目的：先降低大函数复杂度，确保评分提取没有引入数值变化。

### Step 2：保留遮罩前边缘源图

1. 在 ROI 遮罩前 clone 本次请求图像；
2. 沿内部调用链传递；
3. 提取并同步图像翻转；
4. `handle_ret_xialiao()` 改用传入原图；
5. 验证不传 rect 时边缘分数与当前基线基本一致。

### Step 3：启用 rect 并列时的边缘 tie-break

1. rect 唯一时保持原结果；
2. rect 并列时构造 `tied_pairs`；
3. 只对 `tied_pairs` 调用边缘评分；
4. 满足最低分和领先差时选中；
5. 不满足时返回细分失败原因。

### Step 4：增加日志和调试绘图

1. 输出行数量、行中心、rect 分数；
2. 输出边缘上下分数及综合分；
3. 输出最终选择来源；
4. 确认 Release 正常构建时不会产生大量无控制日志。

### Step 5：全量回归并同时构建 Debug/Release

1. 跑最小复现集；
2. 跑 `build/plate` 全部 jpg；
3. 跑 `build/lunzi` 回归，确认未影响轮子模式；
4. 构建 Debug 和 Release 两套库；
5. 对比公共导出符号和 JSON 字段。

---

## 12. 测试矩阵与验收标准

### 12.1 必测复现集

| 用例 | rect | 期望 |
|---|---|---|
| `00156` | `[79,111,...,715]` | 8 空位，选择行对 1 |
| `00157` | 同上 | 8 空位，选择行对 1 |
| `00212` | 同上 | 8 空位，选择行对 1；上方 4 个轮子不得进入结果 |
| `00156` | 不传 | 8 空位，边缘选择行对 1 |
| `00157` | 不传 | 8 空位，边缘选择行对 1 |
| `00212` | 不传 | 8 空位，边缘选择行对 1 |
| 三张图 | 上边界改为 `Y=240` | 均为 8 空位，rect 唯一选择或正常拟合 |

### 12.2 安全失败用例

| 场景 | 期望 |
|---|---|
| rect 内少于 4 个有效候选 | 失败，不用边缘跨过 rect 选盘 |
| rect 并列且边缘图为空 | 失败，消息明确 |
| 两组边缘综合分差小于 6 | 失败，不猜测 |
| 边缘最高分低于 24 | 失败，不猜测 |
| ROI/图像尺寸不匹配 | 失败或禁用边缘裁决，不越界访问 |
| 只有两排候选 | 保持原成功路径，不受 tie-break 影响 |

### 12.3 历史回归

至少包含：

- `plate/00013.jpg`；
- `plate/00014.jpg`；
- `plate/00015.jpg`；
- `plate/00018.jpg`；
- `plate/00038.jpg`；
- `plate/00039.jpg`；
- `plate/00044.jpg`；
- `plate` 目录全部 `.jpg`；
- `lunzi` 目录全部 `.jpg`；
- `top=1` 的单格精拍样本。

批量测试只遍历 `.jpg`，同名 `.png` 作为深度图，不把深度图再次当彩色输入。

### 12.4 稳定性测试

对 `00156`、`00157`、`00212` 每张连续离线调用至少 20 次：

- 选择行对必须一致；
- `pts_cell` 数量必须一致；
- 不得出现上一帧图像或深度残留；
- 不得因候选容器遍历顺序产生不同结果。

### 12.5 验收标准

全部满足才可交付：

1. 原 `stack2_rect` 下，三个最小复现样本全部通过；
2. `00212` 上方相邻托盘的轮子不进入目标 8 格；
3. 低置信边缘场景仍安全失败；
4. `plate` 全量回归无新增失败；
5. `lunzi`、`top=1` 行为不变；
6. 公共 API 和生产 JSON 结构不变；
7. Debug、Release 均编译成功；
8. Release 库尺寸和推理耗时无明显异常增长。

---

## 13. 性能与内存评估

1280×800 BGR 图像单次 clone 约占：

```text
1280 × 800 × 3 ≈ 3.07 MB
```

该内存仅在 `plate` 相关模式、单次请求生命周期内存在。边缘评分只在候选超过两排时产生额外计算，规模通常只有 2～3 个相邻行对，相比模型推理耗时可忽略。

优化原则：

- 不为 `lunzi`、`feilun` 等无关模式 clone；
- 不保留跨请求缓存；
- 不在每个行对内部重复 clone 整图；
- 边缘评分共用一次模糊后的工作图。

---

## 14. 风险与控制措施

### 风险 1：ROI 人工边缘影响评分

控制：边缘算法只使用遮罩前源图，禁止使用 `fillOutsideQuadrilateralFast()` 后的图。

### 风险 2：图像翻转后坐标不一致

控制：推理图和边缘源图调用同一个 `applyConfiguredFlip()`，并增加翻转场景测试。

### 风险 3：边缘规则覆盖 rect 意图

控制：边缘仅比较 rect 最高分并列集合；rect 证据不足时不降级选盘。

### 风险 4：低对比度透明托盘边缘不稳定

控制：保留最低综合分和领先差门槛；不足时失败，不硬选最下两排。

### 风险 5：改动影响其他模式

控制：边缘源图只在 `plate` 相关模式创建；`top=1` 维持原分支；全量回归 `lunzi`。

### 风险 6：调试输出改变应用解析

控制：首版只增加内部诊断结构和受控日志，不增加生产必选 JSON 字段。

---

## 15. 不建议采用的方案

### 15.1 永远选择最下面两排

问题：依赖固定相机方向和托盘相对位置，旋转、换工位或识别上层托盘时容易选错。

### 15.2 rect 并列时直接选择最后一个行对

问题：本质仍是用容器顺序猜测，无法解释，也无法在四排候选场景扩展。

### 15.3 彻底忽略 rect，只用边缘

问题：会丢失应用端明确提供的目标区域先验，并可能选中 ROI 外托盘。

### 15.4 只把 `else if` 改成 `if`

问题：没有定义两种结果如何融合，而且边缘图仍可能来自 ROI 遮罩后的图像。

### 15.5 只缩小生产 ROI

问题：可以作为现场临时规避，但不能解决临界光照导致候选行进入/退出时的算法稳定性，也会增加机器人位置和托盘尺寸变化带来的维护成本。

---

## 16. 回退方案

若新 tie-break 在全量回归中出现不可接受的误选：

1. 保留已提取的评分函数和原图传递；
2. 临时关闭 `rect` 并列时的边缘裁决，恢复安全失败；
3. 应用端临时将 `stack2_rect` 上边界下移至 `Y=220～240`；
4. 根据失败样本重新标定 `min_edge_score` 和 `min_score_margin`；
5. 不回退已验证的图像缓存清理、格位归位及输出协议兼容改动。

---

## 17. 最终施工结论

本次应实现的是“分层证据裁决”，而不是新增一个互相竞争的独立规则：

```text
rect 负责限定业务目标范围
        +
上下外边缘负责解决 rect 内多行组合并列
        +
低置信时安全失败
```

按此方案完成后，应用端可以继续使用现有 `stack2_rect` 和返回解析；算法库内部将在 `rect` 无法唯一选择两排时自动启用已经存在且经样本验证有效的上下边缘方法。
