# WSL 算法库开发交接

交接日期：2026-10-18。适用仓库：[xiaoshuang112/hand_outside_eye](https://github.com/xiaoshuang112/hand_outside_eye)，分支 `main`。初始算法导入基线为 `1cc6675`，来源是原正式工程 `feilun` 分支的 `b25b48b`；现已合入 90 的角度改动 `563e25a`。来源记录见 [PROVENANCE.json](../PROVENANCE.json)。当前版本以 `git rev-parse HEAD` 为准，不把导入基线号当成最新版本。

后续在 WSL 中修改图像算法、拟合脚本和测试，再交叉编译给 RDK X3 使用。WSL 可以完成 CPU 离线回归，但不能据此确认 X3 BPU、MVS 相机、宿主应用和机械臂的实际运行效果。

## 2026-10-19：当前目标改为圆柱工件

当前 `config/top_face/algorithm.json` 默认 `workpiece_shape="circle"`，检测圆柱的圆形顶面。背景为用户指定的 `capture_20261019_163657_840_94154fa5_fused_undistorted.png`，原样复制到 `config/top_face/cylinder_background.png`；配套坐标记录也已保留。只使用本批带 `undistorted` 的图片，它们已是生产 `alpha=1` 坐标，内部不再去畸变、缩放或翻转。

圆形分支采用正向背景差分、实心顶面检查、稳健圆周拟合和局部亚像素边缘细化。它不会加载矩形中心模型或 BPU 背景特征模型。旧矩形代码仅供明确指定 `workpiece_shape="rectangle"` 的历史回归；当前配置不走这条路径。

对外 C ABI、`xialiao_type="top_face"`、请求参数、排序、`top_capture` 和坐标换算保持原有接口。输出 `center_px`、`top_center_px`、`geometric_center_px` 均为拟合的顶部圆心，`center_source="circle_fit"`；`apparent_length_px`、`apparent_width_px` 现在表示直径，对应旧长度输出槽也表示圆柱直径。`corners_px` 保留四点格式，含义是圆的包围框，不是矩形工件的物理四角。新增 `shape="circle"`、`radius_px`、`diameter_px`、圆周覆盖率及拟合残差等诊断字段。

用户已确认算法 `angle_deg` 固定 **0°**（配置 `circle_angle_deg=0`），并以 `angle_defined=false` 标记无唯一方向。这是算法字段，不代表机器人最终欧拉 RZ 为零；宿主应用仍按自己的模式和零位参数处理，算法未改动现场参数或机器人脚本。

运行 `bash tests/run_cylinder_checks.sh` 可执行合成圆心精度、侧壁干扰、污迹/矩形/圆环拒绝、ROI、异常输入和本批七张图片检查，并将圆周叠加图与 JSON 保存到 `calibration/output/cylinder_detection/results/`。无本批数据时只运行合成检查。`tests/run_cpu_checks.sh` 也包含合成圆形检查和原坐标映射回归。

以下各节中的矩形顶部、四边形中心、长轴角、旧模型及生产参数表是此前交接记录；当前圆柱检测行为以本节和当前配置为准。现有坐标模型的几何适用高度仍须随实际圆柱顶面核对。

## 1 当前交付状态

完整算法源码、中心模型、背景快照、当前坐标模型、手眼文件、拟合脚本和离线检查已进入仓库。Qt 应用、机器人 SDK、MVS 安装包、Horizon sysroot、运行凭据和编译产物不在仓库中。

本仓库使用新 `main` 分支。现场旧工程继续使用 `feilun` 分支；本次 GitHub 导出没有修改旧工程的 `main`。GitHub 中的当前三点映射配置，比原正式工程提交中的三硬币配置更新，不能用旧工程的配置覆盖它。

最后已确认的现场参数如下。它们是交接快照，后续修改时应重新读取现场文件。

| 项目 | 已确认值 | 所属位置 |
|---|---|---|
| 平台 | RDK X3，AArch64 | 目标设备 |
| 检测命令 | `xialiao_type: "top_face"` | 算法请求 |
| 最终像素中心 | 顶部四边形对角线交点 | 算法 |
| 当前 XY 模型 | 三点拟合，`homography` | `config/top_face/tcp_xy_mapping.json` |
| X 补偿 | `diff_x = 7.5` mm | 宿主应用，映射后再加 |
| 图像零位角 | `zeroAngle = -88.74`° | 宿主应用 |
| 机器人零位角 | `zeroEuler = 135`° | 宿主应用 |
| 角差方向 | `plusDiff = 0`，减去图像差角 | 宿主应用 |
| 固定下抓 Z | `tcp_z = 225.39` mm | 宿主应用 |
| 高度输入 | 最后读到 `35` mm | 当前固定顶部模型不使用它计算 XY |

上表是同步前最后确认的应用参数，采用旧顺时针角度约定。当前 [config/app_mapping.example.json](../config/app_mapping.example.json) 已改为适配逆时针输出的迁移示例，详见第 7 节；它不是现场参数的实时副本。**算法库不会加载这个示例文件**，也不负责向机器人寄存器写值。

### 版本同步约定

GitHub `main` 是统一开发基线，90 正式工程 `feilun` 与它指向同一提交，分支名可以不同。90 原分支历史保存在 `backup/feilun-before-github-sync-*`，原 `main` 不修改。GitHub 远程在 90 命名为 `hand-outside-eye`，避免与旧项目的其他远程混淆。

WSL 修改先提交到 GitHub `main`；90 无本地修改时，通过以下命令同步：

```bash
git fetch hand-outside-eye main
git switch feilun
git merge --ff-only hand-outside-eye/main
git rev-parse HEAD
```

若临时在 90 的 `feilun` 提交，应显式 `git push hand-outside-eye HEAD:main`，然后 WSL `git pull --ff-only`。两边出现分叉或未提交修改时先检查差异，不用强推或硬重置掩盖问题。对外报版本同时说明仓库、分支、提交号；库二进制另记 SHA256，不能仅凭源码 HEAD 判断已部署版本。

## 2 WSL 中先跑通什么

建议使用 WSL 2 和 Ubuntu，将源码放在 Linux 文件系统中，例如 `~/work/hand_outside_eye`。Microsoft 对 Linux 命令行工作流也建议使用 WSL 文件系统，避免跨 `/mnt/c` 编译产生额外文件访问开销。[WSL 安装说明](https://learn.microsoft.com/en-us/windows/wsl/install)、[文件系统说明](https://learn.microsoft.com/en-us/windows/wsl/filesystems)。

以下命令在 WSL 的 Bash 中执行：

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential cmake ninja-build pkg-config ripgrep \
  python3 python3-venv python3-pip libopencv-dev

mkdir -p ~/work
cd ~/work
git clone https://github.com/xiaoshuang112/hand_outside_eye.git
cd hand_outside_eye
git switch main
pkg-config --modversion opencv4
bash tests/run_cpu_checks.sh
```

这个脚本直接编译圆柱合成检查及四组历史 CPU 回归，不连接相机、机械臂或 BPU。正常结束时包含以下输出；中间出现针对非法输入的 `top_face_init` 报错是负面测试的预期行为，要以进程退出码为准。

```text
diagonal center, optional fusion prior, legacy mapping and ray plane checks passed
Production mapping, height independence and counter-clockwise angles passed
```

不要将根目录 `cmake` 默认配置当成普通 CPU 构建入口：默认 `USE_RDK=ON` 需要 Horizon SDK，改成 `OFF` 又会进入 CUDA/TensorRT 路径。纯图像与映射开发先使用上述 CPU 检查。

Ubuntu 软件源提供的 OpenCV 版本与完整工程要求不同。例如 [Ubuntu 24.04 的 libopencv-dev](https://packages.ubuntu.com/noble/libopencv-dev) 为 4.6 系列，而根 CMake 要求 OpenCV 4.10 或更新。CPU 检查使用系统 OpenCV 4；完整 X3 构建使用目标 sysroot 中的 OpenCV，不能混用宿主库。

Python 拟合单独使用虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/pip install -r calibration/requirements.txt
.venv/bin/python calibration/fit_top_plane.py
```

输出在 `calibration/output/`，不会覆盖生效配置。该目录被 Git 忽略。不要把 Windows 侧已有 Python 虚拟环境直接复制到 WSL。

## 3 代码入口和修改位置

| 要处理的问题 | 先看哪里 |
|---|---|
| 背景匹配、工件候选、顶部四边形、融合评分 | `det-src/top_face.cpp` |
| ONNX / X3 中心模型推理、裁剪和坐标还原 | `det-src/center_heatmap_infer.cpp`、`.hpp` |
| 图像特征 BPU 路径 | `det-src/x3_features.h` |
| 对外接口声明 | `det-src/det.h` |
| 采集、融合、去畸变、请求解析、输出组装 | `det-src/det.cpp` |
| 相机驱动 | `det-src/cameras/MVSCamera.cpp` |
| 坐标映射配置加载与计算 | `top_face_init`、`top_face_tcp_xy`，在 `top_face.cpp` |
| 像素结果转为接口输出 | `det_to_pts_top_face`，在 `det.cpp` |
| 当前三点拟合 | `calibration/fit_top_plane.py` |
| CPU 回归入口 | `tests/run_cpu_checks.sh` |
| 当前矩阵的 C++ 数值和高度独立性测试 | `tests/top_plane_snapshot_test.cpp` |
| 高度感知旧模型、异常输入、模型兼容测试 | `tests/top_face_center_mapping_test.cpp` |
| X3 交叉编译配置 | `toolchain_rdk_x3.cmake` |

当前主链路为：

```text
AI_det_init / AI_det
  → 相机采集或测试图输入
  → 多曝光融合、去畸变
  → top_face_init / top_face_detect
  → 背景分割、粗区域、顶部四边形候选
  → 中心 NN 与几何评分选候选
  → 选中四边形的对角线交点
  → det_to_pts_top_face
  → top_face_tcp_xy
  → JSON 返回宿主应用
```

NN 会影响候选生成与选择，但不会直接把自己的中心点写成最终抓取中心。`center_px`、`top_center_px`、`geometric_center_px` 对应选中四边形的对角线交点，`center_source` 为 `diagonal_intersection`。不要把 `center_model_center_px` 当成最终点。

融合行为以 `selectCandidate`、`coarseCenterPrior` 和 [TOP_FACE_TCP_MAPPING.md](TOP_FACE_TCP_MAPPING.md) 为准。旧聊天中“NN 仅诊断、不影响候选选择”的描述已经过时。

## 4 输入输出约定

保留 C ABI：`AI_det_init`、`AI_det`、`AI_det_close`。具体参数和图像内存约定见 `det-src/det.h`，不要在算法内部改变现有宿主的调用协议或返回数据所有权。

当前检测请求的相关字段示例：

```json
{
  "xialiao_type": "top_face",
  "ref_depth": 580,
  "top_capture": 1,
  "workpiece_height_mm": 35
}
```

这里的 `580` 是历史参考相机深度示例，不是台面 Base Z，也不是工件高度。`ref_depth` 仍参与旧 Z、相机坐标和长度输出；不要把它删掉或换成顶部 Z。

| 输出字段 | 含义 |
|---|---|
| `objects[*].status` | 只将 `ok` 视为有效检测 |
| `objects[*].corners_px` | 顶部四边形四角 |
| `objects[*].center_px` | 最终顶部像素中心 |
| `objects[*].angle_deg` | 从图像 +X 起逆时针为正的长轴角，按 180° 对称表示 |
| `center_model_*`、`fusion_*` | 模型预测和融合诊断信息 |
| `pts_top_face` | 转换后按抓取顺序排列的 `[u,v,angle]` |
| `list_pts_base` | `[映射X, 映射Y, 旧Z, 旧长度]` |
| `angle_deg` | 与 `list_pts_base` 对应的图像方向角数组，归一到约 ±90° |

**`list_pts_base[i][2]` 仍是旧转换结果，不是当前顶部平面 Z，也不是应用固定下抓 Z。** 当前映射只覆盖 X、Y，不能看到字段名有 `base` 就把整组 XYZ 当作新平面求交结果。

转换层按工件中心到图像中心的距离排序，`top_capture=1` 只保留排序后的第一个工件。`objects` 保留粗检测顺序，不能直接按相同索引与 `list_pts_base` 配对；需要使用像素中心建立对应关系。

`top_face_detect` 是内部纯图像入口，不负责去畸变，也不等于完整 `AI_det` 请求链。`tests/board_top_face_check.py` 调用的是完整库初始化且包含现场路径；WSL 上不要把它当成开箱即用的无硬件测试。

## 5 图像和内参必须一致

当前模型对应 1280×1024、完整画面、生产 `alpha=1` 去畸变坐标。相机内的水平和垂直翻转已经设定，算法不要再额外翻转。改变相机翻转、裁剪、缩放或去畸变输出内参，都需要重新核对像素合同。

`calibration/camera_reference.json` 中用于拟合的实际图像内参是：

```text
697.5631103515625   0                  615.7079467773438
0                   698.2827758789062  530.385498046875
0                   0                  1
```

原始 K、D 在 `param/eye_to_hand_result.yaml`。生产流程先将原 K 放入 float 相机字段，再计算 `getOptimalNewCameraMatrix(..., alpha=1)`，最后把新 K 写回 float 字段。重现时保留这个顺序，避免使用原始 K 直接解释已经去畸变的检测像素。

生产 `det.cpp` 在采集图保存为 `src_img/NNNNN.jpg` 之前已经执行 `undistort_frame`，前提是去畸变配置和参数有效。仓库的 `tests/data/00214.jpg`、`00218.jpg` 是对应保存图快照。处理这些样本时，先核对这一链路，不能不加判断地再去畸变一次。原始传感器图则应先按原 K、D 和生产输出 K 校正。

注意三种单位：手眼文件 `T_base_camera` 平移为米；映射模型、测点和 TCP 为毫米；图像坐标为像素。旋转矩阵不乘 1000。

## 6 当前映射模型怎样得到

当前顶面采用用户指定的约束：只随 Base X 倾斜，Base Y 坡度为零。

```text
Z_top = 0.0024996875390576176 * X + 251.07249093863268  (mm)
```

| 图片与工件 | u | v | 参考 X | 参考 Y | 顶部 Z 来源 |
|---|---:|---:|---:|---:|---|
| 00214 左侧 | 134.156459 | 458.229674 | -429.05 | 1165.50 | 实测 250 mm |
| 00214 右侧 | 1126.070812 | 526.619237 | 371.05 | 1152.92 | 实测 252 mm |
| 00218 下方 | 783.168974 | 586.010897 | 100.74 | 1085.36 | 由同一平面推算，未实测 |

当前三点等权最小二乘拟合的参数为：等效相机 `Cx,Cy,Cz` 和绕 Base Z 的一个偏角，共四个参数。手眼矩阵的其余倾斜约束保留；最后把射线与固定顶面求交编码成 `homography`。这不是用三点估计自由的八参数单应矩阵。

当前等效相机中心约为 `[-49.0816, 1138.2242, 813.6913]` mm，水平偏角修正约 `-0.96829°`。原手眼文件的相机 Z 为 `615.87` mm，两者差约 `197.82` mm。这些拟合参数不能直接当作物理相机重新测量值，它们可能吸收了定位、标定或 TCP 对应关系误差。

拟合误差为左侧 `1.713`、右侧 `3.206`、下方 `4.844` mm，RMS `3.497` mm。这些都是训练点误差。原来两点拟合的零误差也不代表实际精度。

当前生效模型 SHA256：

```text
9819a815819ba8d276493559523b7a25ac141d74e473969a9643d862dec656cc
```

拟合脚本中，前两点来自 `calibration/two_point_model.json`，第三点目前写在 `calibration/fit_top_plane.py` 的 `uv`、`xy` 组装处。新增测点时先确认图片与工件对应，再一起进入最小二乘；如果更换平面，要同步更新平面来源，不能只替换两个 JSON 系数。

候选模型只在独立测点通过后替换到 `config/top_face/tcp_xy_mapping.json`。算法会缓存模型，替换文件后需要调用 `top_face_close` 并重新初始化，或者重启宿主应用。背景图可自动重新加载，不代表坐标模型也会自动重新加载。

## 7 工件高度和机器人姿态

当前 `homography` 已直接对应本批工件顶部，不再给 XY 加工件高度。改变输入的 `workpiece_height_mm` 不会改变这条映射的 XY；旧的 `calibrated_camera_ray_plane` 分支仍保留高度输入能力，不能把两种模式混用。

宿主应用仍会把高度写入寄存器 3；不能因当前 XY 不使用高度，就未经核对把现场高度清零。更换工件厚度后，固定顶面不再适用，应重新标定顶部模型，或明确改用高度感知的射线平面模型。

最后确认的上层应用寄存器合同如下，应用代码不在本仓库：

| 浮点寄存器 | 应用写入内容 |
|---|---|
| 0 | 算法 X 加 `diff_x` |
| 1 | 算法 Y |
| 2 | 固定 `tcp_z`；未配置时有旧 Z 回退路径 |
| 3 | 请求对应的工件高度 |
| 4 | 应用计算的绝对欧拉 RZ，角度制 |

旧应用配置 `plusDiff=0` 时，按下式计算寄存器 4：

```text
delta = wrap180(image_angle - zeroAngle)
RZ = wrap360(zeroEuler - delta)     # plusDiff=0
```

当前算法已将图像角从顺时针为正改为逆时针为正，内部几何搜索与最终像素中心不变。为了在仅改变角度约定时保留原有机器人输出，应同时做以下迁移，而不是单独改零位角：

```text
zeroAngle_new = -zeroAngle_old
plusDiff_new = 1 - plusDiff_old
zeroEuler_new = zeroEuler_old
```

因此本例从 `zeroAngle=-88.74, plusDiff=0` 迁移为 `zeroAngle=88.74, plusDiff=1`，`zeroEuler=135` 不变。此关系适用于原来同一条零位差角公式，并沿用长方形的 180° 对称约定；恰好 ±90° 的边界需单独验收。本次源码同步只更新仓库示例，没有自动修改现场应用参数或部署二进制。正式换库前，要核对当前机器人脚本和应用版本后一起验证。

算法输出的图像角、绝对欧拉 RZ、轴角不是同一种量。算法没有估计完整 RX、RY、RZ；也没有用新 H 对工件方向做 Base 变换。以后修改方向时，优先把长轴两点通过相同的几何映射转换到 Base，再处理工具固定偏角。

当前机器人脚本的实际消费方式尚未独立核对。旧截图中的脚本存在“寄存器 4 加到示教欧拉 Z，再转轴角”的流程，如果现场仍如此，就可能与应用写绝对 RZ 的合同冲突。先读取当前脚本和实际到位角，再决定改哪一层；不要在 Euler/轴角之间猜测转换，也不要重复叠加示教角。

## 8 已知问题和后续顺序

**先统一测量点，再补数据。** 左右两组被描述为顶部中心，第三组被描述为抓取 TCP。需要确认都对应图像对角线中心这个物理点，并使用同一 Base、同一已标定 TCP。工具接触点与 TCP 有偏移时，应独立处理工具几何，不能混入像素映射后继续拟合。

**X 补偿暂作为现场单独设置。** `diff_x=7.5` 在应用端叠加。按原三组参考数据回算，模型加这个补偿后的误差约为 `8.59、10.31、5.03` mm，RMS 约 `8.28` mm。如果现场抓取反而更好，应核对测量点、TCP 和姿态，而不是直接据此删除补偿。

**平面和覆盖范围还没充分验证。** 补采覆盖左、中、右、前、后的 6～9 个顶部 XYZ，并另留独立点验证。第三点的推算 Z 不能证明 Base Y 坡度为零。增加高阶多项式前，先验证对应点、图像坐标合同及真实外参。

**图像定位和模型误差分开看。** 先画四边形、对角线中心、NN 中心和长轴，确认选中了顶部而不是侧面或底部台阶。重点记录 `fusion_selection_ambiguous`、`fusion_selection_margin`、边缘支持率和 NN 裁剪间差异，再比较 Base XY；NN 中心与几何中心不同，不能直接判定 NN 更准。

**背景与曝光要成套。** `exposure outside supported range` 来自背景光度匹配范围检查，不等于 MVS 硬件曝光参数越界。现场曾通过重新采背景恢复。更换背景、曝光组、增益或环境光后，先核对背景与检测图是否处于相同采集、融合、去畸变条件，避免直接放宽阈值。

**没有目标和旧坐标是不同问题。** 图像失败会返回相应状态；旧坐标重复发布还涉及宿主的拍照握手、队列和寄存器完成标志。应用修复不在本仓库，重现此类问题要同时保留图片编号、请求标识、检测 JSON 和寄存器写入日志。

## 9 X3 交叉编译和部署

WSL 准备目标 sysroot 后再构建。sysroot 中必须有目标架构的 OpenCV、Horizon DNN/BPU 和 MVS 头文件及库，保持其中的符号链接；不要从 WSL 的 `/usr/lib/x86_64-linux-gnu` 混入宿主库。

```bash
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
export RDK_SYSROOT="$HOME/rdk/rdk_x3_sysroot"

cmake -S . -B build-x3 \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/toolchain_rdk_x3.cmake" \
  -DRDK_SYSROOT="$RDK_SYSROOT" \
  -DUSE_RDK=ON -DuseMVS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-x3 --target ai-det -j1

file build-x3/libai-det.so
aarch64-linux-gnu-readelf -d build-x3/libai-det.so
```

应得到 AArch64 共享库，运行时需能找到 `/opt/MVS/lib/aarch64` 等目标库路径。不要在 x86 WSL 中直接加载 AArch64 `.so`。使用 `-j1`；90 曾发生内存耗尽并被系统杀死编译进程。

已有正式环境：

- 90 算法工程：`/home/nvidia/work/feilun/det-xialiao-MVS-feilun-nano`，分支 `feilun`。
- 90 工具链：`/home/nvidia/work/rdk/toolchain_rdk_x3.cmake`。
- 43 发布目录：`/home/sunrise/Desktop/ai-det-feilun/`。
- 43 算法库：发布目录下的 `libai-det.so`。
- 43 映射配置：`config/top_face/tcp_xy_mapping.json`。
- 43 应用配置：`robot_config.json`，与算法模型是不同文件。
- 43 历史备份：发布目录下的 `.codex-backups/`；启动检查记录在 `.codex-staging/`。

现场代号 90、43 的连接地址、SSH 密钥和口令单独配置，不放入仓库。网络不稳定时，先确认目标主机身份和连通性，不能因为连接失败就覆盖其他设备上的同名目录。

发布时记录库和模型 SHA256，备份旧库、模型和应用配置，停止宿主后替换文件，再重启。只改模型也需要重新初始化算法。确认进程启动后，还要验证真实检测 JSON、当前图片编号、实际写出的坐标和机械臂到位结果。进程存在不等于模型已通过现场验证。

## 10 下一位开发者从这里开始

1. 读本文、README、`top_face.cpp` 的候选选择和映射分支、`det.cpp` 的 `det_to_pts_top_face`。
2. 在 WSL 跑 CPU 检查和 Python 拟合，确认基线结果可复现。
3. 绘制当前样本顶部四角、对角线中心及模型预测，先验证像素定位。
4. 确认 TCP 对应点和当前机器人角度合同，再补独立测点。
5. 逐项修改，保留输入输出和旧模式兼容测试，避免同时改中心定义、映射、补偿和角度。
6. CPU 通过后交叉编译，再到 X3 检查 BPU、采集、延时和实际抓取。

交接时已通过四组 CPU 离线检查和 Python 矩阵复现。仓库导入后未重新进行整库 X3 构建；以上 WSL 安装命令未在独立 Windows WSL 实例中执行。现有模型仍需要独立位置与不同转角的实测验收。
