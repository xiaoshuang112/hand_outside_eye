# 眼在手外工件检测与抓取坐标算法库

完整 C++ 视觉检测算法库，来源于正式工程 `det-xialiao-MVS-feilun-nano` 的 `feilun` 分支。包含顶部四边形定位、几何与中心模型融合、对角线交点、像素到 Base XY 映射、MVS 相机接口，以及 RDK X3 / TensorRT 后端源码。来源提交和模型校验值见 [PROVENANCE.json](PROVENANCE.json)。

后续开发先读 [WSL 算法库开发交接](docs/HANDOFF_WSL.md)，其中记录环境搭建、代码入口、坐标与寄存器约定、已知问题和 X3 发布流程。

## 当前检测流程

1. 图像采集、曝光融合和去畸变。
2. 与背景图比较，提取工件候选区域。
3. 搜索顶部矩形，拟合四条边；中心模型参与候选选择。
4. 以顶部四边形的对角线交点作为最终像素中心。
5. 加载 `config/top_face/tcp_xy_mapping.json`，转换为机器人 Base XY。

保留已有输入输出接口：`AI_det_init`、`AI_det`、`AI_det_close`，声明见 `det-src/det.h`。`top_face` 返回的 `angle_deg` 是图像方向角，不是机器人轴角或完整欧拉姿态；机器人姿态和寄存器发送由上层应用处理。

当前输出角度以图像 +X 为基准，**逆时针为正，按 180° 对称表示**。这是从 90 合入的 `563e25a` 改动，旧导入基线 `1cc6675` 使用顺时针为正。已有零位角与差角加减配置必须按同一约定迁移，详见交接文档，不能只替换库而沿用旧角度参数。

## 当前坐标模型

默认配置是三点等权最小二乘得到的固定顶部平面模型，最终编码为单应矩阵 `homography`：

```text
Z_top = 0.0024996875390576176 * BaseX + 251.07249093863268  (mm)
```

假设顶部只随 Base X 倾斜，Base Y 方向坡度为零。相机内参与手眼倾斜方向作为约束，重拟合等效相机位置及 Base Z 方向偏角。这里并非用三个点拟合不受约束的八自由度单应矩阵。

像素合同：1280×1024、完整画面、与生产 `alpha=1` 去畸变结果相同的内参；不能直接混用原始畸变图像、裁剪或缩放后的坐标。相机中的水平、垂直翻转已在现场配置，算法不再额外翻转。

固定平面与相机条件下，射线和平面求交可合成为该矩阵。**此模型的 XY 已对应工件顶部，不再额外加工件高度。** 更换工件厚度、移动相机或调整台面后，需要重新标定或改用对应的高度感知模型。

| 参考点 | 图片 | u | v | 参考 X | 参考 Y | XY 拟合误差 |
|---|---|---:|---:|---:|---:|---:|
| 左侧 | 00214.jpg | 134.156459 | 458.229674 | -429.05 | 1165.50 | 1.713 mm |
| 右侧 | 00214.jpg | 1126.070812 | 526.619237 | 371.05 | 1152.92 | 3.206 mm |
| 下方 | 00218.jpg | 783.168974 | 586.010897 | 100.74 | 1085.36 | 4.844 mm |

左、右顶部实测 Z 分别为 250、252 mm；第三点 Z 从上述平面推算，未实测。训练 RMS 为 3.497 mm，**不代表独立位置的抓取精度**。拟合报告位于 `calibration/fit-report.json`。

库的此分支只输出映射后的 XY；顶部平面 Z 是几何参考，不会自动替换上层应用的下抓 Z。`config/app_mapping.example.json` 是适配当前逆时针角度的应用示例，不是现场配置文件的实时副本；该文件不由算法库加载。`diff_x=7.5` 是后加的 Base X 补偿，不属于以上拟合误差。

## 模型与配置

- `config/top_face/algorithm.json`：现场 X3 配置，中心模型使用 `center_only.bin`。
- `config/top_face/center_only.bin` / `center_only.onnx`：X3 BPU 模型及 CPU ONNX 模型。
- `config/top_face/background.bmp`：现场背景快照；实际使用前按现场光照与相机参数重新采集。
- `config/top_face/tcp_xy_mapping.json`：当前三点固定顶部平面映射。
- `param/eye_to_hand_result.yaml`：原始手眼、相机内参及畸变参数；其中平移单位为米，映射模型坐标单位为毫米。

CPU 图像检测时应在自己的配置副本中把 `center_model` 改为 `center_only.onnx`；X3 的 `.bin` 不能交给 OpenCV DNN 读取。

`tests/data/00214.jpg`、`00218.jpg` 为对应现场融合图快照。表中的像素来自生产检测使用的去畸变画面；直接对这些采集图运行顶部检测前，应先核对并复现生产去畸变步骤，不能把原图像素与表中坐标混用。

## X3 交叉编译

外部依赖：AArch64 GCC、目标 sysroot、OpenCV 4.10 或更新、Horizon DNN/BPU SDK、MVS SDK。spdlog 必需头文件及其许可证已包含；厂商 SDK 和 sysroot 不在仓库中。

```bash
cmake -S . -B build-x3 \
  -DCMAKE_TOOLCHAIN_FILE=toolchain_rdk_x3.cmake \
  -DRDK_SYSROOT=/path/to/rdk_x3_sysroot \
  -DUSE_RDK=ON -DuseMVS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-x3 --target ai-det -j1
```

正式开发环境也可使用既有 `/home/nvidia/work/rdk/toolchain_rdk_x3.cmake`。使用 `-j1` 限制内存占用。算法运行还需要宿主应用的相机配置、SDK 运行库及设备授权；本仓库未包含宿主 Qt 应用、连接口令或设备授权文件。历史机器人自动标定辅助源码另需相应机器人 SDK，不属于默认 `ai-det` 构建目标。

## CPU 离线检查

需要 C++17 编译器、pkg-config、包含 DNN 模块的 OpenCV 4：

```bash
bash tests/run_cpu_checks.sh
```

检查融合逻辑、模型配置校验、对角线中心、旧模型兼容性、当前三点矩阵与 Python 数值一致性，以及固定顶面模型不会重复加工件高度；不连接相机、机械臂或 BPU。包含硬件集成检查的原脚本 `tests/run_top_face_offline_checks.sh` 另需 RDK 头文件及匹配的应用运行目录。

## 复现三点拟合

```bash
python3 -m venv .venv
.venv/bin/pip install -r calibration/requirements.txt
.venv/bin/python calibration/fit_top_plane.py
```

候选结果写入 `calibration/output/`，不会覆盖当前生效的配置。检查误差与独立测试后，再替换 `config/top_face/tcp_xy_mapping.json`；模型在初始化时加载，替换后需重新初始化算法或重启宿主应用。

当前模型沿用 Base Y 零坡度和手眼倾斜方向的约束，只有三个训练点。正式验收需要覆盖前后左右的独立测点，并确认像素顶部中心与示教 TCP 对应同一物理点；工具偏移与方向转换不能靠增加模型阶数代替。
