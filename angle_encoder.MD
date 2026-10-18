# AngleEncoder — xike 模型支持

## 概述

`AngleEncoder` 是一个独立的 C++ 类，封装了特征编码器模型的 TensorRT 推理以及基于 FFT 互相关（CorrFFTHead）的角度偏差计算。

- **模型文件**: `build/models/xike.m`（TensorRT 序列化引擎）
- **对应 ONNX**: `build/models/model_encoder.onnx`
- **输入**: 极坐标展开图像 `[1, 3, 128, 8192]`
- **输出**: 特征图 `[1, 64, 1, 8192]`

## 新增文件

| 文件 | 说明 |
|------|------|
| `include/angle_encoder.h` | 类声明 |
| `det-src/angle_encoder.cpp` | 完整实现（TensorRT 推理 + FFT 互相关） |

`CMakeLists.txt` 中已添加 `det-src/angle_encoder.cpp`。

## 管线

```
输入图像 (极坐标展开)
    │
    ▼
preprocess(): resize → BGR→RGB → 归一化 [0,1] → CHW 重排 → GPU
    │
    ▼
infer(): TensorRT enqueueV3
    │
    ▼
encode(): 返回特征向量 (64×8192 float)
    │
    ▼
compute_angle(): 模板特征 + 查询特征
    ├── L2 逐列归一化 (dim=1)
    ├── 逐行 DFT (OpenCV cv::dft)
    ├── 频域共轭相乘 → 跨通道求和
    ├── DC 分量置零
    ├── IDFT → 互相关 logits (8192,)
    ├── 三点抛物线插值 → 亚像素峰值
    └── shift_index × 360.0 / 8192 → 角度（度）
```

## 使用方式

```cpp
#include "angle_encoder.h"

// 1. 加载模型（支持 .m 和 .onnx）
AngleEncoder encoder("models/xike.m", logger);

// 2. 编码模板图像（已对准 0 度的基准图）
vector<float> tmpl_feat = encoder.encode(template_image);

// 3. 编码查询图像
vector<float> query_feat = encoder.encode(query_image);

// 4. 计算角度偏差，单位：度，范围 [0, 360)
float angle_deg = AngleEncoder::compute_angle(tmpl_feat, query_feat, 8192);
```

## 重要说明

- **极坐标展开由调用方负责**：`encode()` 接收的是已展开至极坐标的图像，只做 resize / 颜色转换 / 标准化
- **模板注册**：每个新零件需先拍摄一张基准图（0 度位），调用 `encode()` 保存模板特征，后续查询图与模板比对得到角度偏差
- **类独立于 YOLOv11**：`AngleEncoder` 不继承自 `YOLOv11`，是独立的推理类，可单独使用

## 对应 Python 参考

`build/models/validate.py` 中的 Python 管线与此 C++ 实现对应：

| Python (validate.py) | C++ (AngleEncoder) |
|---------------------|-------------------|
| `AngleRotationModel(cfg)` | `AngleEncoder(model_path, logger)` |
| `model.encode(base)` / `model.encode(query)` | `encoder.encode(image)` |
| `CorrFFTHead.forward(fx, fy)` | `compute_angle(tmpl, query, bins)` |
| `_subpixel_shift_index(logits)` | 抛物线插值（内嵌） |
| `shift_index_float_to_angle_deg(shift, bins)` | `peak_float * 360.0f / bins` |

## 构建

```bash
cd build && cmake .. && make -j$(nproc)
```

构建产物 `build/libai-det.so` 包含所有 `AngleEncoder` 符号。
