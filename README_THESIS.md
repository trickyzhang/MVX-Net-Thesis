# RGB–LiDAR 毕设适配（2026-10-26）

本目录在原项目之上增加 `thesis_adapter`，面向“融合检测 → 二维投影与距离关联 → 目标记录 → 结构化查询”。原论文、版权、许可证和算法实现保持可追溯；新增适配不代表已经提升检测精度。

## 已实现和验证范围

- 两个模型共用记录字段、坐标约定和 SQLite 查询接口，分别运行在各自的软件环境中。
- 单相机 KITTI 批量运行入口：每次运行只初始化一次模型，MVX-Net 的数据处理流水线只构造一次，避免逐帧重载整个 infos 文件。
- 投影时使用 `P2 @ R0_rect @ Tr_velo_to_cam`，近裁剪面穿越、相机后方和图像外框有显式状态。
- 距离分为 LiDAR 原点到预测框几何中心的欧氏距离、矫正相机参考坐标系 z 深度、预测框内实际点的径向距离中位数。三者不是同一量。
- JSONL、SQLite、源图像路径、帧号、观测编号和时间戳；同一 run/frame 重导入原子替换，不同模型运行互不覆盖。
- 类别、距离、画面位置、时间筛选与证据回溯；SQL 参数化。
- `torch.inference_mode()`、显式 CUDA 同步、预热排除、均值/中位数/P95、端到端吞吐与 CUDA 峰值显存。框内点统计按需开启。
- FP32 默认；`amp-fp16` 是待验证选项，未声称兼容所有稀疏点云算子，也未声称提速。

本环境只有 NumPy/Pillow，没有 PyTorch/CUDA、训练权重或完整 KITTI。CPU 几何、存储、查询及合成样例已测试；GPU 后端尚未执行，不提供 mAP、真实 FPS 或 Orin NX 性能结论。未改动网络结构，未训练、剪枝、量化或导出 TensorRT。

## CPU 快速验证

在仓库根目录运行：

```bash
python -m pip install -r thesis_configs/requirements-cpu.txt
python -m unittest discover -s thesis_tests -v
python -m thesis_adapter doctor
python -m thesis_adapter export \
  --predictions thesis_configs/predictions.synthetic.json \
  --calib thesis_configs/calib.synthetic.txt --run-id synthetic-test \
  --output thesis_outputs/synthetic/observation.json \
  --database thesis_outputs/synthetic/targets.sqlite
python -m thesis_adapter query \
  --database thesis_outputs/synthetic/targets.sqlite \
  --category Car --max-range 20 --side center
```

该样例为人工构造的几何输入，仅验证接口，不是模型预测。不要用它报告检测效果。

## 输入、输出及约定

复制 `thesis_configs/manifest.example.json` 后填写真实 KITTI 路径。清单必须是 JSON 数组，每项有唯一字符串 `frame_id`、`image`、`points`、`calib`；相对路径相对于清单所在目录。点云为 float32 XYZI `.bin`；标定为原始 KITTI `P2/R0_rect/Tr_velo_to_cam` 文件。不要使用演示目录中只有矩阵数字、没有字段名的 `.txt` 文件。

`timestamp` 是真实观测时间（秒）或 null；不能把随机图片编号、导入时间作为采集时间。`run_id` 每次实验明确指定，例如 `mvx-fp32-val-v1`。

| 字段 | 定义 |
|---|---|
| `box_3d_lidar` | MMDetection3D LiDAR 框 `[x,y,z_bottom,dx,dy,dz,yaw]`，米、弧度 |
| `center_lidar_m` | 几何中心，已经把底面中心 z 加上 dz/2 |
| `range_center_lidar_m` | 相对 LiDAR 原点的预测框中心欧氏距离 |
| `depth_camera_m` | 矫正参考相机坐标系 z，不是相机中心欧氏距离 |
| `surface_median_range_m` | 预测 3D 框内有效点到 LiDAR 原点距离的中位数；不足点数为 null |
| `bbox_2d` | 原始图像坐标 `[xmin,ymin,xmax,ymax]`，连续像素边界；投影无效为 null |
| `projection_status` | valid / near_plane_clipped / behind_camera / outside_image |
| `instance_id` | run/frame/detection_index 派生的观测标识，不是跨帧 tracking ID |

画面左、中、右定义为二维框中心横坐标位于图像三等分区域，不表示世界坐标方向。`--max-range` 查询预测框中心距离；`--category` 使用模型标准类别，能重复传入多个类别。没有实现自然语言解析或 Transformer/RAG；这些后续模块可读取同一数据库，但必须把数值条件交给结构化筛选。

运行输出为 `observations.jsonl`、`targets.sqlite`、`run.json`、`performance.json`。`run.json` 保存配置展开结果、配置/清单 SHA256、类别顺序和运行版本。推理计时包括模型自身的预处理、网络和后处理；frame_total 还包括标定读取、目标记录和存储，不应称为纯网络时延。

## MVX-Net GPU 入口

MVX-Net 使用此仓库的 MMDetection3D 实现，不是另一套未经核查的第三方复现。先按原项目 `docs/en/get_started.md` 构建环境、安装 CUDA 算子和本地包，再按 `configs/mvxnet/README.md` 下载与配置对应的完整训练检查点。不要把配置中的图像分支预训练权重当成完整融合模型。

```bash
python -m thesis_adapter run --backend mvxnet \
  --manifest /data/kitti-val-manifest.json \
  --config configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py \
  --checkpoint thesis_weights/mvxnet-kitti.pth \
  --run-id mvx-fp32-val-v1 --output thesis_outputs/mvx-fp32-val-v1
```

不传 `--checkpoint` 会报错，防止把随机初始化结果误当预训练预测。类别顺序从模型元数据读取，不假定 Car 是索引 0。

## LCF3D GPU 入口

在 LCF3D 副本使用相同命令模块；先按原 README 的版本构建独立环境，安装修改过的 `src/mmdetection3d`，并确认 `scipy/networkx/ultralytics` 等上游导入依赖完整。不要在同一个 Python 环境同时以 editable 方式安装两个 mmdet3d 副本。

`thesis_configs/lcf3d.kitti.single.json` 使用仓库内配置与 `${LCF3D_WEIGHTS}`，替换原示例作者机器绝对路径。下载上游权重后，将匹配文件放在该目录，并采用模板约定的文件名（或修改 JSON）：

- `faster_rcnn_kitti.pth`：KITTI Faster R-CNN。
- `pointpillars_kitti.pth`：对应 PointPillars 配置的完整权重。
- `frustum_single_kitti.pth`：单相机 Frustum Localizer。

```bash
export LCF3D_WEIGHTS=/data/weights/lcf3d
python -m thesis_adapter run --backend lcf3d \
  --manifest /data/kitti-val-manifest.json \
  --config thesis_configs/lcf3d.kitti.single.json \
  --run-id lcf-fp32-val-v1 --output thesis_outputs/lcf-fp32-val-v1
```

模板保留检测恢复模块；关闭它属于消融实验，必须单独报告。适配器只负责单相机推理，原仓库的双目、多相机和评测入口继续保留。

LCF3D 副本还修复二维分支不读取顶层 `score_thr_2d` 的问题（嵌套阈值优先），增加标量展开与合法性校验，并让原 demo 的标定张量服从 `-device`。

## 下一步真实实验

1. 先在 GPU 上用少量真实样本核验两种模型的类别、底面中心坐标、投影和点云叠加；本适配器未改变 LCF3D 内部候选框匹配算法。
2. 使用同一 KITTI 验证划分调用原模型评测入口，记录 3D AP/BEV AP；二维投影另与 2D 标注计算 IoU。
3. 距离误差用匹配 GT 框的同一定义中心距离计算。框内点中位数不是独立 GT，不能把预测框派生深度当真实测距精度。
4. 固定模型、清单、阈值与预热次数，对比 FP32/AMP FP16；若稀疏算子不兼容则保留 FP32。统计精度变化后才能判断优化收益。
5. Orin NX 需单独记录 JetPack/CUDA/TensorRT、功耗模式与端到端时延。此版本没有声称 TensorRT 导出成功。
6. V2U4Real 需要格式、类别、点云分布与标定适配，当前没有用 KITTI 坐标假设直接套用无人机数据。
