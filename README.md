# 卫星图像航空辅助分析

基于 Streamlit 与 Ultralytics YOLO 的多模型图像分析系统，面向普通 RGB 卫星图像和 MMSSTV 接收图像，提供场景自动路由、云系检测、机场周边环境检测、跑道状态分割及专业化辅助建议。

## 已实现功能

- `YOLOv8n 通用检测`：展示检测前后图像和检测物体清单。
- `Auto 智能路由`：先把图像分为云场景、机场概览、跑道细节或其它，再调用对应业务模型。
- `云目标检测`：识别积雨云、层积云和呈白色涡旋状的台风云系，并生成气象说明和航行建议。
- `机场环境检测`：识别机场、大型建筑群和施工区域，生成净空风险线索和维护建议，不直接作出违建法律结论。
- `跑道状态分割`：分割跑道、积雪和疑似积水区域，并生成清理和跑道维护建议。
- 图片来源：手动上传、内置分析样例、Supabase 云端收件箱。
- MMSSTV 文件夹监听：本地脚本发现新图片后上传到 Supabase，远程网页持续读取待分析队列。

> 系统输出用于图像智能筛查与辅助研判。航班运行、机场开放、适航和气象处置必须由具备资质的人员结合现场检查及正式业务资料决定。

## 直接运行

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

部署包包含四个业务模型和 YOLOv8n 通用检测权重：

```text
models/scene_classifier/best.pt
models/cloud_detector/best.pt
models/airport_detector/best.pt
models/runway_segmenter/best.pt
yolov8n.pt
```

因此应用启动时不依赖在线下载模型。未配置 Supabase 时，仍可使用手动上传和内置分析样例。

## 模型验证结果

四个业务模型基于整理后的项目数据集训练，并在独立测试划分上验证：

| 模型 | 主要测试指标 |
| --- | --- |
| 场景分类器 | Top-1 accuracy 98.44% |
| 云系检测器 | Precision 81.94%，Recall 50.05%，mAP50 61.65% |
| 机场环境检测器 | Precision 46.28%，Recall 39.44%，mAP50 42.35% |
| 跑道状态分割器 | Mask precision 76.76%，Mask recall 40.00%，Mask mAP50 44.59% |

完整指标保存在 `models/training_metrics.json`。这些数值反映当前测试划分表现；由于数据规模和来源有限，部署到新卫星、不同缩放倍率或强压缩 MMSSTV 图像时仍需持续复核和补充数据。

数据准备、LabelImg/Labelme 标注和验收流程见 [datasets/README.md](datasets/README.md)、[docs/DATASET_WORKFLOW.md](docs/DATASET_WORKFLOW.md) 与 [docs/ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md)。数据采集工具的额外依赖位于 `requirements-data.txt`。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\smoke_deployment.py
```

## Supabase 与 MMSSTV

Supabase 是可选功能，不影响内置样例和手动上传：

1. 在 Supabase SQL Editor 执行 `supabase/schema.sql`。
2. 把 `.streamlit/secrets.example.toml` 的字段填写到 Community Cloud 的 Secrets。
3. 按 `local_uploader/README.md` 配置 Windows 本地监听器和 MMSSTV 图片输出文件夹。

`service_role` 密钥只能保存在本地环境变量或 Streamlit Secrets，严禁提交到 Git。

## 部署

Community Cloud 的入口文件为 `streamlit_app.py`，建议 Python 3.12。详细步骤见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。云端只负责推理，不用于训练；应用一次处理一张图片，并只缓存一个业务模型以控制内存。

## 许可提醒

Ultralytics PyPI 包采用 AGPL-3.0-or-later。公开部署前应保留依赖来源与相应许可说明。数据集还必须逐项记录许可和来源，项目已在 `datasets/manifests/` 中预留清单。
