# 卫星图像航空辅助分析

用于课程展示的 Streamlit + Ultralytics YOLO 多模型项目。当前版本已包含四个最小合成数据占位模型，可完整演示上传、Auto 分类路由、检测/分割、图像对比与文字建议；占位模型不能用于评价真实识别精度。

## 已实现功能

- `YOLOv8n 通用检测`：展示检测前后图像和检测物体清单。
- `Auto 智能路由`：先把图像分为云场景、机场概览、跑道细节或其它，再调用对应业务模型。
- `云目标检测`：积雨云、层积云、呈白色涡旋状的台风云系，并生成气象说明和航行建议。
- `机场环境检测`：机场、大型建筑群和施工区域，并生成风险提示和维护建议；不直接下“违建”法律结论。
- `跑道状态分割`：跑道、积雪和疑似积水，并生成清理与维护建议。
- 图片来源：手动上传、内置合成样例、Supabase 云端收件箱。
- MMSSTV 文件夹监听：本地脚本发现新图片后上传到 Supabase，远程网页读取队列。

> 本系统仅用于课程项目展示，不能替代正式气象、适航、机场检查或航空运行结论。

## 直接运行

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

四个占位权重已经放在以下位置：

```text
models/scene_classifier/best.pt
models/cloud_detector/best.pt
models/airport_detector/best.pt
models/runway_segmenter/best.pt
```

`yolov8n.pt` 也已随部署包提供，因此演示不依赖启动时下载模型。首页选择“内置样例”即可在没有 Supabase 配置时完整演示。

## 占位数据和模型

生成合成占位数据：

```powershell
.\.venv\Scripts\python.exe scripts\data\build_placeholder_dataset.py
```

重新训练三个检测/分割占位模型和分类占位模型：

```powershell
.\.venv\Scripts\python.exe training\train_placeholder_models.py
.\.venv\Scripts\python.exe training\train_placeholder_classifier.py
```

合成数据上的结果只证明程序链路可运行，不代表对真实卫星 RGB 或 MMSSTV 图像的泛化能力。真实数据准备、LabelImg/Labelme 标注和验收流程见 [datasets/README.md](datasets/README.md)、[docs/DATASET_WORKFLOW.md](docs/DATASET_WORKFLOW.md) 与 [docs/ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md)。数据采集工具的额外依赖位于 `requirements-data.txt`。

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

Community Cloud 的入口文件为 `streamlit_app.py`，建议 Python 3.12。详细步骤见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。Cloud 只负责推理，不用于训练；应用一次处理一张图片，并只缓存一个模型以控制内存。

## 许可提醒

Ultralytics PyPI 包采用 AGPL-3.0-or-later。公开部署前应保留依赖来源与相应许可说明。真实数据集还必须逐项记录许可和来源，项目已在 `datasets/manifests/` 中预留清单。
