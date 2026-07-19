# 数据集执行方案

## 阶段 1：清晰 RGB 基线

先建立在清晰 RGB 条件下可用的模型，不把 MMSSTV 信号退化与基础识别能力混为一个问题。

目标数量：

| 任务 | 第一版 | 目标版 |
|---|---:|---:|
| 场景分类 | 每类 200 张 | 每类 400–600 张 |
| 云检测 | 400–500 张 | 900–1,500 张 |
| 机场环境检测 | 300–400 张 | 800–1,200 张 |
| 跑道分割 | 200–300 张 | 700–1,000 张 |

机场和跑道图片优先通过 OurAirports 坐标、Planetary Computer Sentinel-2 RGB 和 OSM runway 几何建立。DOTA v2 与 SpaceNet 只作为机场/建筑初始样本来源，不把建筑轮廓解释为高度。

云型和跑道积雪/积水缺少完全匹配的公开框/掩膜标签，需要采取“少量人工种子标签 → 第一版模型 → 预标注 → 逐张复核”的迭代方式。

## 阶段 2：标签审核

所有自动生成标签状态均为 `needs_review`。只有完成以下检查后才可进入训练集：

- 删除误检；
- 补充漏检；
- 修正类别和边界；
- 确认负样本确实无目标；
- 将审核状态改为 `approved`。

LabelImg 用于两个检测任务，Labelme Polygon 用于跑道分割。类别顺序以 `data.yaml` 和 `labeling/*.txt` 为准，标注后不得更改顺序。

## 阶段 3：分组切分与审计

- 训练/验证/测试比例约 70%/15%/15%；
- 按机场、来源图、气象事件或接收批次分组；
- 同一原图的相邻裁片不能跨集合；
- 执行 `audit_yolo_dataset.py`，修复跨集合重复、非法坐标和类别错误；
- 测试集在模型定型前保持封闭。

清单至少包含 `image,group_id,review_status` 三列。审核完成后执行：

```powershell
.\.venv\Scripts\python.exe scripts\data\split_dataset.py yolo `
  --images datasets\incoming\cloud_detection\images `
  --labels datasets\prelabels\cloud_detection `
  --output datasets\cloud_detection `
  --manifest datasets\manifests\cloud_review.csv `
  --require-approved
```

分类任务把待切分图片按 `输入目录/类别名/图片` 放置，然后将 `task` 改为
`classification`。脚本只复制文件，不移动或删除原始数据，并输出
`split_manifest.csv` 供审计追踪。

## 阶段 4：SSTV 域适配

清晰图模型通过后，再加入模拟 SSTV 退化样本。模拟样本必须与其原图处于同一数据切分，防止泄漏。获取真实 MMSSTV 图片后，保留独立真实测试集，并用少量真实样本微调。

## 数据来源原则

- 只使用允许研究、课程展示和再分发/派生处理的数据；
- 不自动抓取 Google Earth 等许可不明确的网页图像；
- 每个文件记录来源 URL、许可、获取日期、任务和审核状态；
- 来源清单不完整的数据不得进入最终训练集。
