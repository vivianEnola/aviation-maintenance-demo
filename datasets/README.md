# 数据集占位目录

本目录保存四个业务模型的数据。当前仅包含目录占位、类别配置和来源清单模板，不包含训练图片或标签。

## 任务与类别

1. `classification`：路由分类
   - `cloud_scene`
   - `airport_overview`
   - `runway_detail`
   - `other`
2. `cloud_detection`：目标检测
   - `cumulonimbus`（积雨云）
   - `stratocumulus`（层积云）
   - `typhoon_vortex`（呈白色涡旋状的台风云系）
3. `airport_detection`：目标检测
   - `airport`
   - `large_building_cluster`（大型建筑或建筑群，不代表已知高度）
   - `construction_area`（可见施工区域）
4. `runway_segmentation`：分割
   - `runway`
   - `snow`
   - `standing_water`

## 目录规则

- 分类任务按 `train/类别名`、`val/类别名`、`test/类别名` 放置图片。
- 检测与分割任务按 Ultralytics YOLO 格式，将图片放在 `images/{train,val,test}`，同名标签放在 `labels/{train,val,test}`。
- 切分时按来源、地点或接收批次分组，禁止把同一原图的相邻裁片分散到不同集合。
- 所有图片都应登记到 `manifests/sources.csv`，记录来源、许可、原始或 MMSSTV 解码域及备注。
- 不要把训练图片和模型权重直接提交到普通 Git 历史；后续根据部署方式决定使用 Git LFS 或外部存储。

## 标签语义提醒

- `typhoon_vortex` 是“视觉上疑似白色涡旋状台风云系”，不是气象部门确认的台风结论。
- `large_building_cluster` 不表达建筑高度或违建性质，只用于提示人工复核机场周边环境。
- `standing_water` 只标注在原图中确实可见的积水区域；阴影、深色铺装和普通湿润表面不得作为确定积水标注。
