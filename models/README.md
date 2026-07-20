# 模型权重目录

已训练并验证的权重放置为：

- `scene_classifier/best.pt`
- `cloud_detector/best.pt`
- `airport_detector/best.pt`
- `runway_segmenter/best.pt`

当前四个权重均低于 GitHub 单文件限制，随部署仓库提供给 Streamlit Community Cloud。测试指标记录在 `training_metrics.json`；更新模型时应同步记录训练数据版本、Ultralytics 版本和验证指标。
