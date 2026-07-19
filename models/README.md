# 模型权重目录

模型文件不提交到普通 Git 历史。训练验证通过后，将权重放置为：

- `scene_classifier/best.pt`
- `cloud_detector/best.pt`
- `airport_detector/best.pt`
- `runway_segmenter/best.pt`

Community Cloud 部署时使用 Git LFS 或受控对象存储。每个权重应同时记录训练数据版本、Ultralytics 版本和验证指标。
