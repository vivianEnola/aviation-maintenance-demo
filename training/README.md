# 模型训练

训练建议在 Google Colab 或带 NVIDIA GPU 的环境执行，本机当前 PyTorch 为 CPU 版本。

示例：

```powershell
.\.venv\Scripts\python.exe training\train_model.py cloud_detector --device 0 --epochs 100
```

训练输出默认保存在 `runs/train/<model_id>`。检查验证指标、混淆矩阵和错误样本后，再使用 `--promote` 将 `best.pt` 发布到应用读取的位置：

```powershell
.\.venv\Scripts\python.exe training\train_model.py cloud_detector --device 0 --epochs 100 --promote
```

不要仅因训练完成就发布权重。至少检查：

- 分类：Macro F1、混淆矩阵、低置信度拒识；
- 检测：逐类 Precision、Recall、mAP50-95；
- 分割：逐类 IoU、Dice、积雪/积水误检样本；
- Auto：最终路由准确率。
