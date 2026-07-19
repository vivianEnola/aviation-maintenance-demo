# Streamlit Community Cloud 部署

## 最小部署

1. 将整个项目推送到 GitHub 仓库，保留 `models/**/best.pt`、`yolov8n.pt`、`datasets/placeholder/`、`requirements.txt` 和 `.streamlit/config.toml`。
2. 登录 Streamlit Community Cloud，选择该仓库与 `main` 分支。
3. Main file path 填写 `streamlit_app.py`，Python 版本选择 3.12。
4. 不配置 Secrets 也可使用“手动上传”和“内置样例”。
5. 部署完成后先选择“内置样例”，分别检查 Auto、云检测、机场环境检测和跑道分割模式。

首次启动和首次切换模型会较慢，这是 Cloud 从磁盘加载 CPU 模型造成的；后续同一模型推理会复用缓存。

## 可选：Supabase 云端收件箱

先在 Supabase 执行 `supabase/schema.sql`，再把以下内容加入 Community Cloud 的 Secrets：

```toml
[supabase]
url = "https://YOUR_PROJECT.supabase.co"
key = "YOUR_SERVICE_ROLE_KEY"
bucket = "mmsstv-images"
table = "image_queue"
```

不要把真实密钥写入仓库。完成后，按 `local_uploader/README.md` 在接收 MMSSTV 图片的 Windows 电脑上启动文件夹监听器。

## 更新真实模型

完成真实数据标注和训练后，只需替换四个 `models/.../best.pt` 权重并重新推送。若权重超过 GitHub 单文件限制，应改用 Git LFS 或私有对象存储，并在应用启动时下载。
