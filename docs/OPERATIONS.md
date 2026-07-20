# 运行、文件夹同步与模型维护

## 在另一台 Windows 电脑运行

不建议直接复制现有 `.venv`。虚拟环境包含原电脑的绝对路径、Python 解释器和平台相关二进制文件，换电脑、换用户名或换 Python 安装位置后很容易失效，而且当前环境接近 2 GB。

推荐使用项目生成的精简分发包：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_distribution.ps1
```

把 `dist\AviationVisionApp.zip` 发给对方。对方按以下步骤操作：

1. 解压到不含特殊符号的目录，例如 `D:\AviationVisionApp`。
2. 安装 64 位 Python 3.12，并在安装界面勾选 `Add python.exe to PATH`。
3. 双击 `setup_windows.bat`。脚本会创建该电脑自己的 `.venv` 并安装依赖。
4. 双击 `run_app.bat`，浏览器访问 `http://localhost:8501`。

模型推理可以使用 CPU，不要求 NVIDIA GPU。第一次安装需要联网下载 Python 依赖；第一次启动及切换模型通常较慢。

如果对方完全不能安装 Python，不能可靠地直接使用你电脑中的 `.venv`。可继续使用已经部署的 Streamlit Cloud 链接；若要求完全离线且零 Python 安装，需要另行制作体积较大的 Windows 安装包，这不等同于复制虚拟环境。

## 监听本地图片目录并同步到远程网页

浏览器安全机制决定了远程网页不能直接读取访问者输入的 `D:\MMSSTV\images`。正确结构是：

```text
MMSSTV 图片目录 → 本地监听器 → Supabase 收件箱 → Streamlit Cloud
```

首次配置：

1. 在 Supabase SQL Editor 执行 `supabase/schema.sql`。
2. 在 Supabase Storage 创建私有 bucket `mmsstv-images`。
3. 在 Streamlit Community Cloud 的 App settings → Secrets 中填写：

   ```toml
   [supabase]
   url = "https://你的项目.supabase.co"
   key = "你的 service_role key"
   bucket = "mmsstv-images"
   table = "image_queue"
   ```

每次启动监听器：

1. 在接收 MMSSTV 图片的电脑打开 PowerShell，进入项目目录。
2. 仅在当前窗口设置密钥：

   ```powershell
   $env:SUPABASE_URL="https://你的项目.supabase.co"
   $env:SUPABASE_SERVICE_ROLE_KEY="你的 service_role key"
   .\start_folder_sync.bat
   ```

3. 按提示输入本地图片文件夹路径和设备 ID。
4. 保持该窗口运行。监听器每 5 秒检查一次，只上传已经写入完成的新图片，不移动或删除原文件，并按 SHA-256 去重。
5. 用户打开远程网页，选择“云端收件箱”；网页每 10 秒刷新一次，选择待处理图片后点击分析。

`service_role` 权限很高，不要把它写入脚本、截图、Git 仓库或发送给无关人员。若需要让很多普通用户上传，后续应改为低权限上传令牌或独立上传接口。

## 更新重新训练的权重

假设新的训练结果目录结构为：

```text
aviation_trained_models/
├── scene_classifier/best.pt
├── cloud_detector/best.pt
├── airport_detector/best.pt
├── runway_segmenter/best.pt
└── metrics.json
```

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts\update_model_weights.py "D:\训练输出\aviation_trained_models"
```

脚本会先验证四个模型的任务类型与类别顺序，全部通过后才覆盖 `models/.../best.pt`，并把旧权重备份到 `.model_backups`。随后运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\smoke_deployment.py
git add models
git commit -m "Update trained model weights"
git push origin main
```

Streamlit Community Cloud 会在 `main` 更新后自动重新部署。部署完成后至少用云、机场、跑道和 Auto 四类图片各检查一次。若重新训练时修改了类别名称或顺序，不要强行覆盖权重，还需要同步修改 `configs/models.yaml`、Auto 路由和建议规则。
