# MMSSTV 本地文件夹上传器

1. 将 `uploader.example.toml` 复制为 `uploader.toml`，填写 MMSSTV 输出目录和设备 ID。
2. 在 PowerShell 中设置密钥：

   ```powershell
   $env:SUPABASE_URL="https://YOUR_PROJECT.supabase.co"
   $env:SUPABASE_SERVICE_ROLE_KEY="YOUR_SERVICE_ROLE_KEY"
   ```

3. 先执行无上传测试：

   ```powershell
   .\.venv\Scripts\python.exe local_uploader\watch_folder.py --dry-run --once
   ```

4. 正式监听：

   ```powershell
   .\.venv\Scripts\python.exe local_uploader\watch_folder.py
   ```

上传器只读取图片，不移动或删除 MMSSTV 原文件。每张图片按 SHA-256 去重，并在确认文件大小和修改时间稳定后上传。

`SUPABASE_SERVICE_ROLE_KEY` 权限很高，只能放在本机环境变量和 Streamlit Secrets 中，严禁提交到 Git。
