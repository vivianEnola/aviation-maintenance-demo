# MMSSTV 本地文件夹上传器

远程网页不能直接读取访问者电脑上的文件路径。本工具在 MMSSTV 所在电脑持续扫描目录，并把新图片上传到网页使用的 Supabase 收件箱。

1. 在 PowerShell 中设置密钥：

   ```powershell
   $env:SUPABASE_URL="......"
   $env:SUPABASE_SERVICE_ROLE_KEY="......"
   ```

2. 推荐直接运行交互式启动器，并按提示输入图片目录和设备 ID：

   ```powershell
   .\start_folder_sync.bat
   ```

3. 如需先执行无上传测试，可将路径直接传给监听器：

   ```powershell
   .\.venv\Scripts\python.exe local_uploader\watch_folder.py --watch-folder "D:\MMSSTV\images" --device-id "mmsstv-windows-01" --dry-run --once
   ```

4. 也可不使用启动器，直接正式监听：

   ```powershell
   .\.venv\Scripts\python.exe local_uploader\watch_folder.py --watch-folder "D:\MMSSTV\images" --device-id "mmsstv-windows-01"
   ```

上传器只读取图片，不移动或删除 MMSSTV 原文件。每张图片按 SHA-256 去重，并在确认文件大小和修改时间稳定后上传。

`SUPABASE_SERVICE_ROLE_KEY` 权限很高，只能放在本机环境变量和 Streamlit Secrets 中，严禁提交到 Git。

完整的 Supabase、Streamlit Secrets 和分发运行步骤见 `docs/OPERATIONS.md`。
