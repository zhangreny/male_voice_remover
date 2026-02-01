import subprocess
import time
import requests
import os

# 1. 启动 API
print("[SYSTEM] Starting API server on port 5001...")
api_proc = subprocess.Popen([r".venv\Scripts\python.exe", "api.py"])

# 2. 等待启动
time.sleep(10)

# 3. 发送测试请求
url = "http://127.0.0.1:5001/process"
video_path = r"C:\Users\zhangrenyu\Downloads\exported_video_1769875161866.mp4"
output_file = "api_test_result.wav"

print(f"[RUN] Sending video to AI processor...")
try:
    with open(video_path, "rb") as f:
        response = requests.post(url, files={"video": f}, timeout=300)

    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"[SUCCESS] Processed audio saved to: {os.path.abspath(output_file)}")
    else:
        print(f"[ERROR] API failed: {response.status_code}")
except Exception as e:
    print(f"[ERROR] Test failed: {e}")

# 4. 关闭 API
print("[SYSTEM] Shutting down API...")
api_proc.terminate()
