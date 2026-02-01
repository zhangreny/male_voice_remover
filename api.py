import os
import sys
import uuid
import torch
import librosa
import numpy as np
import soundfile as sf
import tempfile
import shutil
from pathlib import Path
from flask import Flask, request, send_file, jsonify
import imageio_ffmpeg as im_ffmpeg
from moviepy import VideoFileClip
from audio_separator.separator import Separator
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# --- 1. 环境与硬件配置 ---
ffmpeg_exe = im_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] = os.path.dirname(ffmpeg_exe) + os.pathsep + os.environ["PATH"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CACHE = Path(os.path.expanduser("~")) / ".male_voice_remover" / "models"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("D:/male_voice_remover/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- 2. 全局模型初始化 (在 API 启动前加载到内存) ---
print(f"[INIT] 正在预加载 AI 性别检测模型至 {DEVICE}...")
GENDER_MODEL_NAME = "prithivMLmods/Common-Voice-Gender-Detection"
gender_extractor = Wav2Vec2FeatureExtractor.from_pretrained(GENDER_MODEL_NAME)
gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(GENDER_MODEL_NAME).to(
    DEVICE
)
gender_model.eval()

app = Flask(__name__)


@app.route("/process", methods=["POST"])
def process():
    """
    接收视频文件，执行完整的 AI 分离和过滤管线，返回 WAV。
    """
    if "video" not in request.files:
        return jsonify({"error": "Missing 'video' file in request"}), 400

    task_id = str(uuid.uuid4())[:8]
    temp_dir = Path(tempfile.mkdtemp())
    video_file = request.files["video"]
    input_video_path = temp_dir / f"input_{task_id}.mp4"

    try:
        # 1. 保存上传文件
        print(f"[{task_id}] 接收到请求，保存视频...")
        video_file.save(str(input_video_path))

        # 2. 提取原始音频
        print(f"[{task_id}] 提取视频音频轨道...")
        raw_audio = temp_dir / "raw.wav"
        with VideoFileClip(str(input_video_path)) as video:
            if video.audio is None:
                return jsonify({"error": "Video has no audio track"}), 400
            video.audio.write_audiofile(str(raw_audio), fps=44100, logger=None)

        # 3. AI 人声伴奏分离 (MDX-NET)
        print(f"[{task_id}] 正在通过 AI 分离人声和伴奏...")
        separator = Separator(output_dir=str(temp_dir), model_file_dir=str(MODEL_CACHE))
        separator.load_model("UVR-MDX-NET-Voc_FT.onnx")
        separated_files = separator.separate(str(raw_audio))

        vocal_path = None
        inst_path = None
        for f in separated_files:
            full_f = temp_dir / f
            if "vocals" in f.lower():
                vocal_path = full_f
            else:
                inst_path = full_f

        # 4. 性别检测与静音过滤
        print(f"[{task_id}] 正在识别并剔除男性声音...")
        v_audio, sr = librosa.load(str(vocal_path), sr=None)
        v_16k, _ = librosa.load(str(vocal_path), sr=16000)  # 性别模型必须用 16k

        seg_len = 16000 * 1  # 每 1 秒检测一次
        mask = np.ones(len(v_16k), dtype=np.float32)

        for i in range(0, len(v_16k), seg_len):
            segment = v_16k[i : i + seg_len]
            if len(segment) < 16000:
                continue

            inputs = gender_extractor(
                segment, sampling_rate=16000, return_tensors="pt", padding=True
            ).to(DEVICE)
            with torch.no_grad():
                logits = gender_model(**inputs).logits
                is_male = torch.argmax(logits).item() == 1

            if is_male:
                mask[i : i + seg_len] = 0.0  # 男性声音设为完全静音

        # 对齐掩码到原始采样率
        mask_full = librosa.resample(mask, orig_sr=16000, target_sr=sr)
        min_len = min(len(v_audio), len(mask_full))
        filtered_vocals = v_audio[:min_len] * mask_full[:min_len]

        # 5. 最终混缩 (过滤后的人声 + 原始伴奏)
        print(f"[{task_id}] 正在合成最终音频...")
        inst_audio, _ = librosa.load(str(inst_path), sr=sr)
        min_mix = min(len(filtered_vocals), len(inst_audio))
        final_mixed = filtered_vocals[:min_mix] + inst_audio[:min_mix]

        output_filename = f"api_result_{task_id}.wav"
        final_output_path = OUTPUT_DIR / output_filename
        sf.write(str(final_output_path), final_mixed, sr)

        print(f"[{task_id}] 处理完成，文件已存至: {final_output_path}")
        return send_file(
            str(final_output_path),
            as_attachment=True,
            download_name=f"no_male_voice_{task_id}.wav",
        )

    except Exception as e:
        print(f"[{task_id}] 运行崩溃: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # 清理临时任务目录
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 集成化 AI 视频转音频 API 已启动")
    print("📍 接口地址: http://127.0.0.1:5001/process")
    print(
        "🧠 硬件加速: "
        + ("Enabled (GPU)" if torch.cuda.is_available() else "Disabled (CPU)")
    )
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5001, threaded=True)
