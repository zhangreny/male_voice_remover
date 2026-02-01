import os
import torch
import librosa
import numpy as np
import soundfile as sf
import tempfile
import shutil
import logging
from pathlib import Path
from moviepy import VideoFileClip
from audio_separator.separator import Separator
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


class LeanProcessor:
    """精简版 AI 处理器"""

    def __init__(self):
        self.output_dir = Path("D:/male_voice_remover/outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型缓存
        self.model_dir = (
            Path(os.path.expanduser("~")) / ".male_voice_remover" / "models"
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 加载性别检测模型
        model_name = "prithivMLmods/Common-Voice-Gender-Detection"
        self.gender_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.gender_model.eval()

    def extract_and_filter(self, video_path, threshold=0.6):
        """核心流程：提取音频 -> 分离 -> 过滤男性 -> 保存音频"""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 1. 提取原始音频
            raw_audio = temp_dir / "raw.wav"
            with VideoFileClip(video_path) as video:
                video.audio.write_audiofile(str(raw_audio), fps=44100, logger=None)

            # 2. 分离人声
            separator = Separator(
                output_dir=str(temp_dir), model_file_dir=str(self.model_dir)
            )
            separator.load_model("UVR-MDX-NET-Voc_FT.onnx")
            separated_files = separator.separate(str(raw_audio))

            vocal_path = None
            inst_path = None
            for f in separated_files:
                if "vocals" in f.lower():
                    vocal_path = temp_dir / f
                else:
                    inst_path = temp_dir / f

            # 3. 性别检测与过滤
            v_audio, sr = librosa.load(str(vocal_path), sr=None)
            # 内部检测使用 16kHz
            v_16k, _ = librosa.load(str(vocal_path), sr=16000)

            # 分段检测 (每3秒一段)
            seg_len = 16000 * 3
            mask = np.ones(len(v_16k), dtype=np.float32)

            for i in range(0, len(v_16k), seg_len):
                segment = v_16k[i : i + seg_len]
                if len(segment) < 16000:
                    continue  # 太短跳过

                inputs = self.gender_extractor(
                    segment, sampling_rate=16000, return_tensors="pt", padding=True
                ).to(self.device)
                with torch.no_grad():
                    logits = self.gender_model(**inputs).logits
                    is_male = torch.argmax(logits).item() == 1  # 1 是男性标签

                if is_male:
                    mask[i : i + seg_len] = 0.0  # 100% 消除

            # 缩放掩码到原始采样率
            mask_full = librosa.resample(mask, orig_sr=16000, target_sr=sr)
            # 确保对齐
            min_len = min(len(v_audio), len(mask_full))
            final_vocals = v_audio[:min_len] * mask_full[:min_len]

            # 4. 混合伴奏并导出
            inst_audio, _ = librosa.load(str(inst_path), sr=sr)
            min_mix = min(len(final_vocals), len(inst_audio))
            mixed = final_vocals[:min_mix] + inst_audio[:min_mix]

            # 导出文件名
            output_name = f"audio_{Path(video_path).stem}.wav"
            output_path = self.output_dir / output_name
            sf.write(str(output_path), mixed, sr)

            return str(output_path)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
