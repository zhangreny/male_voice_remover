import os
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# 设置镜像源环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_CACHE = Path("D:/male_voice_remover/models")
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

GENDER_MODEL_NAME = "prithivMLmods/Common-Voice-Gender-Detection"
MIRROR_MODEL_NAME = "hf-mirror.com/prithivMLmods/Common-Voice-Gender-Detection"

print(f"正在尝试下载/缓存模型 {GENDER_MODEL_NAME}...")


def download():
    # 尝试 1: 使用环境变量 + 标准名称
    try:
        print("尝试 1: 使用 HF_ENDPOINT 镜像站...")
        Wav2Vec2FeatureExtractor.from_pretrained(
            GENDER_MODEL_NAME, cache_dir=str(MODEL_CACHE)
        )
        Wav2Vec2ForSequenceClassification.from_pretrained(
            GENDER_MODEL_NAME, cache_dir=str(MODEL_CACHE)
        )
        return True
    except Exception as e:
        print(f"尝试 1 失败: {e}")

    # 尝试 2: 使用直接镜像路径
    try:
        print("\n尝试 2: 使用直接镜像路径...")
        Wav2Vec2FeatureExtractor.from_pretrained(
            MIRROR_MODEL_NAME, cache_dir=str(MODEL_CACHE)
        )
        Wav2Vec2ForSequenceClassification.from_pretrained(
            MIRROR_MODEL_NAME, cache_dir=str(MODEL_CACHE)
        )
        return True
    except Exception as e:
        print(f"尝试 2 失败: {e}")

    return False


if download():
    print("\n✅ 缓存完成！")
else:
    print("\n❌ 所有下载尝试均失败。请检查您的网络或代理设置。")
