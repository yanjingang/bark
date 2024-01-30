import os
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SUNO_USE_SMALL_MODELS"] = "1"
os.environ["SUNO_OFFLOAD_CPU"] = "1"

# text_prompt = """
#     추석은 내가 가장 좋아하는 명절이다. 나는 며칠 동안 휴식을 취하고 친구 및 가족과 시간을 보낼 수 있습니다.
# """
text_prompt = """
    你好，我是外星人，想找个地球人做朋友。
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("audio/bark_zh_test.wav", SAMPLE_RATE, audio_array)