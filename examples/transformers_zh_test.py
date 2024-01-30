import os
from transformers import AutoProcessor, BarkModel
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SUNO_USE_SMALL_MODELS"] = "1"
os.environ["SUNO_OFFLOAD_CPU"] = "1"

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

# 注意，这个语音包依赖speaker_embeddings npy文件，如果本地不存在会自动下载，但是需要打开科学上网才能自动下载成功，否则会报找不到对应语言包的npy错误
voice_preset = "v2/zh_speaker_9"    # https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
inputs = processor("你好，我有一只小狗，叫果酱，它是一只金毛，很活泼，很粘人。", voice_preset=voice_preset)
# 注意：这里的中文语音包，生成出来都一股外国人说中文的口音...

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

# save audio to disk
write_wav("audio/transformers_zh_test.wav", SAMPLE_RATE, audio_array)