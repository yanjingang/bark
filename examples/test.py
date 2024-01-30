import os
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["SUNO_USE_SMALL_MODELS"] = "1"

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("audio/bark_generation.wav", SAMPLE_RATE, audio_array)
  
