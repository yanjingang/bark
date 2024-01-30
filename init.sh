#!/bin/bash

# download code
cd ~/project/
git clone https://github.com/yanjingang/bark
cd bark/

# install depend
pip install --upgrade pip
pip install .
pip install --upgrade transformers scipy

# download model file
mkdir model && cd ~/project/bark/model/
wget -c -O config.json  https://huggingface.co/suno/bark/resolve/main/config.json?download=true
wget -c -O vocab.txt  https://huggingface.co/suno/bark/resolve/main/vocab.txt?download=true
wget -c -O tokenizer.json  https://huggingface.co/suno/bark/resolve/main/tokenizer.json?download=true
wget -c -O tokenizer_config.json  https://huggingface.co/suno/bark/resolve/main/tokenizer_config.json?download=true
wget -c -O special_tokens_map.json  https://huggingface.co/suno/bark/resolve/main/special_tokens_map.json?download=true
wget -c -O speaker_embeddings_path.json  https://huggingface.co/suno/bark/resolve/main/speaker_embeddings_path.json?download=true
wget -c -O generation_config.json  https://huggingface.co/suno/bark/resolve/main/generation_config.json?download=true
wget -c -O pytorch_model.bin https://huggingface.co/suno/bark/resolve/main/pytorch_model.bin?download=true
wget -c -O text_2.pt https://huggingface.co/suno/bark/resolve/main/text_2.pt?download=true
wget -c -O coarse_2.pt https://huggingface.co/suno/bark/resolve/main/coarse_2.pt?download=true
wget -c -O fine_2.pt https://huggingface.co/suno/bark/resolve/main/fine_2.pt?download=true
cp pytorch_model.bin *.pt *.json *.txt ~/.cache/huggingface/hub/models--suno--bark/snapshots/70a8a7d34168586dc5d028fa9666aceade177992/

# test
cd ~/project/bark/examples/
python3 test.py


