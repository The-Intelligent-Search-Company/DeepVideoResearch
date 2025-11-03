import json
import os
import whisperx
import gc 
from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF 

device = "cuda" 
batch_size = 64 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v3", device, compute_type=compute_type)
# 3. Assign speaker labels
HF_TOKEN = "YOUR_HF_TOKEN"
# diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

root = "./lvbench_vdb"
for file in os.listdir(root):
    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)
    
    if not file.endswith(".mp3"):
        continue
    
    audio_file = os.path.join(root, file)

    if os.path.exists(audio_file.replace(".mp3", ".json")):
        print(f"File {audio_file.replace('.mp3', '.json')} already exists, skipping...")
        with open(audio_file.replace(".mp3", ".json"), "r") as f:
            legacy_result = json.load(f)
    else:
        legacy_result = None

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    if result["language"] in DEFAULT_ALIGN_MODELS_TORCH or \
        result["language"] in DEFAULT_ALIGN_MODELS_HF:
        lang = result["language"]
    else:
        lang = 'en'
        print(f"Language {result['language']} not supported, using English instead for {audio_file}.")

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    with open(audio_file.replace(".mp3", ".json"), "w") as f:
        json.dump(result, f, indent=4)
    print(f"saved as {audio_file.replace('.mp3', '.json')}")


