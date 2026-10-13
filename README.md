---
license: mit
pipeline_tag: automatic-speech-recognition
---
# **TELANGANA POLICE STT APP**

Speech-to-text application for Telangana Police. It transcribes field recordings, statements and calls in **22 Indian languages + English**, entirely on-premise — audio never leaves the machine. Word timestamps and SRT / VTT / JSON export are included.

## **Engines**
| Engine | Languages | Notes |
|---|---|---|
| [AI4Bharat IndicConformer-600M-Multi](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) (local ONNX) | 22 Indian languages (IN-22) | Multilingual Conformer, hybrid CTC + RNNT decoding, MIT license |
| OpenAI Whisper `small` via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | English (`en`) | 8-bit CTranslate2 on CPU, beam search, MIT license |

> **Why Whisper for English?** IndicConformer is trained only on the 22 official Indian languages — its vocabulary, language masks and RNNT heads contain no English (`en`) entry. English audio is therefore routed to Whisper automatically inside the app, while all Indian languages use IndicConformer.

## **Running the App**
Ensure that you have the dependencies installed:
```bash
pip install transformers torchaudio "onnxruntime==1.20.1" "onnx==1.20.1" "onnxruntime-gpu==1.20.1" faster-whisper
```
Then launch the Streamlit UI:
```bash
streamlit run app.py
```
The English model (`Systran/faster-whisper-small`, ~460 MB) is downloaded once into the Hugging Face cache on the first English transcription and works fully offline afterwards. Set the environment variable `WHISPER_MODEL_SIZE=tiny|base|small|medium` to trade accuracy for speed (default: `small`).

## **Supported Languages**
### Indian languages — IndicConformer-600M-Multi
- Assamese(`as`)
- Bengali(`bn`)
- Bodo(`brx`)
- Dogri(`doi`)
- Gujarati(`gu`)
- Hindi(`hi`)
- Kannada(`kn`)
- Konkani(`kok`)
- Kashmiri(`ks`)
- Maithili(`mai`)
- Malayalam(`ml`)
- Manipuri(`mni`)
- Marathi(`mr`)
- Nepali(`ne`)
- Odia(`or`)
- Punjabi(`pa`)
- Sanskrit(`sa`)
- Santali(`sat`)
- Sindhi(`sd`)
- Tamil(`ta`)
- Telugu(`te`)
- Urdu(`ur`)

The tokenizers used for these languages can be found [here](https://github.com/AI4Bharat/IndicVoices/tree/master/artifacts/tokenizers)

### English — Whisper (faster-whisper)
- English(`en`) — Whisper `small`, 8-bit CTranslate2, with word timestamps

## **IndicConformer Backend Details**
- **Model Name:** IndicConformer-600M-Multi
- **Repository:** [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
- **Architecture:** Multilingual Conformer-based Hybrid CTC + RNNT ASR model
- **Parameter Size:** 600M
- **Languages Supported:** IN-22 (English handled separately by Whisper)

### **Inference Example (Indian languages)**
```python
from transformers import AutoModel
import torch, torchaudio

# Load the model
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

# Load an audio file
wav, sr = torchaudio.load("audio.flac")
wav = torch.mean(wav, dim=0, keepdim=True)

target_sample_rate = 16000  # Expected sample rate
if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
    wav = resampler(wav)

# Perform ASR with CTC decoding
transcription_ctc = model(wav, "hi", "ctc")
print("CTC Transcription:", transcription_ctc)

# Perform ASR with RNNT decoding
transcription_rnnt = model(wav, "hi", "rnnt")
print("RNNT Transcription:", transcription_rnnt)
```

### **Inference Example (English)**
```python
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = model.transcribe("audio.flac", language="en", beam_size=5, word_timestamps=True)
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

## **Contact**
For any questions or feedback, please contact:

- Tahir Javed (tahir@cse.iitm.ac.in)
- Kaushal Bhogale (CS22D006@cse.iitm.ac.in)