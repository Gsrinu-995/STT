"""TELANGANA POLICE STT APP — polished Streamlit UI (local, no cloud).

Indian languages: AI4Bharat IndicConformer-600M-Multi (22 languages, local ONNX).
English:           OpenAI Whisper 'small' via faster-whisper (CTranslate2, int8).
Output:            every transcript is also rendered in Telugu + English via
                   NLLB-200-distilled-600M (local, int8 on CPU) — no matter
                   which language was spoken/selected.

Deps come from the repo README:
    pip install transformers torchaudio "onnxruntime==1.20.1" "onnx==1.20.1" faster-whisper

Run with:
    /home/srinivas/speech/speech/bin/streamlit run /home/srinivas/speech/app.py
"""

import html
import io
import json
import os
import re
import sys
import time

import soundfile as sf
import streamlit as st
import torch
from torchaudio.transforms import Resample

MODEL_DIR = os.environ.get(
    "INDIC_CONFORMER_DIR", "/home/srinivas/speech/indic-conformer"
)
TARGET_SR = 16000
MAX_SECONDS = 300  # guard against very long recordings on CPU
RNNT_WARN_SECONDS = 30  # RNNT is a slow per-frame loop on CPU

LANGUAGES = {
    # Telangana Police priorities first; "English" is served by Whisper
    # (see EN / load_whisper) because IndicConformer has no English support.
    "Telugu": "te",
    "English": "en",
    "Assamese": "as",
    "Bengali": "bn",
    "Bodo": "brx",
    "Dogri": "doi",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Konkani": "kok",
    "Kashmiri": "ks",
    "Maithili": "mai",
    "Malayalam": "ml",
    "Manipuri": "mni",
    "Marathi": "mr",
    "Nepali": "ne",
    "Odia": "or",
    "Punjabi": "pa",
    "Sanskrit": "sa",
    "Santali": "sat",
    "Sindhi": "sd",
    "Tamil": "ta",
    "Urdu": "ur",
}

# native script names shown next to the English name in the picker
NATIVE = {
    "Telugu": "తెలుగు", "English": "English", "Assamese": "অসমীয়া",
    "Bengali": "বাংলা", "Bodo": "बड़ो", "Dogri": "डोगरी",
    "Gujarati": "ગુજરાતી", "Hindi": "हिन्दी", "Kannada": "ಕನ್ನಡ",
    "Konkani": "कोंकणी", "Kashmiri": "کٲشُر", "Maithili": "मैथिली",
    "Malayalam": "മലയാളം", "Manipuri": "ꯃꯤꯇꯩꯂꯣꯟ", "Marathi": "मराठी",
    "Nepali": "नेपाली", "Odia": "ଓଡ଼ିଆ", "Punjabi": "ਪੰਜਾਬੀ",
    "Sanskrit": "संस्कृतम्", "Santali": "ᱥᱟᱱᱛᱟᱲᱤ", "Sindhi": "سنڌي",
    "Tamil": "தமிழ்", "Urdu": "اردو",
}

EN = "English"  # routed to Whisper — IndicConformer has no English tokenizer/head
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")

# ------------------------------------------------------------ translation ----
# NLLB-200 FLORES codes for every language the app supports. Translation is
# always target -> English so the transcript can be shown side by side.
NLLB = {
    "Telugu": "tel_Telu",
    "English": "eng_Latn",
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Konkani": "kok_Deva",
    "Kashmiri": "kas_Arab",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Manipuri": "mni_Mtei",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi": "snd_Arab",
    "Tamil": "tam_Taml",
    "Urdu": "urd_Arab",
}
NLLB_EN = "eng_Latn"
NLLB_TEL = "tel_Telu"  # every transcript is always shown in Telugu + English
NLLB_CHUNK_TOKENS = 512  # NLLB-200 supports 1024 position embeddings
TRANSLATOR_MODEL = os.environ.get(
    "TRANSLATOR_MODEL", "facebook/nllb-200-distilled-600M"
)


# --------------------------------------------------------------------- model ----
@st.cache_resource(show_spinner="Loading IndicConformer model (first run only)…")
def load_model(model_dir: str):
    """Load the locally downloaded ONNX IndicConformer model once per process."""
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from model_onnx import IndicASRConfig, IndicASRModel

    config = IndicASRConfig.from_pretrained(
        model_dir, ts_folder=model_dir, FRAME_DURATION_MS=0.08
    )
    return IndicASRModel(config)


# ------------------------------------------------------- whisper (english) ----
@st.cache_resource(show_spinner="Loading Whisper model for English (first run only)…")
def load_whisper(model_size: str):
    """faster-whisper (CTranslate2) model used for English — cached per process.

    IndicConformer covers only the 22 Indian languages, so English audio is
    routed here. The 'small' model (~460 MB) is fetched once into the
    Hugging Face cache and then works fully offline.
    """
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


# --------------------------------------------------------------------- audio ----
def read_audio(data: bytes) -> torch.Tensor:
    """Decode raw audio bytes -> float32 mono tensor of shape (1, samples)."""
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = audio.mean(axis=1)  # downmix to mono
    wav_t = torch.from_numpy(wav.copy())
    if sr != TARGET_SR:
        wav_t = Resample(sr, TARGET_SR)(wav_t)
    wav_t = wav_t.clamp(-1.0, 1.0)  # in case input is not normalized
    return wav_t.unsqueeze(0)


def transcribe(model, wav: torch.Tensor, lang: str, decoding: str, timestamps: bool):
    """Returns (text, word_timestamps | None).

    English ("en") -> faster-whisper (beam search; word timestamps supported).
    Indian languages -> IndicConformer (word timestamps need CTC decoding).
    """
    if lang == "en":
        return _transcribe_english(wav, timestamps)
    if decoding == "ctc":
        if timestamps:
            text, word_ts = model(wav, lang, "ctc", compute_timestamps="w")
            return text, word_ts[0]
        return model(wav, lang, "ctc"), None
    return model(wav, lang, "rnnt"), None


def _transcribe_english(wav: torch.Tensor, timestamps: bool):
    """English path — Whisper 'small' via faster-whisper, mono 16 kHz float32."""
    whisper = load_whisper(WHISPER_MODEL_SIZE)
    segments, _info = whisper.transcribe(
        wav.squeeze(0).numpy().astype("float32"),
        language="en",
        beam_size=5,
        word_timestamps=timestamps,
    )
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs if s.text.strip())
    words = None
    if timestamps:
        words = [
            (w.word.strip(), w.start, w.end)
            for s in segs
            for w in (s.words or [])
            if w.word.strip()
        ]
    return text, words


# ------------------------------------------------------------- translation ----
@st.cache_resource(show_spinner="Loading NLLB translator (first run only)…")
def load_translator(model_name: str):
    """NLLB-200-distilled-600M tokenizer + model, CPU int8 (fp32 fallback).

    Used to render every transcript in Telugu and English no matter which
    language was spoken. Downloaded once into the HF cache, then offline.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    m.eval()
    # avoid the "max_length vs max_new_tokens" warning: generation is always
    # driven by max_new_tokens in translate_text().
    m.generation_config.max_length = None
    try:  # ~4x smaller and much faster on CPU; keep fp32 if it fails
        from torch.ao.quantization import quantize_dynamic

        m = quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception:
        pass
    return tok, m


_SENT_SPLIT = re.compile(r"(?<=[।॥.!?])\s+")


def _chunk_for_nllb(text: str, max_words: int = 150) -> list:
    """Split on sentence boundaries into ~max_words chunks so long
    transcripts never overflow the model context."""
    chunks, cur = [], []
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if cur and len((" ".join(cur) + " " + sent).split()) > max_words:
            chunks.append(" ".join(cur))
            cur = [sent]
        else:
            cur.append(sent)
        # pathological single sentence without punctuation — hard split
        while len(cur[-1].split()) > max_words * 2:
            words = cur[-1].split()
            chunks.append(" ".join(words[:max_words]))
            cur[-1] = " ".join(words[max_words:])
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def translate_text(text: str, src_code: str, tgt_code: str) -> str:
    """Translate `text` from NLLB `src_code` to `tgt_code` (chunked, beam=4)."""
    if not text.strip() or src_code == tgt_code:
        return text
    tok, translator = load_translator(TRANSLATOR_MODEL)
    tok.src_lang = src_code
    bos = tok.convert_tokens_to_ids(tgt_code)
    out = []
    with torch.no_grad():
        for chunk in _chunk_for_nllb(text):
            enc = tok(
                chunk, return_tensors="pt", truncation=True,
                max_length=NLLB_CHUNK_TOKENS,
            )
            gen = translator.generate(
                **enc, forced_bos_token_id=bos,
                max_new_tokens=NLLB_CHUNK_TOKENS, num_beams=4,
            )
            out.append(tok.batch_decode(gen, skip_special_tokens=True)[0].strip())
    return " ".join(p for p in out if p)


def to_telugu_english(text: str, lang_name: str):
    """Return (telugu, english) renderings of the transcript.

    Spoken Telugu / English is reused verbatim; every other language is
    translated with NLLB-200. Empty transcripts short-circuit to "".
    """
    if not text or not text.strip():
        return "", ""
    src = NLLB.get(lang_name)
    if src is None:
        return "", ""
    telugu = text if lang_name == "Telugu" else translate_text(text, src, NLLB_TEL)
    english = text if lang_name == EN else translate_text(text, src, NLLB_EN)
    return telugu, english



# -------------------------------------------------------------- time helpers ----
def _stamp(t: float, sep: str = ",") -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def cues_from_words(words, duration: float, max_words=8, max_span=5.0, max_gap=1.0):
    """Group (word, start, end) tuples into subtitle cues."""
    if not words:
        return [("", 0.0, duration)] if duration > 0 else []
    cues, cur = [], []
    for w, s, e in words:
        if cur and (
            len(cur) >= max_words
            or s - cur[0][1] > max_span
            or s - cur[-1][2] > max_gap
        ):
            cues.append(cur)
            cur = []
        cur.append((w, s, e))
    if cur:
        cues.append(cur)
    return [(" ".join(w for w, _, _ in c), c[0][1], c[-1][2]) for c in cues]


def to_srt(cues) -> str:
    out = []
    for i, (text, s, e) in enumerate(cues, 1):
        out.append(f"{i}\n{_stamp(s)} --> {_stamp(e)}\n{text}\n")
    return "\n".join(out)


def to_vtt(cues) -> str:
    out = ["WEBVTT", ""]
    for text, s, e in cues:
        out.append(f"{_stamp(s, '.')} --> {_stamp(e, '.')}\n{text}\n")
    return "\n".join(out)


def timeline_html(words, duration: float) -> str:
    """Absolute-positioned per-word bars laid out on a 0→duration track."""
    if not words or duration <= 0:
        return ""
    segs = []
    for w, s, e in words:
        s = min(max(s, 0.0), duration)
        e = min(max(e, s), duration)
        left = s / duration * 100.0
        width = max((e - s) / duration * 100.0, 0.5)
        esc = html.escape(str(w))
        segs.append(
            f'<div class="ic-seg" style="left:{left:.3f}%;width:{width:.3f}%" '
            f'title="{esc} — {s:.2f}s → {e:.2f}s"><span>{esc}</span></div>'
        )
    return (
        f'<div class="ic-track">{"".join(segs)}</div>'
        f'<div class="ic-axis"><span>0.0 s</span><span>{duration:.1f} s</span></div>'
    )


# ------------------------------------------------------------------- styling ----
PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp { font-family: 'Inter', system-ui, sans-serif; }

/* hero banner */
.ic-hero {
  background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 45%, #0ea5e9 100%);
  border-radius: 1.2rem;
  padding: 1.6rem 1.8rem;
  margin-bottom: .8rem;
  color: #fff;
  box-shadow: 0 10px 30px rgba(79, 70, 229, .35);
}
.ic-hero h1 { margin: 0; font-size: 1.9rem; font-weight: 700; color: #fff; }
.ic-hero p  { margin: .35rem 0 0; opacity: .92; font-size: .95rem; }
.ic-chips { margin-top: .8rem; display: flex; flex-wrap: wrap; gap: .45rem; }
.ic-chip {
  background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3);
  padding: .18rem .65rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
}

/* glass card */
.ic-card {
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 1rem;
  padding: 1.1rem 1.3rem;
  margin: .4rem 0 .8rem;
}
[data-theme="light"] .ic-card { background: #f8fafc; border-color: #e2e8f0; }

/* transcript */
.ic-transcript {
  font-size: 1.35rem; line-height: 1.75; white-space: pre-wrap;
  word-break: break-word; min-height: 3rem; color: inherit;
}
.ic-meta { display: flex; gap: 1.4rem; flex-wrap: wrap; margin-top: .8rem;
           font-size: .82rem; opacity: .75; }

/* timeline */
.ic-track {
  position: relative; height: 44px; border-radius: .6rem;
  background: rgba(127,127,127,.10); overflow: hidden; margin-top: .5rem;
}
.ic-seg {
  position: absolute; top: 0; bottom: 0; display: flex; align-items: center;
  justify-content: center; overflow: hidden;
  background: linear-gradient(180deg, #6366f1, #8b5cf6);
  border-right: 1px solid rgba(255,255,255,.35);
  transition: filter .15s;
}
.ic-seg:hover { filter: brightness(1.35); }
.ic-seg span {
  color: #fff; font-size: .68rem; font-weight: 600; padding: 0 .2rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ic-axis { display: flex; justify-content: space-between;
           font-size: .72rem; opacity: .6; margin-top: .2rem; }

/* buttons */
.stButton>button, .stDownloadButton>button {
  border-radius: .6rem; font-weight: 600; letter-spacing: .01em;
  transition: transform .12s ease, box-shadow .12s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
  transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0,0,0,.35);
}

/* metrics row */
.ic-stats [data-testid="stMetric"] {
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09);
  border-radius: .8rem; padding: .7rem .9rem;
}
[data-theme="light"] .ic-stats [data-testid="stMetric"] {
  background: #f8fafc; border-color: #e2e8f0;
}

/* footer */
.ic-footer { text-align: center; font-size: .78rem; opacity: .55; margin-top: 1.5rem; }
</style>
"""


def hero():
    st.markdown(
        """
<div class="ic-hero">
  <h1>🎙️ TELANGANA POLICE STT APP</h1>
  <p>Speech-to-text for Telangana Police — AI4Bharat's 600M-parameter IndicConformer
     for 22 Indian languages plus Whisper for English, running <b>entirely on this machine</b>.
     Every transcript is translated to <b>Telugu + English</b> with NLLB-200.</p>
  <div class="ic-chips">
    <span class="ic-chip">🔒 100% offline</span>
    <span class="ic-chip">⚡ ONNX Runtime</span>
    <span class="ic-chip">🌍 23 languages</span>
    <span class="ic-chip">🇮🇳 Always Telugu + English</span>
    <span class="ic-chip">📝 Word timestamps</span>
    <span class="ic-chip">📄 SRT / VTT export</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------- page config ----
st.set_page_config(
    page_title="TELANGANA POLICE STT APP",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(PAGE_CSS, unsafe_allow_html=True)
hero()

# ------------------------------------------------------------------ sidebar ----
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    lang_labels = [f"{n}  ·  {NATIVE.get(n, '')}" for n in LANGUAGES]
    lang_label = st.selectbox(
        "Language",
        lang_labels,
        index=lang_labels.index(f"Telugu  ·  {NATIVE['Telugu']}"),
        help="Language of the spoken audio. Pick the closest match for best accuracy.",
    )
    lang_name = lang_label.split("  ·  ")[0]
    lang = LANGUAGES[lang_name]

    is_english = lang_name == EN

    decoding = st.radio(
        "Decoding strategy",
        ["ctc", "rnnt"],
        format_func=str.upper,
        disabled=is_english,
        help=(
            "English audio always uses Whisper (beam search) instead of CTC/RNNT."
            if is_english
            else "CTC is fast. RNNT is usually more accurate but much slower on CPU."
        ),
    )

    show_ts = st.checkbox(
        "Word timestamps",
        value=False,
        disabled=not (is_english or decoding == "ctc"),
        help=(
            "Word-level timestamps from Whisper."
            if is_english
            else "Only supported with CTC decoding. Frame duration: 0.08 s."
        ),
    )

    st.divider()
    st.markdown("**Environment**")
    st.code(
        f"model   : {MODEL_DIR}\n"
        f"device  : {'cuda' if torch.cuda.is_available() else 'cpu'}\n"
        f"runtime : onnxruntime + faster-whisper",
        language="text",
    )
    st.caption(
        "Indian languages: ai4bharat/indic-conformer-600m-multilingual · "
        "English: Systran/faster-whisper-small · "
        f"Translation: {TRANSLATOR_MODEL} · MIT license"
    )


# --------------------------------------------------------------------- main ----
model = load_model(MODEL_DIR)

tab_record, tab_upload = st.tabs(["🎤  Record", "📁  Upload file"])

audio_bytes = None
source = None

with tab_record:
    st.caption("Click the microphone, speak, then stop. (Audio stays in your browser.)")
    rec = st.audio_input("Record audio", sample_rate=TARGET_SR)
    if rec is not None:
        audio_bytes = rec.getvalue()
        source = "recording"

with tab_upload:
    up = st.file_uploader(
        "Audio file — wav / flac / ogg / mp3",
        type=["wav", "flac", "ogg", "mp3"],
        help="Files are decoded locally with libsndfile; nothing is uploaded anywhere.",
    )
    if up is not None:
        audio_bytes = up.getvalue()
        source = up.name

if audio_bytes:
    st.audio(audio_bytes)

    go = st.button(
        "✨  Transcribe",
        type="primary",
        width="stretch",
        icon=":material/graphic_eq:",
        help="Run the model on this audio",
    )

    if go:
        try:
            t0 = time.time()
            with st.status("Transcribing…", expanded=True) as status:
                st.write("Decoding audio…")
                wav = read_audio(audio_bytes)
                duration = wav.shape[-1] / TARGET_SR

                if duration < 0.3:
                    status.update(label="Audio too short", state="error")
                    st.warning("Audio is too short — please record at least 0.3 seconds.")
                    st.stop()
                if duration > MAX_SECONDS:
                    status.update(label="Audio too long", state="error")
                    st.warning(
                        f"Audio is longer than {MAX_SECONDS} s — please use a shorter clip."
                    )
                    st.stop()
                if decoding == "rnnt" and not is_english and duration > RNNT_WARN_SECONDS:
                    st.info(
                        f"RNNT decoding on {duration:.0f} s of CPU audio may take a while…"
                    )

                if is_english:
                    st.write("Running Whisper decoding (English)…")
                    decoding_used = "whisper"
                else:
                    st.write(f"Running {decoding.upper()} decoding…")
                    decoding_used = decoding
                text, word_ts = transcribe(model, wav, lang, decoding, show_ts)

                st.write("Translating to Telugu + English (NLLB)…")
                telugu_text, english_text = to_telugu_english(text, lang_name)

            elapsed = time.time() - t0
            status.update(label=f"Done in {elapsed:.2f} s", state="complete")

            st.session_state["result"] = {
                "text": text,
                "telugu": telugu_text,
                "english": english_text,
                "word_ts": word_ts,
                "lang": lang_name,
                "decoding": decoding_used,
                "duration": duration,
                "elapsed": elapsed,
                "source": source,
                "ts": time.strftime("%H:%M:%S"),
            }
            # keep a rolling history (newest first, max 10)
            hist = st.session_state.setdefault("history", [])
            hist.insert(0, st.session_state["result"])
            del hist[10:]
        except Exception as exc:  # surface errors in the UI instead of crashing
            st.error(f"Transcription failed: {exc}")

# ------------------------------------------------------------------ results ----
result = st.session_state.get("result")

if result:
    st.markdown("### 📝 Transcript")

    def _card(label: str, body: str) -> str:
        esc = html.escape(body) if body else "<i>(no speech detected)</i>"
        return (
            f'<div class="ic-card"><div class="ic-meta" style="margin:0 0 .4rem">'
            f"<b>{label}</b></div>"
            f'<div class="ic-transcript">{esc}</div></div>'
        )

    tc, ec = st.columns(2)
    tc.markdown(_card("🇮🇳 తెలుగు — Telugu", result.get("telugu", "")),
                unsafe_allow_html=True)
    ec.markdown(_card("🇬🇧 English", result.get("english", "")),
                unsafe_allow_html=True)

    st.markdown(
        f'<div class="ic-meta">'
        f"<span>🌐 source: {result['lang']}</span>"
        f"<span>⚙️ {result['decoding'].upper()}</span>"
        f"<span>🎧 {result['duration']:.1f} s of audio</span>"
        f"<span>⏱️ {result['elapsed']:.2f} s to decode</span>"
        f"<span>💾 {result['source'] or 'audio'}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander(
        f"Original transcript — {result['lang']} (source language)", expanded=False
    ):
        st.markdown(result.get("text") or "_(no speech detected)_")

    # ---- stat cards ----
    st.markdown('<div class="ic-stats">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Language", result["lang"])
    c2.metric("Audio length", f"{result['duration']:.1f} s")
    c3.metric("Decode time", f"{result['elapsed']:.2f} s")
    rtf = result["elapsed"] / result["duration"] if result["duration"] else 0.0
    c4.metric("Speed (RTF)", f"{rtf:.2f}×")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- exports ----
    words = result.get("word_ts")
    cues = cues_from_words(words, result["duration"])
    if not words and result["text"]:
        cues = [(result["text"], 0.0, result["duration"])]
    fname = "transcript_telugu_english"

    text_payload = (
        f"Telugu:\n{result.get('telugu', '')}\n\n"
        f"English:\n{result.get('english', '')}\n\n"
        f"Original ({result['lang']}):\n{result['text']}\n"
    )

    json_payload = json.dumps(
        {
            "text": result["text"],
            "telugu": result.get("telugu", ""),
            "english": result.get("english", ""),
            "language": result["lang"],
            "language_code": LANGUAGES[result["lang"]],
            "decoding": result["decoding"],
            "audio_seconds": round(result["duration"], 3),
            "decode_seconds": round(result["elapsed"], 3),
            "words": [
                {"word": w, "start": round(s, 3), "end": round(e, 3)}
                for w, s, e in (words or [])
            ],
        },
        ensure_ascii=False,
        indent=2,
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(
        "⬇️ Text", text_payload, file_name=f"{fname}.txt",
        mime="text/plain", width="stretch", icon=":material/description:",
    )
    d2.download_button(
        "⬇️ SRT", to_srt(cues), file_name=f"{fname}.srt",
        mime="text/plain", width="stretch", icon=":material/subtitles:",
    )
    d3.download_button(
        "⬇️ VTT", to_vtt(cues), file_name=f"{fname}.vtt",
        mime="text/vtt", width="stretch", icon=":material/movie:",
    )
    d4.download_button(
        "⬇️ JSON", json_payload, file_name=f"{fname}.json",
        mime="application/json", width="stretch", icon=":material/data_object:",
    )

    # ---- word timeline ----
    if words:
        with st.expander("📊 Word timeline", expanded=True):
            st.markdown(timeline_html(words, result["duration"]),
                        unsafe_allow_html=True)
            st.dataframe(
                {
                    "Word": [w for w, _, _ in words],
                    "Start (s)": [round(s, 2) for _, s, _ in words],
                    "End (s)": [round(e, 2) for _, _, e in words],
                },
                width="stretch",
                hide_index=True,
                height=min(3 + 35 * len(words), 400),
            )

# ------------------------------------------------------------------ history ----
history = st.session_state.get("history") or []
if history:
    with st.expander(f"🕘  Session history ({len(history)})", expanded=False):
        for i, h in enumerate(history):
            m1, m2 = st.columns([4, 1])
            preview = (h.get("telugu") or h.get("english") or h.get("text") or "(empty)")[:120]
            m1.markdown(
                f"`{h['ts']}` **{h['lang']} → TE + EN** · {h['decoding'].upper()} · "
                f"{h['duration']:.1f}s — {html.escape(preview)}"
                + ("…" if len(h.get("telugu") or h.get("english") or h.get("text") or "") > 120 else "")
            )
            if m2.button("Restore", key=f"restore_{i}", width="stretch"):
                st.session_state["result"] = h
                st.rerun()

st.markdown(
    '<div class="ic-footer">TELANGANA POLICE STT APP · '
    'IndicConformer-600M-Multi (AI4Bharat) + Whisper (English) + NLLB-200 '
    '(Telugu/English translation) · MIT license · '
    'powered by Streamlit + ONNX Runtime + faster-whisper</div>',
    unsafe_allow_html=True,
)


