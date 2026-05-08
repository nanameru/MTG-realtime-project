#!/usr/bin/env python3
"""MTG Realtime Translator — desktop app (OpenAI-styled)."""

import asyncio
import base64
import collections
import json
import os
import queue
import sys
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv
from PySide6.QtCore import Qt, QObject, Signal, Slot
from PySide6.QtGui import QFont, QFontDatabase, QTextCursor, QIcon, QPixmap, QPainter, QColor
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

load_dotenv()

SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_MS = 20                              # mic → WS chunk size
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
OUTPUT_BLOCK_MS = 20                       # OutputStream callback block size
OUTPUT_BLOCK_SAMPLES = SAMPLE_RATE * OUTPUT_BLOCK_MS // 1000
AUDIO_LATENCY = "low"                      # PortAudio: ask for smallest OS buffer
MODEL = "gpt-realtime-translate"
WS_URL = f"wss://api.openai.com/v1/realtime/translations?model={MODEL}"

# Silero VAD — runs locally on the mic stream so we (a) only ship speech to
# the server and (b) can fire an explicit commit the instant the talker stops,
# instead of waiting for the opaque server-side VAD to notice silence.
VAD_SAMPLE_RATE        = 16000             # silero is trained at 16k
VAD_WINDOW_SAMPLES     = 512               # required window size at 16k (32ms)
VAD_RESAMPLE_BLOCK_24K = 768               # 32ms @ 24k → 512 @ 16k (ratio 2/3)
VAD_THRESHOLD          = 0.5
VAD_MIN_SILENCE_MS     = 320               # silence run before declaring end-of-speech
VAD_LOOKBACK_MS        = 240               # pre-roll prepended on speech start so we don't clip onsets
VAD_LOOKBACK_CHUNKS    = VAD_LOOKBACK_MS // CHUNK_MS
# Translation endpoint rejects manual `input_audio_buffer.commit`. Instead,
# after our local VAD detects end-of-speech we ship a short tail of digital
# silence so the *server* VAD trips its own end-of-utterance and commits.
VAD_SILENCE_TAIL_MS     = 800
VAD_SILENCE_TAIL_CHUNKS = VAD_SILENCE_TAIL_MS // CHUNK_MS

LANGUAGES = [
    ("English",     "en"),
    ("日本語",       "ja"),
    ("Español",     "es"),
    ("Français",    "fr"),
    ("Deutsch",     "de"),
    ("中文",         "zh"),
    ("한국어",        "ko"),
    ("Italiano",    "it"),
    ("Português",   "pt"),
    ("Русский",     "ru"),
    ("العربية",      "ar"),
    ("हिन्दी",       "hi"),
    ("ไทย",          "th"),
    ("Tiếng Việt",  "vi"),
    ("Türkçe",      "tr"),
]

# OpenAI-inspired palette
COL_BG          = "#0d0d0d"
COL_SURFACE     = "#171717"
COL_SURFACE_2   = "#1f1f1f"
COL_BORDER      = "#2a2a2a"
COL_BORDER_HI   = "#3a3a3a"
COL_TEXT        = "#ececec"
COL_TEXT_DIM    = "#8e8ea0"
COL_ACCENT      = "#10a37f"   # OpenAI green
COL_ACCENT_HOV  = "#0e8e6e"
COL_ACCENT_DIM  = "#1f3a32"
COL_DANGER      = "#ef4146"

QSS = f"""
* {{
    font-family: ".AppleSystemUIFont", "SF Pro Text", "Helvetica Neue", "Inter", sans-serif;
    color: {COL_TEXT};
}}
QMainWindow, QWidget#central {{
    background: {COL_BG};
}}
QLabel {{
    background: transparent;
    font-size: 13px;
}}
QLabel#title {{
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.2px;
}}
QLabel#caption {{
    color: {COL_TEXT_DIM};
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}}
QLabel#status {{
    color: {COL_TEXT_DIM};
    font-size: 12px;
    font-weight: 500;
}}

QFrame#divider {{
    background: {COL_BORDER};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}

QComboBox {{
    background: {COL_SURFACE};
    border: 1px solid {COL_BORDER};
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 13px;
    min-height: 20px;
}}
QComboBox:hover {{
    border-color: {COL_BORDER_HI};
}}
QComboBox:focus {{
    border-color: {COL_ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    image: none;
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {COL_TEXT_DIM};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background: {COL_SURFACE};
    border: 1px solid {COL_BORDER};
    border-radius: 10px;
    padding: 4px;
    outline: none;
    selection-background-color: {COL_SURFACE_2};
}}

QPushButton {{
    background: {COL_SURFACE};
    border: 1px solid {COL_BORDER};
    border-radius: 18px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
    min-height: 18px;
}}
QPushButton:hover {{
    background: {COL_SURFACE_2};
    border-color: {COL_BORDER_HI};
}}
QPushButton:disabled {{
    color: #4a4a4a;
    border-color: {COL_BORDER};
}}

QPushButton#primary {{
    background: {COL_ACCENT};
    color: #ffffff;
    border: none;
    font-weight: 600;
    padding: 10px 24px;
    border-radius: 20px;
}}
QPushButton#primary:hover {{
    background: {COL_ACCENT_HOV};
}}
QPushButton#primary:disabled {{
    background: {COL_ACCENT_DIM};
    color: #6a8c80;
}}

QPushButton#stop {{
    background: transparent;
    border: 1px solid {COL_BORDER_HI};
    color: {COL_TEXT};
    border-radius: 20px;
    padding: 10px 22px;
    font-weight: 500;
}}
QPushButton#stop:hover {{
    background: {COL_SURFACE};
}}

QTextEdit {{
    background: {COL_SURFACE};
    color: {COL_TEXT};
    border: 1px solid {COL_BORDER};
    border-radius: 14px;
    padding: 20px 22px;
    font-size: 16px;
    selection-background-color: {COL_ACCENT_DIM};
}}
QTextEdit:focus {{
    border-color: {COL_BORDER_HI};
}}

QProgressBar {{
    background: {COL_SURFACE_2};
    border: none;
    border-radius: 2px;
    max-height: 4px;
}}
QProgressBar::chunk {{
    background: {COL_ACCENT};
    border-radius: 2px;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px;
}}
QScrollBar::handle:vertical {{
    background: {COL_BORDER_HI};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COL_TEXT_DIM};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
"""


class StatusDot(QLabel):
    """A small colored circle indicating connection state."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(10, 10)
        self.set_color(COL_TEXT_DIM)

    def set_color(self, color: str):
        pm = QPixmap(10, 10)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(color))
        p.drawEllipse(0, 0, 10, 10)
        p.end()
        self.setPixmap(pm)


# -------- worker (unchanged logic, only signals) --------

class TranslationWorker(QObject):
    status_changed     = Signal(str)
    transcript_delta   = Signal(str)
    transcript_done    = Signal()
    mic_level          = Signal(float)
    error_occurred     = Signal(str)
    language_confirmed = Signal(str)

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        self._cmd_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self.target_lang = "en"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # Pre-warm: open the WebSocket immediately at app launch and keep it
    # idle. The next start_audio() bypasses the ~300–500 ms TLS+WS handshake.
    def preconnect(self, target_lang: str = "en"):
        if self.running:
            return
        self.target_lang = target_lang
        self._stop_event.clear()
        while not self._cmd_queue.empty():
            try: self._cmd_queue.get_nowait()
            except queue.Empty: break
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def start_audio(self, in_dev: Optional[int], out_dev: Optional[int]):
        if not self.running:
            self.preconnect(self.target_lang)
        self._cmd_queue.put({"type": "start_audio", "in": in_dev, "out": out_dev})

    def stop_audio(self):
        if self.running:
            self._cmd_queue.put({"type": "stop_audio"})

    def shutdown(self):
        self._stop_event.set()
        self._cmd_queue.put({"type": "stop"})

    def change_language(self, lang: str):
        self.target_lang = lang
        if self.running:
            self._cmd_queue.put({"type": "set_lang", "lang": lang})

    def change_input_device(self, in_dev: Optional[int]):
        if self.running:
            self._cmd_queue.put({"type": "set_input", "in": in_dev})

    def change_output_device(self, out_dev: Optional[int]):
        if self.running:
            self._cmd_queue.put({"type": "set_output", "out": out_dev})

    def _run(self):
        try:
            asyncio.run(self._async_main())
        except Exception as e:
            self.error_occurred.emit(f"{type(e).__name__}: {e}")
        finally:
            self.status_changed.emit("disconnected")

    async def _async_main(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.error_occurred.emit("OPENAI_API_KEY not set in environment / .env")
            return

        self.status_changed.emit("connecting")
        headers = {"Authorization": f"Bearer {api_key}"}

        async with websockets.connect(
            WS_URL,
            additional_headers=headers,
            max_size=2 ** 22,
            compression=None,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            first = json.loads(await ws.recv())
            if first.get("type") != "session.created":
                self.error_occurred.emit(f"Unexpected first message: {first.get('type')}")
                return

            # session.update — translation endpoint accepts only:
            #   * audio.input.noise_reduction.type ("near_field" | "far_field")
            #   * audio.output.language
            # Other sub-fields (turn_detection, transcription.delay, voice,
            # format, output_modalities, top-level translation) are rejected.
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "audio": {
                        "input": {"noise_reduction": {"type": "near_field"}},
                        "output": {"language": self.target_lang},
                    },
                },
            }))
            self.status_changed.emit("ready")  # connected, idle, awaiting Start

            loop = asyncio.get_event_loop()
            audio_q: asyncio.Queue = asyncio.Queue()
            playback_buf = collections.deque()
            playback_lock = threading.Lock()

            # streams are opened/closed on demand via cmd_processor
            in_stream: Optional[sd.InputStream] = None
            out_stream: Optional[sd.OutputStream] = None
            sender_active = threading.Event()  # gate: only emit audio when live

            # ---- Silero VAD ----------------------------------------------
            # Loaded once per session. ONNX backend keeps the runtime light;
            # inference on a 32ms window at 16kHz is ~0.3 ms on CPU.
            try:
                from silero_vad import load_silero_vad, VADIterator
                import scipy.signal as scisig
                vad_model = load_silero_vad(onnx=True)
                vad_iter = VADIterator(
                    vad_model,
                    threshold=VAD_THRESHOLD,
                    sampling_rate=VAD_SAMPLE_RATE,
                    min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                    speech_pad_ms=0,  # we manage pre-roll ourselves via lookback
                )
            except Exception as e:
                self.error_occurred.emit(f"Silero VAD load failed: {e}")
                return

            vad_buf_24k = np.zeros(0, dtype=np.float32)
            vad_lookback: collections.deque = collections.deque(maxlen=VAD_LOOKBACK_CHUNKS)
            vad_state = {"in_speech": False, "tail_remaining": 0}
            mic_emit_count = {"n": 0}
            silent_pcm16 = np.zeros(CHUNK_SAMPLES, dtype=np.int16).tobytes()

            def vad_reset():
                nonlocal vad_buf_24k
                vad_iter.reset_states()
                vad_buf_24k = np.zeros(0, dtype=np.float32)
                vad_lookback.clear()
                vad_state["in_speech"] = False
                vad_state["tail_remaining"] = 0

            def out_cb(outdata, frames, time_info, status):
                needed = frames
                produced = 0
                with playback_lock:
                    while produced < needed and playback_buf:
                        chunk = playback_buf[0]
                        take = min(len(chunk), needed - produced)
                        outdata[produced:produced + take, 0] = chunk[:take]
                        if take == len(chunk):
                            playback_buf.popleft()
                        else:
                            playback_buf[0] = chunk[take:]
                        produced += take
                if produced < needed:
                    outdata[produced:, 0] = 0

            def mic_cb(indata, frames, time_info, status):
                nonlocal vad_buf_24k
                if not sender_active.is_set():
                    return
                chunk = indata[:, 0].astype(np.float32, copy=False)
                pcm16 = (chunk * 32767).astype(np.int16).tobytes()
                level = float(np.abs(chunk).mean())

                # UI level update — throttled to ~60ms so the bar still moves
                # even when we're gating audio off.
                mic_emit_count["n"] += 1
                if mic_emit_count["n"] % 3 == 0:
                    self.mic_level.emit(level)

                # Run Silero VAD on resampled 16kHz windows. Accumulate raw
                # 24kHz samples and pop 32ms blocks (768 → 512 samples).
                vad_buf_24k = np.concatenate([vad_buf_24k, chunk])
                while len(vad_buf_24k) >= VAD_RESAMPLE_BLOCK_24K:
                    block = vad_buf_24k[:VAD_RESAMPLE_BLOCK_24K]
                    vad_buf_24k = vad_buf_24k[VAD_RESAMPLE_BLOCK_24K:]
                    block_16k = scisig.resample_poly(block, 2, 3).astype(np.float32)
                    if len(block_16k) < VAD_WINDOW_SAMPLES:
                        continue
                    evt = vad_iter(block_16k[:VAD_WINDOW_SAMPLES], return_seconds=False)
                    if not evt:
                        continue
                    if "start" in evt and not vad_state["in_speech"]:
                        vad_state["in_speech"] = True
                        vad_state["tail_remaining"] = 0  # cancel any in-flight silence tail
                        # Flush pre-roll so the model hears the first phoneme.
                        while vad_lookback:
                            pre_pcm, pre_level = vad_lookback.popleft()
                            try:
                                loop.call_soon_threadsafe(
                                    audio_q.put_nowait, (pre_pcm, pre_level),
                                )
                            except RuntimeError:
                                pass
                    elif "end" in evt and vad_state["in_speech"]:
                        vad_state["in_speech"] = False
                        # Schedule a digital-silence tail so the server-side
                        # VAD trips end-of-utterance immediately.
                        vad_state["tail_remaining"] = VAD_SILENCE_TAIL_CHUNKS

                if vad_state["in_speech"]:
                    try:
                        loop.call_soon_threadsafe(audio_q.put_nowait, (pcm16, level))
                    except RuntimeError:
                        pass
                elif vad_state["tail_remaining"] > 0:
                    vad_state["tail_remaining"] -= 1
                    try:
                        loop.call_soon_threadsafe(audio_q.put_nowait, (silent_pcm16, 0.0))
                    except RuntimeError:
                        pass
                else:
                    vad_lookback.append((pcm16, level))

            def open_streams(in_dev: Optional[int], out_dev: Optional[int]):
                nonlocal in_stream, out_stream
                if in_stream is not None:
                    return
                with playback_lock:
                    playback_buf.clear()
                vad_reset()
                out_stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE, channels=CHANNELS,
                    dtype="int16", device=out_dev,
                    blocksize=OUTPUT_BLOCK_SAMPLES, latency=AUDIO_LATENCY,
                    callback=out_cb,
                )
                out_stream.start()
                in_stream = sd.InputStream(
                    samplerate=SAMPLE_RATE, channels=CHANNELS,
                    dtype="float32", device=in_dev,
                    blocksize=CHUNK_SAMPLES, latency=AUDIO_LATENCY,
                    callback=mic_cb,
                )
                in_stream.start()
                sender_active.set()

            def close_streams():
                nonlocal in_stream, out_stream
                sender_active.clear()
                vad_reset()
                if in_stream is not None:
                    try: in_stream.stop(); in_stream.close()
                    except Exception: pass
                    in_stream = None
                if out_stream is not None:
                    try: out_stream.stop(); out_stream.close()
                    except Exception: pass
                    out_stream = None
                with playback_lock:
                    playback_buf.clear()

            stop_flag = self._stop_event

            async def sender():
                while not stop_flag.is_set():
                    try:
                        pcm16, _level = await asyncio.wait_for(audio_q.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    if not sender_active.is_set():
                        continue
                    await ws.send(json.dumps({
                        "type": "session.input_audio_buffer.append",
                        "audio": base64.b64encode(pcm16).decode("ascii"),
                    }))

            async def receiver():
                while not stop_flag.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    msg = json.loads(raw)
                    etype = msg.get("type", "")

                    if etype == "session.updated":
                        sess = msg.get("session", {})
                        out_lang = sess.get("audio", {}).get("output", {}).get("language") \
                                   or sess.get("output_language")
                        if out_lang:
                            self.language_confirmed.emit(out_lang)

                    elif etype in (
                        "response.audio.delta", "response.output_audio.delta",
                        "session.output_audio.delta",
                    ):
                        audio_b64 = msg.get("delta") or msg.get("audio") or ""
                        if audio_b64:
                            pcm = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                            with playback_lock:
                                playback_buf.append(pcm)

                    elif etype in (
                        "response.audio_transcript.delta", "response.output_text.delta",
                        "response.text.delta", "session.output_transcript.delta",
                        "session.output_text.delta",
                    ):
                        delta = msg.get("delta", "")
                        if delta:
                            self.transcript_delta.emit(delta)

                    elif etype in (
                        "response.audio_transcript.done", "response.output_text.done",
                        "response.text.done", "session.output_transcript.done",
                        "session.output_text.done",
                    ):
                        self.transcript_done.emit()

                    elif etype == "error":
                        err = msg.get("error", {})
                        self.error_occurred.emit(err.get("message", json.dumps(err)))

            # remember current devices so set_input / set_output can rebuild
            current_dev = {"in": None, "out": None}

            async def cmd_processor():
                while not stop_flag.is_set():
                    try:
                        cmd = self._cmd_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        continue
                    t = cmd["type"]
                    if t == "stop":
                        break
                    elif t == "start_audio":
                        current_dev["in"] = cmd["in"]
                        current_dev["out"] = cmd["out"]
                        open_streams(cmd["in"], cmd["out"])
                        self.status_changed.emit("live")
                    elif t == "stop_audio":
                        close_streams()
                        self.status_changed.emit("ready")
                    elif t == "set_input":
                        current_dev["in"] = cmd["in"]
                        if in_stream is not None:  # live → swap
                            close_streams()
                            open_streams(current_dev["in"], current_dev["out"])
                            self.status_changed.emit("live")
                    elif t == "set_output":
                        current_dev["out"] = cmd["out"]
                        if out_stream is not None:  # live → swap
                            close_streams()
                            open_streams(current_dev["in"], current_dev["out"])
                            self.status_changed.emit("live")
                    elif t == "set_lang":
                        await ws.send(json.dumps({
                            "type": "session.update",
                            "session": {"audio": {"output": {"language": cmd["lang"]}}},
                        }))

            try:
                await asyncio.gather(sender(), receiver(), cmd_processor())
            finally:
                close_streams()


# -------- helpers --------

def list_input_devices():
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]

def list_output_devices():
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]


# -------- main window --------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Translator")
        self.resize(880, 660)
        self.setMinimumSize(640, 480)

        self.worker = TranslationWorker()
        self.worker.status_changed.connect(self.on_status)
        self.worker.transcript_delta.connect(self.on_transcript_delta)
        self.worker.transcript_done.connect(self.on_transcript_done)
        self.worker.mic_level.connect(self.on_mic_level)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.language_confirmed.connect(self.on_lang_confirmed)

        self._build_ui()
        # Pre-warm the WebSocket the moment the window appears so the first
        # Start click hits an already-handshaked, session.updated connection.
        self.worker.preconnect(self.lang_combo.currentData())

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)

        # ── header ─────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(10)

        title = QLabel("Realtime Translator")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch(1)

        self.status_dot = StatusDot()
        header.addWidget(self.status_dot)
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("status")
        header.addWidget(self.status_label)
        root.addLayout(header)

        div = QFrame(); div.setObjectName("divider")
        root.addWidget(div)

        # ── controls (language + devices) ──────────────────────
        controls = QHBoxLayout()
        controls.setSpacing(20)

        # language column
        lang_col = QVBoxLayout(); lang_col.setSpacing(6)
        cap_lang = QLabel("Output language"); cap_lang.setObjectName("caption")
        lang_col.addWidget(cap_lang)
        self.lang_combo = QComboBox()
        for label, code in LANGUAGES:
            self.lang_combo.addItem(f"{label}  ·  {code}", code)
        self.lang_combo.setCurrentIndex(0)
        self.lang_combo.currentIndexChanged.connect(self.on_lang_changed)
        lang_col.addWidget(self.lang_combo)
        controls.addLayout(lang_col, 1)

        # mic column
        mic_col = QVBoxLayout(); mic_col.setSpacing(6)
        cap_mic = QLabel("Input"); cap_mic.setObjectName("caption")
        mic_col.addWidget(cap_mic)
        self.input_combo = QComboBox()
        self.input_combo.addItem("System default", None)
        for idx, name in list_input_devices():
            self.input_combo.addItem(name, idx)
        self.input_combo.currentIndexChanged.connect(self.on_input_changed)
        mic_col.addWidget(self.input_combo)
        controls.addLayout(mic_col, 1)

        # output column
        out_col = QVBoxLayout(); out_col.setSpacing(6)
        cap_out = QLabel("Output"); cap_out.setObjectName("caption")
        out_col.addWidget(cap_out)
        self.output_combo = QComboBox()
        self.output_combo.addItem("System default", None)
        for idx, name in list_output_devices():
            self.output_combo.addItem(name, idx)
        self.output_combo.currentIndexChanged.connect(self.on_output_changed)
        out_col.addWidget(self.output_combo)
        controls.addLayout(out_col, 1)

        root.addLayout(controls)

        # ── transcript ─────────────────────────────────────────
        cap_trans = QLabel("Translation"); cap_trans.setObjectName("caption")
        root.addWidget(cap_trans)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        f = QFont(); f.setPointSize(16); f.setWeight(QFont.Normal)
        self.transcript.setFont(f)
        root.addWidget(self.transcript, 1)

        # ── footer (mic level + actions) ───────────────────────
        footer = QHBoxLayout(); footer.setSpacing(14)

        level_box = QVBoxLayout(); level_box.setSpacing(4)
        cap_level = QLabel("MIC LEVEL"); cap_level.setObjectName("caption")
        level_box.addWidget(cap_level)
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100); self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        level_box.addWidget(self.level_bar)
        footer.addLayout(level_box, 1)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(lambda: self.transcript.clear())
        footer.addWidget(self.clear_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        footer.addWidget(self.stop_btn)

        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.on_start)
        footer.addWidget(self.start_btn)

        root.addLayout(footer)

    # ── handlers ──────────────────────────────────────────────
    @Slot()
    def on_start(self):
        in_dev = self.input_combo.currentData()
        out_dev = self.output_combo.currentData()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # Devices stay live-switchable — no need to disable the combos.
        self.worker.start_audio(in_dev, out_dev)

    @Slot()
    def on_stop(self):
        self.worker.stop_audio()
        self.stop_btn.setEnabled(False)

    @Slot(int)
    def on_lang_changed(self, _idx: int):
        lang = self.lang_combo.currentData()
        self.worker.change_language(lang)

    @Slot(int)
    def on_input_changed(self, _idx: int):
        self.worker.change_input_device(self.input_combo.currentData())

    @Slot(int)
    def on_output_changed(self, _idx: int):
        self.worker.change_output_device(self.output_combo.currentData())

    @Slot(str)
    def on_status(self, status: str):
        color, text = {
            "connecting":   (COL_TEXT_DIM, "Connecting…"),
            "ready":        ("#f4b740",    "Ready"),
            "live":         (COL_ACCENT,   "Live"),
            "disconnected": (COL_TEXT_DIM, "Idle"),
        }.get(status, (COL_TEXT_DIM, status))
        self.status_dot.set_color(color)
        self.status_label.setText(text)
        if status == "ready":
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.level_bar.setValue(0)
        elif status == "disconnected":
            # WS dropped — relaunch in the background so the next Start works.
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.level_bar.setValue(0)
            self.worker.preconnect(self.lang_combo.currentData())

    @Slot(str)
    def on_transcript_delta(self, delta: str):
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(delta)
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    @Slot()
    def on_transcript_done(self):
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText("\n\n")
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    @Slot(float)
    def on_mic_level(self, level: float):
        self.level_bar.setValue(min(int(level * 400), 100))

    @Slot(str)
    def on_error(self, msg: str):
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(
            f'<div style="color:{COL_DANGER};font-size:13px;">⚠ {msg}</div><br>'
        )
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    @Slot(str)
    def on_lang_confirmed(self, lang: str):
        for i in range(self.lang_combo.count()):
            if self.lang_combo.itemData(i) == lang:
                if self.lang_combo.currentIndex() != i:
                    self.lang_combo.blockSignals(True)
                    self.lang_combo.setCurrentIndex(i)
                    self.lang_combo.blockSignals(False)
                break

    def closeEvent(self, event):
        self.worker.shutdown()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Realtime Translator")
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
