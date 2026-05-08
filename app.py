#!/usr/bin/env python3
"""MTG Realtime Translator — desktop app.

GUI front-end for OpenAI gpt-realtime-translate. Lets you:
- pick input/output audio devices
- pick output language from a dropdown (live-switchable via session.update)
- see translation text streamed in real time
- see live mic level

Run:
    python app.py
"""

import asyncio
import base64
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
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow,
    QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

load_dotenv()

SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_MS = 40
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
MODEL = "gpt-realtime-translate"
WS_URL = f"wss://api.openai.com/v1/realtime/translations?model={MODEL}"

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


class TranslationWorker(QObject):
    """Runs the websocket session in a background thread; talks to GUI via signals."""

    status_changed     = Signal(str)   # "connecting" | "connected" | "disconnected"
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
        self.input_device: Optional[int] = None
        self.output_device: Optional[int] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, target_lang: str, in_dev: Optional[int], out_dev: Optional[int]):
        if self.running:
            return
        self.target_lang = target_lang
        self.input_device = in_dev
        self.output_device = out_dev
        self._stop_event.clear()
        while not self._cmd_queue.empty():
            try:
                self._cmd_queue.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._cmd_queue.put({"type": "stop"})

    def change_language(self, lang: str):
        self.target_lang = lang
        if self.running:
            self._cmd_queue.put({"type": "set_lang", "lang": lang})

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

        async with websockets.connect(WS_URL, additional_headers=headers, max_size=None) as ws:
            first = json.loads(await ws.recv())
            if first.get("type") != "session.created":
                self.error_occurred.emit(f"Unexpected first message: {first.get('type')}")
                return

            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "audio": {"output": {"language": self.target_lang}},
                },
            }))

            self.status_changed.emit("connected")

            loop = asyncio.get_event_loop()
            audio_q: asyncio.Queue = asyncio.Queue()

            def mic_cb(indata, frames, time_info, status):
                pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                level = float(np.abs(indata).mean())
                try:
                    loop.call_soon_threadsafe(audio_q.put_nowait, (pcm16, level))
                except RuntimeError:
                    pass

            out_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS,
                dtype="int16", device=self.output_device,
            )
            out_stream.start()

            in_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS,
                dtype="float32", device=self.input_device,
                blocksize=CHUNK_SAMPLES, callback=mic_cb,
            )
            in_stream.start()

            stop_flag = self._stop_event

            async def sender():
                count = 0
                while not stop_flag.is_set():
                    try:
                        pcm16, level = await asyncio.wait_for(audio_q.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    await ws.send(json.dumps({
                        "type": "session.input_audio_buffer.append",
                        "audio": base64.b64encode(pcm16).decode("ascii"),
                    }))
                    count += 1
                    if count % 3 == 0:
                        self.mic_level.emit(level)

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
                        out_lang = (
                            sess.get("audio", {}).get("output", {}).get("language")
                            or sess.get("output_language")
                        )
                        if out_lang:
                            self.language_confirmed.emit(out_lang)

                    elif etype in (
                        "response.audio.delta",
                        "response.output_audio.delta",
                        "session.output_audio.delta",
                    ):
                        audio_b64 = msg.get("delta") or msg.get("audio") or ""
                        if audio_b64:
                            pcm = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                            try:
                                out_stream.write(pcm)
                            except Exception:
                                pass

                    elif etype in (
                        "response.audio_transcript.delta",
                        "response.output_text.delta",
                        "response.text.delta",
                        "session.output_transcript.delta",
                        "session.output_text.delta",
                    ):
                        delta = msg.get("delta", "")
                        if delta:
                            self.transcript_delta.emit(delta)

                    elif etype in (
                        "response.audio_transcript.done",
                        "response.output_text.done",
                        "response.text.done",
                        "session.output_transcript.done",
                        "session.output_text.done",
                    ):
                        self.transcript_done.emit()

                    elif etype == "error":
                        err = msg.get("error", {})
                        self.error_occurred.emit(err.get("message", json.dumps(err)))

            async def cmd_processor():
                while not stop_flag.is_set():
                    try:
                        cmd = self._cmd_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.1)
                        continue
                    if cmd["type"] == "stop":
                        break
                    if cmd["type"] == "set_lang":
                        await ws.send(json.dumps({
                            "type": "session.update",
                            "session": {
                                "audio": {"output": {"language": cmd["lang"]}},
                            },
                        }))

            try:
                await asyncio.gather(sender(), receiver(), cmd_processor())
            finally:
                try:
                    in_stream.stop(); in_stream.close()
                except Exception:
                    pass
                try:
                    out_stream.stop(); out_stream.close()
                except Exception:
                    pass


def list_input_devices():
    out = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            out.append((idx, dev["name"]))
    return out


def list_output_devices():
    out = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            out.append((idx, dev["name"]))
    return out


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MTG Realtime Translator")
        self.resize(820, 600)

        self.worker = TranslationWorker()
        self.worker.status_changed.connect(self.on_status)
        self.worker.transcript_delta.connect(self.on_transcript_delta)
        self.worker.transcript_done.connect(self.on_transcript_done)
        self.worker.mic_level.connect(self.on_mic_level)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.language_confirmed.connect(self.on_lang_confirmed)

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Row 1: language + start/stop
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Translate to:"))
        self.lang_combo = QComboBox()
        for label, code in LANGUAGES:
            self.lang_combo.addItem(f"{label}  ({code})", code)
        self.lang_combo.setCurrentIndex(0)
        self.lang_combo.currentIndexChanged.connect(self.on_lang_changed)
        row1.addWidget(self.lang_combo, 1)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.on_start)
        row1.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        row1.addWidget(self.stop_btn)
        layout.addLayout(row1)

        # Row 2: devices
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Mic:"))
        self.input_combo = QComboBox()
        self.input_combo.addItem("default", None)
        for idx, name in list_input_devices():
            self.input_combo.addItem(name, idx)
        row2.addWidget(self.input_combo, 1)

        row2.addWidget(QLabel("Output:"))
        self.output_combo = QComboBox()
        self.output_combo.addItem("default", None)
        for idx, name in list_output_devices():
            self.output_combo.addItem(name, idx)
        row2.addWidget(self.output_combo, 1)
        layout.addLayout(row2)

        # Row 3: status + mic level
        row3 = QHBoxLayout()
        self.status_label = QLabel("● disconnected")
        self.status_label.setStyleSheet("color: gray; font-weight: bold;")
        row3.addWidget(self.status_label)

        row3.addWidget(QLabel("Mic:"))
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(14)
        row3.addWidget(self.level_bar, 1)
        layout.addLayout(row3)

        # Transcript
        layout.addWidget(QLabel("Translation"))
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        font = QFont()
        font.setPointSize(15)
        self.transcript.setFont(font)
        self.transcript.setStyleSheet(
            "QTextEdit { background: #1e1e1e; color: #f5f5f5; border-radius: 8px; padding: 12px; }"
        )
        layout.addWidget(self.transcript, 1)

        # Bottom row: clear button
        row4 = QHBoxLayout()
        row4.addStretch(1)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.transcript.clear())
        row4.addWidget(clear_btn)
        layout.addLayout(row4)

    # --- handlers ---
    @Slot()
    def on_start(self):
        lang = self.lang_combo.currentData()
        in_dev = self.input_combo.currentData()
        out_dev = self.output_combo.currentData()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.input_combo.setEnabled(False)
        self.output_combo.setEnabled(False)
        self.worker.start(lang, in_dev, out_dev)

    @Slot()
    def on_stop(self):
        self.worker.stop()
        self.stop_btn.setEnabled(False)

    @Slot(int)
    def on_lang_changed(self, _idx: int):
        lang = self.lang_combo.currentData()
        self.worker.change_language(lang)

    @Slot(str)
    def on_status(self, status: str):
        color = {"connecting": "orange", "connected": "#3ddc84", "disconnected": "gray"}.get(status, "gray")
        self.status_label.setText(f"● {status}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        if status == "disconnected":
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.input_combo.setEnabled(True)
            self.output_combo.setEnabled(True)
            self.level_bar.setValue(0)

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
        cursor.insertText("\n")
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    @Slot(float)
    def on_mic_level(self, level: float):
        self.level_bar.setValue(min(int(level * 400), 100))

    @Slot(str)
    def on_error(self, msg: str):
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(f'<span style="color:#ff6b6b;">[error] {msg}</span><br>')
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    @Slot(str)
    def on_lang_confirmed(self, lang: str):
        # ensure dropdown reflects what server confirmed (in case it was different)
        for i in range(self.lang_combo.count()):
            if self.lang_combo.itemData(i) == lang:
                if self.lang_combo.currentIndex() != i:
                    self.lang_combo.blockSignals(True)
                    self.lang_combo.setCurrentIndex(i)
                    self.lang_combo.blockSignals(False)
                break

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MTG Realtime Translator")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
