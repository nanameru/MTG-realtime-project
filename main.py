#!/usr/bin/env python3
"""Real-time speech translation via OpenAI gpt-realtime-translate.

Captures mic -> sends PCM16 over websocket to /v1/realtime/translations ->
plays the translated audio back to a chosen output device (e.g. BlackHole 2ch
for routing into Zoom).

Usage:
    python main.py -t en --debug
    python main.py -t en -o "BlackHole 2ch"
    python main.py --list-devices
"""

import argparse
import asyncio
import base64
import json
import os
import sys
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_MS = 40
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000

MODEL = "gpt-realtime-translate"
WS_URL = f"wss://api.openai.com/v1/realtime/translations?model={MODEL}"

LEVEL_PRINT_EVERY = 25  # ~1s at 40ms chunks


def find_device(name_substring: Optional[str], kind: str) -> Optional[int]:
    if not name_substring:
        return None
    needle = name_substring.lower()
    for idx, dev in enumerate(sd.query_devices()):
        if needle in dev["name"].lower():
            if kind == "input" and dev["max_input_channels"] > 0:
                return idx
            if kind == "output" and dev["max_output_channels"] > 0:
                return idx
    return None


def device_name(idx: Optional[int], fallback: str) -> str:
    if idx is None:
        return fallback
    return sd.query_devices(idx)["name"]


async def run(target_lang: str, in_dev: Optional[int], out_dev: Optional[int], debug: bool):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"connecting to {MODEL} ...")
    async with websockets.connect(WS_URL, additional_headers=headers, max_size=None) as ws:
        first = json.loads(await ws.recv())
        print(f"[server greeting: {first.get('type')}]")
        if debug:
            print(f"[session.created payload: {json.dumps(first, indent=2)}]")

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "audio": {
                    "output": {
                        "language": target_lang,
                    },
                },
            },
        }))
        print(f"[sent session.update with target language={target_lang}]")

        in_name = device_name(in_dev, "default mic")
        out_name = device_name(out_dev, "default speaker")
        print(
            f"listening — translating to {target_lang}. "
            f"Speak into '{in_name}', audio out -> '{out_name}'. Ctrl+C to quit."
        )

        loop = asyncio.get_event_loop()
        audio_q: asyncio.Queue = asyncio.Queue()

        def mic_cb(indata, frames, time_info, status):
            if status and debug:
                print(f"[mic status: {status}]")
            pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            level = float(np.abs(indata).mean())
            try:
                loop.call_soon_threadsafe(audio_q.put_nowait, (pcm16, level))
            except RuntimeError:
                pass

        out_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS,
            dtype="int16", device=out_dev,
        )
        out_stream.start()

        in_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS,
            dtype="float32", device=in_dev,
            blocksize=CHUNK_SAMPLES, callback=mic_cb,
        )
        in_stream.start()

        async def sender():
            count = 0
            while True:
                pcm16, level = await audio_q.get()
                await ws.send(json.dumps({
                    "type": "session.input_audio_buffer.append",
                    "audio": base64.b64encode(pcm16).decode("ascii"),
                }))
                count += 1
                if count % LEVEL_PRINT_EVERY == 0:
                    bars = "#" * min(int(level * 60), 30)
                    print(f"[mic level: {level:.2f} {bars}]")

        async def receiver():
            seen_errors = set()
            while True:
                msg = json.loads(await ws.recv())
                etype = msg.get("type", "")
                if debug:
                    print(f"[event: {etype}]")

                if etype == "session.updated":
                    sess = msg.get("session", {})
                    out_lang = (
                        sess.get("audio", {}).get("output", {}).get("language")
                        or sess.get("translation", {}).get("output_language")
                        or sess.get("output_language")
                    )
                    print(f"[session updated: server says output language={out_lang}]")
                    if debug:
                        print(f"[session.updated payload: {json.dumps(sess, indent=2)}]")

                elif etype == "input_audio_buffer.speech_started":
                    print("[speech detected by VAD]")

                elif etype == "input_audio_buffer.speech_stopped":
                    print("[end of speech, translating...]")

                elif etype == "response.created":
                    print("[generating translation]")

                elif etype in (
                    "response.audio.delta",
                    "response.output_audio.delta",
                    "session.output_audio.delta",
                ):
                    audio_b64 = msg.get("delta") or msg.get("audio") or ""
                    if audio_b64:
                        pcm = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                        out_stream.write(pcm)

                elif etype in (
                    "response.audio_transcript.delta",
                    "response.output_text.delta",
                    "response.text.delta",
                    "session.output_transcript.delta",
                    "session.output_text.delta",
                ):
                    delta = msg.get("delta", "")
                    if delta:
                        print(delta, end="", flush=True)

                elif etype in (
                    "response.audio_transcript.done",
                    "response.output_text.done",
                    "response.text.done",
                    "session.output_transcript.done",
                    "session.output_text.done",
                ):
                    print()  # newline after streamed text

                elif etype == "response.done":
                    status = msg.get("response", {}).get("status", "completed")
                    print(f"[translation done: {status}]")
                    if status != "completed":
                        details = msg.get("response", {}).get("status_details")
                        print(f"  failure: {details}")

                elif etype == "error":
                    err = msg.get("error", {})
                    sig = (err.get("code"), err.get("message"))
                    if sig not in seen_errors:
                        seen_errors.add(sig)
                        print(f"[error: {err}]")

        try:
            await asyncio.gather(sender(), receiver())
        finally:
            in_stream.stop(); in_stream.close()
            out_stream.stop(); out_stream.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-t", "--target", default="en",
                   help="output language code (en, ja, es, fr, ...)")
    p.add_argument("-i", "--input-device", default=None,
                   help="input device name substring (default = system default)")
    p.add_argument("-o", "--output-device", default=None,
                   help="output device name substring, e.g. 'BlackHole 2ch'")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--debug", action="store_true", help="print every websocket event")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    in_dev = find_device(args.input_device, "input")
    out_dev = find_device(args.output_device, "output")
    if args.input_device and in_dev is None:
        print(f"input device '{args.input_device}' not found. Try --list-devices.",
              file=sys.stderr)
        sys.exit(1)
    if args.output_device and out_dev is None:
        print(f"output device '{args.output_device}' not found. Try --list-devices.",
              file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(run(args.target, in_dev, out_dev, args.debug))
    except KeyboardInterrupt:
        print("\nbye.")


if __name__ == "__main__":
    main()
