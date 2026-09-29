"""Render article MP3s using a locally hosted Qwen3-TTS CustomVoice server.

The server runs from the separately licensed qwen3-tts C inference engine and
the official Qwen3-TTS-12Hz-0.6B-CustomVoice model. Neither weights nor a
reference recording are included in this repository.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import re
import time
import urllib.error
import urllib.request
import wave
from array import array
from html.parser import HTMLParser
from pathlib import Path

import lameenc


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "public" / "audio" / "articles"
SAMPLE_RATE = 24000
MAX_CHARS = 110
VOICE = "uncle_fu"


class ArticleBlocks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.current: list[str] | None = None
        self.depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "h2", "h3", "blockquote"} and self.current is None:
            self.current = []
            self.depth = 1
        elif self.current is not None:
            if tag == "br":
                preceding = "".join(self.current).rstrip()
                if preceding and preceding[-1] not in "。！？!?；;，、：:":
                    self.current.append("。")
            elif tag not in {"img", "hr", "input", "wbr"}:
                self.depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self.handle_starttag(tag, attrs)
        else:
            super().handle_startendtag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self.current is None:
            return
        self.depth -= 1
        if self.depth == 0:
            text = re.sub(r"\s+", "", html.unescape("".join(self.current)))
            if text:
                self.blocks.append(text)
            self.current = None

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.current.append(data)


def load_articles() -> list[dict]:
    articles = []
    for filename in ("articles.json", "additional-articles.json"):
        articles.extend(json.loads((PROJECT / "lib" / filename).read_text(encoding="utf-8")))
    return sorted(articles, key=lambda item: (item["number"], item["id"] != "30", item["id"]))


def chunks(text: str) -> list[str]:
    """Keep natural sentence endings while bounding each TTS request."""
    result: list[str] = []
    rest = text
    while len(rest) > MAX_CHARS:
        boundary = max(
            (index + 1 for index, character in enumerate(rest[:MAX_CHARS])
             if character in "。！？!?；;，、：:,"),
            default=MAX_CHARS,
        )
        result.append(rest[:boundary])
        rest = rest[boundary:]
    if rest:
        result.append(rest)
    if "".join(result) != text:
        raise ValueError("Text chunking changed article content")
    return result


def request_audio(server: str, text: str, seed: int) -> bytes:
    payload = json.dumps(
        {"text": text, "speaker": VOICE, "language": "Chinese", "seed": seed,
         "temperature": 0.5, "top_k": 50, "rep_penalty": 1.05},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{server.rstrip('/')}/v1/tts", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=360) as response:
            if not response.headers.get("Content-Type", "").startswith("audio/"):
                raise ValueError(f"TTS server returned {response.headers.get('Content-Type')}")
            data = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"TTS server HTTP {error.code}: {error.read()[:300]!r}") from error
    with wave.open(io.BytesIO(data), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Unexpected WAV format for {text[:20]!r}")
        samples = audio.readframes(audio.getnframes())
    duration = len(samples) / (2 * SAMPLE_RATE)
    if duration < max(0.8, 0.08 * len(text)) or duration > max(15, 0.85 * len(text)):
        raise ValueError(f"Implausible {duration:.1f}s reading of {len(text)} characters: {text[:35]!r}")
    pcm = array("h")
    pcm.frombytes(samples)
    rms = math.sqrt(sum(value * value for value in pcm[::32]) / max(1, len(pcm[::32])))
    if rms < 60:
        raise ValueError(f"Near-silent reading: {text[:35]!r}")
    return samples


def fade(samples: bytes) -> bytes:
    pcm = array("h")
    pcm.frombytes(samples)
    count = min(240, len(pcm) // 2)
    for index in range(count):
        gain = (index + 1) / count
        pcm[index] = int(pcm[index] * gain)
        pcm[-index - 1] = int(pcm[-index - 1] * gain)
    return pcm.tobytes()


def encoder() -> lameenc.Encoder:
    result = lameenc.Encoder()
    result.set_bit_rate(48)
    result.set_in_sample_rate(SAMPLE_RATE)
    result.set_channels(1)
    result.set_quality(2)
    return result


def generate(article: dict, server: str, output_dir: Path, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{article['id']}.mp3"
    if destination.exists() and not force:
        print(f"SKIP {article['id']}", flush=True)
        return
    parser = ArticleBlocks()
    parser.feed(article["html"])
    if not parser.blocks:
        raise ValueError(f"No readable text for article {article['id']}")
    blocks = [article["title"], *parser.blocks]
    segments = [(part, block_index) for block_index, block in enumerate(blocks) for part in chunks(block)]
    coder = encoder()
    temp = destination.with_suffix(".mp3.part")
    started = time.monotonic()
    duration = 0.0
    try:
        with temp.open("wb") as target:
            for index, (part, block_index) in enumerate(segments):
                samples = fade(request_audio(server, part, 20260929 + article["number"] * 1000 + index))
                target.write(coder.encode(samples))
                duration += len(samples) / (2 * SAMPLE_RATE)
                if index < len(segments) - 1:
                    next_block = segments[index + 1][1]
                    pause = 0.85 if block_index == 0 else (0.55 if next_block != block_index else 0.17)
                    target.write(coder.encode(bytes(round(SAMPLE_RATE * pause) * 2)))
                    duration += pause
                if index % 10 == 0 or index == len(segments) - 1:
                    print(f"{article['id']} {index + 1}/{len(segments)} {duration / 60:.1f} min", flush=True)
            target.write(coder.flush())
        if temp.stat().st_size < 4096:
            raise ValueError(f"Audio unexpectedly small: {temp}")
        temp.replace(destination)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print(f"DONE {article['id']} {duration / 60:.1f} min, {destination.stat().st_size / 1e6:.1f} MB, {time.monotonic() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--article", help="One two-digit article ID")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    articles = load_articles()
    if args.article:
        articles = [article for article in articles if article["id"] == args.article]
        if not articles:
            parser.error(f"Unknown article: {args.article}")
    for article in articles:
        generate(article, args.server, args.output_dir, args.force)


if __name__ == "__main__":
    main()
