"""Generate the site's article narration with the Apache-licensed Kokoro Chinese model.

Install kokoro-onnx, misaki[zh], numpy and lameenc in an isolated Python
environment. Download kokoro-v1.1-zh.onnx and voices-v1.1-zh.bin from the
kokoro-onnx model-files-v1.1 release, then pass their directory with
--model-dir. No reference recording or proprietary voice is used.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import lameenc
import numpy as np
import onnxruntime as ort
import kokoro_onnx
from kokoro_onnx import Kokoro
from misaki.zh import ZHG2P

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "public" / "audio" / "articles"
SAMPLE_RATE = 24000


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
                self.current.append("。")
            elif tag not in {"img", "hr", "input", "wbr"}:
                self.depth += 1

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


def limit_model_threads(threads: int) -> None:
    """Keep concurrent article jobs from creating one full CPU pool each."""
    def session(model_path: str) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        return ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])

    kokoro_onnx.create_session = session


def make_encoder() -> lameenc.Encoder:
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(48)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)
    return encoder


def pcm_bytes(samples: np.ndarray) -> bytes:
    samples = np.asarray(samples, dtype=np.float32)
    if len(samples) > 480:
        ramp = np.linspace(0, 1, 240, dtype=np.float32)
        samples[:240] *= ramp
        samples[-240:] *= ramp[::-1]
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def silence(seconds: float) -> bytes:
    return bytes(round(SAMPLE_RATE * seconds) * 2)


def synthesize(text: str, model: Kokoro, phonemizer: ZHG2P, voice: str, speed: float) -> np.ndarray:
    phonemes, _ = phonemizer(text)
    if not phonemes.strip():
        raise ValueError(f"Could not phonemize text: {text[:50]}")
    samples, rate = model.create(
        phonemes,
        voice=voice,
        speed=speed,
        is_phonemes=True,
        continuous=True,
        sentence_pause=0.33,
        clause_pause=0.12,
    )
    if rate != SAMPLE_RATE or len(samples) == 0:
        raise ValueError(f"Unexpected synthesis output: {rate} Hz, {len(samples)} samples")
    return samples


def generate(article: dict, model: Kokoro, phonemizer: ZHG2P, voice: str, speed: float, force: bool, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{article['id']}.mp3"
    if destination.exists() and not force:
        print(f"SKIP {article['id']} {destination.stat().st_size} bytes", flush=True)
        return
    parser = ArticleBlocks()
    parser.feed(article["html"])
    if not parser.blocks:
        raise ValueError(f"No readable article blocks for {article['id']}")
    blocks = [article["title"], *parser.blocks]
    encoder = make_encoder()
    temporary = destination.with_suffix(".mp3.part")
    started = time.monotonic()
    duration = 0.0
    try:
        with temporary.open("wb") as target:
            for index, block in enumerate(blocks):
                samples = synthesize(block, model, phonemizer, voice, speed)
                duration += len(samples) / SAMPLE_RATE
                target.write(encoder.encode(pcm_bytes(samples)))
                pause = 0.9 if index == 0 else 0.58
                if index != len(blocks) - 1:
                    target.write(encoder.encode(silence(pause)))
                    duration += pause
                if index % 8 == 0 or index == len(blocks) - 1:
                    print(f"{article['id']} {index + 1}/{len(blocks)} {duration / 60:.1f} min", flush=True)
            target.write(encoder.flush())
        if temporary.stat().st_size < 4096:
            raise ValueError(f"Audio is unexpectedly small: {temporary}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"DONE {article['id']} {duration / 60:.1f} min, {destination.stat().st_size / 1_000_000:.1f} MB, {time.monotonic() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--voice", default="zm_009")
    parser.add_argument("--speed", type=float, default=0.94)
    parser.add_argument("--threads", type=int, default=3, help="ONNX CPU threads per process")
    parser.add_argument("--article", help="Only generate one two-digit article ID")
    parser.add_argument("--start-number", type=int, default=1, help="First article number in reading order")
    parser.add_argument("--end-number", type=int, default=30, help="Last article number in reading order")
    parser.add_argument("--force", action="store_true", help="Regenerate existing audio")
    args = parser.parse_args()
    articles = load_articles()
    if args.article:
        articles = [item for item in articles if item["id"] == args.article]
        if not articles:
            parser.error(f"Unknown article: {args.article}")
    else:
        articles = [item for number, item in enumerate(articles, 1) if args.start_number <= number <= args.end_number]
        if not articles:
            parser.error("Article range is empty")
    if args.threads < 1:
        parser.error("--threads must be positive")
    limit_model_threads(args.threads)
    model = Kokoro(str(args.model_dir / "kokoro-v1.1-zh.onnx"), str(args.model_dir / "voices-v1.1-zh.bin"))
    if args.voice not in model.voices:
        parser.error(f"Unknown voice: {args.voice}")
    phonemizer = ZHG2P()
    for article in articles:
        generate(article, model, phonemizer, args.voice, args.speed, args.force, args.output_dir)
    print(f"Generated {len(articles)} article(s) with {args.voice}", flush=True)


if __name__ == "__main__":
    main()
