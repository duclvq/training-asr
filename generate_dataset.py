"""Sinh dataset câu tiếng Anh có chứa tên/địa danh tiếng Việt — phục vụ finetune ASR.

Yêu cầu: DEEPSEEK_API_KEY trong .env (DeepSeek API tương thích OpenAI).

Output: data/generated_sentences.jsonl
Mỗi dòng:
{
  "id": "<uuid-short>",
  "sentence": "...",                # câu tiếng Anh
  "vietnamese_terms": ["...", ...], # các tên VN xuất hiện trong câu
  "category": "leader|landmark|province|mixed",
  "seed": {...}                     # input dùng để sinh
}

Cách dùng:
    python generate_dataset.py --count 200
    python generate_dataset.py --count 50 --category province
    python generate_dataset.py --dry-run    # chỉ in prompt, không gọi API
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_PATH = DATA_DIR / "generated_sentences.jsonl"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# Tên model do user chỉ định. Nếu API trả lỗi "model not found", thử "deepseek-chat".
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

SYSTEM_PROMPT = """You are a data generator for fine-tuning an English ASR (automatic speech recognition) model that must correctly recognize Vietnamese proper nouns.

Your task: produce natural, fluent ENGLISH sentences that EMBED specific Vietnamese names exactly as given. The Vietnamese names MUST appear verbatim with the original Vietnamese diacritics (e.g., "Phạm Minh Chính", "Hạ Long", "Đà Nẵng") — do not romanize them, do not translate them, do not paraphrase them.

Hard requirements:
1. The CARRIER sentence is in natural English. The Vietnamese names are inserted as proper nouns inside that English sentence.
2. Every Vietnamese name from the input MUST appear at least once, written with the exact original spelling and diacritics.
3. Match the requested STYLE for each item. Vary syntax, vocabulary, and sentence rhythm — do NOT reuse templates across items. Position the Vietnamese name(s) in different parts of the sentence (subject, object, prepositional phrase, apposition, attributive).
4. 12-32 words per sentence. One sentence per item.
5. Do NOT add commentary, numbering, quotes, or markdown. Output ONLY a JSON array of objects.

Style guide:
- news: factual reporting tone, often with a date, location, or quoted source.
- travel: descriptive, sensory, evocative — sights, sounds, food, weather.
- biography: factual life-story tone, mentioning roles, dates, achievements.
- conversation: casual spoken English, contractions, first/second person.
- interview: a direct quote attributed to someone.
- academic: formal register, hedged claims, citations possible.
- business: corporate/economic reporting, numbers, deals, markets.
- sports: commentary tone, action verbs, dramatic.
- weather: weather/forecast report tone with regional details.
- history: narrative past tense, war/colonial/dynastic context.
- social: short, modern, internet-flavored, sometimes emoji-free but informal.
- documentary: voiceover style, "Here in <place>, …" patterns allowed sparingly.

Output format — return ONLY a JSON array. Each element:
{"sentence": "<english sentence containing the Vietnamese name(s) verbatim>", "vietnamese_terms": ["<name1>", "<name2>"]}
"""

STYLES = [
    "news", "travel", "biography", "conversation", "interview",
    "academic", "business", "sports", "weather", "history",
    "social", "documentary",
]

CONTEXTS = [
    "morning routine", "election aftermath", "monsoon season", "Tết holiday",
    "diplomatic visit", "tourism boom", "infrastructure project", "literary review",
    "culinary feature", "wildlife conservation", "tech conference", "school field trip",
    "podcast segment", "academic seminar", "trade deal", "music festival",
    "film premiere", "scientific expedition", "anniversary commemoration",
    "stock market open", "natural disaster aftermath", "historical anniversary",
    "vlog episode", "press briefing", "UN speech", "marathon coverage",
    "art exhibition opening", "harvest season",
]


def _short_id() -> str:
    return uuid.uuid4().hex[:10]


def load_data() -> dict[str, Any]:
    provinces = json.loads((DATA_DIR / "provinces.json").read_text(encoding="utf-8"))
    leaders = json.loads((DATA_DIR / "leaders.json").read_text(encoding="utf-8"))
    landmarks = json.loads((DATA_DIR / "landmarks.json").read_text(encoding="utf-8"))

    all_provinces = [p["name"] for p in provinces["cities"]] + [
        p["name"] for p in provinces["provinces"]
    ]
    top_leaders_with_title = [
        (l["name"], l["title"]) for l in leaders.get("top_leaders", [])
    ]
    top_leaders = [l["name"] for l in leaders.get("top_leaders", [])]
    all_leaders = [
        l["name"]
        for group in ("top_leaders", "key_party_state", "key_ministers", "historical_figures")
        for l in leaders.get(group, [])
    ]
    leader_with_title = [
        (l["name"], l["title"])
        for group in ("top_leaders", "key_party_state", "key_ministers", "historical_figures")
        for l in leaders.get(group, [])
    ]
    all_landmarks = [
        lm["name"]
        for group in ("unesco_world_heritage", "famous_places", "hanoi_specific", "rivers_mountains")
        for lm in landmarks.get(group, [])
    ]
    return {
        "provinces": all_provinces,
        "top_leaders": top_leaders,
        "top_leaders_with_title": top_leaders_with_title,
        "leaders": all_leaders,
        "leaders_with_title": leader_with_title,
        "landmarks": all_landmarks,
    }


def _seed_for_category(data: dict[str, Any], cat: str) -> dict[str, Any]:
    if cat == "top_leader":
        name, title = random.choice(data["top_leaders_with_title"])
        return {"category": "top_leader", "terms": [name], "hint": title}
    if cat == "leader":
        name, title = random.choice(data["leaders_with_title"])
        return {"category": "leader", "terms": [name], "hint": title}
    if cat == "landmark":
        n = random.choice([1, 1, 2])
        return {
            "category": "landmark",
            "terms": random.sample(data["landmarks"], n),
            "hint": "famous Vietnamese landmark",
        }
    if cat == "province":
        n = random.choice([1, 1, 2])
        return {
            "category": "province",
            "terms": random.sample(data["provinces"], n),
            "hint": "Vietnamese province or city",
        }
    if cat == "mixed":
        return {
            "category": "mixed",
            "terms": [
                random.choice(data["top_leaders"] + data["leaders"]),
                random.choice(data["landmarks"] + data["provinces"]),
            ],
            "hint": "mixed Vietnamese proper nouns",
        }
    raise ValueError(f"unknown category: {cat}")


def build_batch(
    data: dict[str, Any],
    pools: list[str],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Một batch gồm `batch_size` 'seeds'. Mỗi seed pick ngẫu nhiên từ `pools`."""
    seeds: list[dict[str, Any]] = []
    for _ in range(batch_size):
        cat = random.choice(pools)
        seed = _seed_for_category(data, cat)
        seed["style"] = random.choice(STYLES)
        seed["context"] = random.choice(CONTEXTS)
        seeds.append(seed)
    return seeds


def build_user_prompt(seeds: list[dict[str, Any]]) -> str:
    lines = [
        "Generate one fluent English sentence per item. Each sentence MUST contain the listed Vietnamese name(s) verbatim with original diacritics.",
        "Match the requested STYLE and CONTEXT. Vary syntax, sentence opening, and where the Vietnamese name appears (subject / object / prepositional / appositive).",
        "Return ONLY a JSON array of objects with keys: sentence, vietnamese_terms.",
        "",
        "Items:",
    ]
    for i, seed in enumerate(seeds, 1):
        terms = ", ".join(f'"{t}"' for t in seed["terms"])
        lines.append(
            f'{i}. category={seed["category"]} | style={seed["style"]} | context={seed["context"]} '
            f'| must_include=[{terms}] | hint={seed["hint"]}'
        )
    return "\n".join(lines)


def parse_response(text: str) -> list[dict[str, Any]]:
    """Robust JSON array parser — strip code fences if model wraps output."""
    text = text.strip()
    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Locate first [ and last ]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in response: {text[:200]}")
    return json.loads(text[start : end + 1])


def verify_sentence(item: dict[str, Any], required_terms: list[str]) -> tuple[bool, str]:
    """Check that all required VN terms appear verbatim in the sentence."""
    sent = item.get("sentence", "")
    if not sent or not isinstance(sent, str):
        return False, "empty sentence"
    missing = [t for t in required_terms if t not in sent]
    if missing:
        return False, f"missing terms: {missing}"
    if len(sent.split()) < 8:
        return False, "sentence too short"
    return True, "ok"


def generate(args: argparse.Namespace) -> None:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: DEEPSEEK_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    data = load_data()
    print(
        f"[load] {len(data['provinces'])} provinces, "
        f"{len(data['leaders'])} leaders, {len(data['landmarks'])} landmarks"
    )

    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    valid = {"province", "top_leader", "leader", "landmark", "mixed"}
    bad = [p for p in pools if p not in valid]
    if bad:
        print(f"ERROR: invalid pools {bad}. Valid: {sorted(valid)}", file=sys.stderr)
        sys.exit(1)
    print(f"[config] pools = {pools}")

    client = None if args.dry_run else OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    DATA_DIR.mkdir(exist_ok=True)
    out_file = open(OUT_PATH, "a", encoding="utf-8")

    total_target = args.count
    total_written = 0
    total_rejected = 0
    batch_size = args.batch_size

    pbar = tqdm(total=total_target, desc="generating", unit="sent")
    try:
        while total_written < total_target:
            need = min(batch_size, total_target - total_written)
            seeds = build_batch(data, pools, need)
            user_prompt = build_user_prompt(seeds)

            if args.dry_run:
                print("\n--- SYSTEM ---\n" + SYSTEM_PROMPT)
                print("\n--- USER ---\n" + user_prompt)
                return

            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.9,
                    max_tokens=2048,
                )
            except Exception as e:
                print(f"\n[api] error: {e}. Retrying in 3s...", file=sys.stderr)
                time.sleep(3)
                continue

            text = resp.choices[0].message.content or ""
            try:
                items = parse_response(text)
            except Exception as e:
                print(f"\n[parse] error: {e}", file=sys.stderr)
                continue

            for seed, item in zip(seeds, items):
                ok, why = verify_sentence(item, seed["terms"])
                if not ok:
                    total_rejected += 1
                    continue
                record = {
                    "id": _short_id(),
                    "sentence": item["sentence"].strip(),
                    "vietnamese_terms": seed["terms"],
                    "category": seed["category"],
                    "seed": seed,
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()
                total_written += 1
                pbar.update(1)
                if total_written >= total_target:
                    break
    finally:
        pbar.close()
        out_file.close()

    print(
        f"\n[done] wrote {total_written} sentences to {OUT_PATH} "
        f"(rejected {total_rejected})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Generate English sentences with embedded Vietnamese names.")
    p.add_argument("--count", type=int, default=100, help="how many sentences to generate")
    p.add_argument(
        "--pools",
        default="province,top_leader,landmark,mixed",
        help=(
            "comma-separated pools to draw from. "
            "Choices: province, top_leader, leader, landmark, mixed."
        ),
    )
    p.add_argument("--batch-size", type=int, default=10, help="items per API call")
    p.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model name")
    p.add_argument("--dry-run", action="store_true", help="print prompt and exit (no API call)")
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
