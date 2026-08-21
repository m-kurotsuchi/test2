from __future__ import annotations

import os
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_FILE = BASE_DIR / "高難易度.txt"
SIMILARITY_THRESHOLD = 0.08
PROMPT_LABELS = {"医師", "ai", "doctor", "dr", "先生", "相手"}
RESPONSE_LABELS = {"理想回答", "回答", "返答", "提案", "あなた", "自分", "私"}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def build_ngrams(text: str, size: int = 2) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) < size:
        return {normalized}
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def similarity(left: str, right: str) -> float:
    left_ngrams = build_ngrams(left)
    right_ngrams = build_ngrams(right)
    if not left_ngrams or not right_ngrams:
        return 0.0
    return len(left_ngrams & right_ngrams) / len(left_ngrams | right_ngrams)


def split_label(line: str) -> tuple[str | None, str]:
    match = re.match(r"^\s*([^:：]{1,20})\s*[:：]\s*(.+)$", line)
    if not match:
        return None, line.strip()
    return match.group(1).strip().lower(), match.group(2).strip()


def is_prompt_label(label: str | None) -> bool:
    return label in PROMPT_LABELS if label else False


def is_response_label(label: str | None) -> bool:
    return label in RESPONSE_LABELS if label else False


def clean_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def load_reference_pairs(reference_path: Path) -> list[dict[str, str]]:
    if not reference_path.exists():
        return []

    text = reference_path.read_text(encoding="utf-8")
    pairs: list[dict[str, str]] = []

    for block in re.split(r"\n\s*\n+", text):
        lines = clean_lines(block)
        if len(lines) < 2:
            continue

        prompt = ""
        response = ""
        for line in lines:
            label, content = split_label(line)
            if not prompt and (is_prompt_label(label) or label is None):
                prompt = content
                continue
            if prompt and not response and (is_response_label(label) or label is None):
                response = content
                break

        if prompt and response:
            pairs.append({"prompt": prompt, "response": response})

    if pairs:
        return pairs

    lines = clean_lines(text)
    pending_prompt = ""
    for line in lines:
        label, content = split_label(line)
        if is_prompt_label(label):
            pending_prompt = content
            continue
        if pending_prompt and (is_response_label(label) or label is None):
            pairs.append({"prompt": pending_prompt, "response": content})
            pending_prompt = ""

    return pairs


def fallback_reply(user_message: str, reference_pairs: list[dict[str, str]]) -> str:
    if reference_pairs:
        example = next(iter(reference_pairs))
        return (
            "参考資料に近い発言は見つかりませんでした。"
            "まずは相手の意図を受け止めてから、確認したい点を1つ添えて返すのがおすすめです。\n\n"
            f"例: 「{example['response']}」"
        )

    return (
        "参考ファイルがまだ十分に読み込めていないため、一般的な提案を返します。"
        "まずは共感→確認→次の質問の順で、短く丁寧に返してください。"
    )


def generate_reply(user_message: str, reference_pairs: list[dict[str, str]]) -> dict[str, str | float]:
    ranked_pairs = sorted(
        (
            {
                "prompt": pair["prompt"],
                "response": pair["response"],
                "score": similarity(user_message, pair["prompt"]),
            }
            for pair in reference_pairs
        ),
        key=lambda pair: pair["score"],
        reverse=True,
    )

    best_match = ranked_pairs[0] if ranked_pairs else None
    if best_match and best_match["score"] >= SIMILARITY_THRESHOLD:
        reply = (
            "参考資料で近い会話が見つかりました。"
            f"\n- 近い発言: {best_match['prompt']}"
            f"\n- 提案例: {best_match['response']}"
        )
        return {"reply": reply, "matched_prompt": best_match["prompt"], "score": best_match["score"]}

    return {"reply": fallback_reply(user_message, reference_pairs), "matched_prompt": "", "score": 0.0}


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        REFERENCE_FILE=DEFAULT_REFERENCE_FILE,
        SECRET_KEY=os.environ.get("SECRET_KEY"),
    )

    if test_config:
        app.config.update(test_config)

    app.config["REFERENCE_PAIRS"] = load_reference_pairs(Path(app.config["REFERENCE_FILE"]))

    @app.get("/")
    def index() -> str:
        reference_path = Path(app.config["REFERENCE_FILE"])
        return render_template(
            "index.html",
            reference_file=reference_path.name,
            reference_loaded=reference_path.exists(),
        )

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or request.form
        user_message = (payload.get("message") or payload.get("user_message") or "").strip()
        if not user_message:
            return jsonify({"error": "message is required"}), 400

        reference_pairs = app.config["REFERENCE_PAIRS"]
        result = generate_reply(user_message, reference_pairs)
        return jsonify(result)

    return app

if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=False)
