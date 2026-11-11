"""LLM-powered synthetic data for a use case.

Providers: Anthropic (official SDK) or any OpenAI-compatible endpoint (OpenAI,
OpenRouter, Groq, Together, vLLM, Ollama...). Generation is label-conditioned
(N states per option) so classes are balanced, and Faker personas diversify the
voice of each batch.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import httpx

from ..paths import load_settings
from . import save_dataset, split_records
from .faker_gen import personas

SYSTEM = ("You generate realistic, diverse training data for a decision model. "
          "Reply with JSON only, no prose and no code fences.")


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None):
        s = load_settings()
        self.provider = provider or s.get("synth_provider", "openai")
        if self.provider == "anthropic":
            self.model = model or s.get("anthropic_model")
            self.key = api_key or s.get("anthropic_api_key")
        else:
            self.model = model or s.get("openai_model")
            self.key = api_key or s.get("openai_api_key")
            self.base_url = (base_url or s.get("openai_base_url") or "https://api.openai.com/v1").rstrip("/")
        if not self.key and not (self.provider == "openai" and "localhost" in getattr(self, "base_url", "")):
            raise RuntimeError(f"No API key configured for provider '{self.provider}'. Add one in Settings or via env.")

    def complete(self, prompt: str, max_tokens: int = 4000) -> str:
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=self.key)
            resp = client.messages.create(model=self.model, max_tokens=max_tokens, system=SYSTEM,
                                          output_config={"effort": "low"},
                                          messages=[{"role": "user", "content": prompt}])
            if resp.stop_reason == "refusal":
                raise RuntimeError("The model declined this request.")
            return "".join(b.text for b in resp.content if b.type == "text")
        r = httpx.post(f"{self.base_url}/chat/completions", timeout=120,
                       headers={"Authorization": f"Bearer {self.key}"} if self.key else {},
                       json={"model": self.model, "max_tokens": max_tokens, "temperature": 1.0,
                             "messages": [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def json(self, prompt: str, max_tokens: int = 4000) -> Any:
        text = self.complete(prompt, max_tokens)
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        return json.loads(m.group(0) if m else text)


def suggest_schema(use_case: str, llm: LLM | None = None) -> dict[str, Any]:
    """Turn a plain-English use case into a typed question definition."""
    llm = llm or LLM()
    prompt = f"""Use case: {use_case}

Design ONE typed decision question for a System One decision model.
Return JSON: {{"type": "choice" | "score" | "noul", "instructions": "<question to the model>",
 "options": {{"<snake_case_name>": "<one-sentence description>", ...}}}}
- choice: 3-12 mutually exclusive options. score: 3-5 ordered levels, lowest first. noul: options {{"no": "...", "yes": "..."}}.
- Descriptions must make the boundaries between options unambiguous."""
    return llm.json(prompt, 2000)


def generate(use_case: str, question: dict[str, Any], n_per_option: int = 40, language: str = "English",
             style: str = "", llm: LLM | None = None, batch: int = 10, seed: int = 42,
             progress: Callable[[int, int, str], None] | None = None) -> list[dict[str, Any]]:
    """question = {"type", "instructions", "options": {name: description}}"""
    llm = llm or LLM()
    qtype = question.get("type", "choice")
    opts: dict[str, str] = question["options"]
    names = list(opts)
    if qtype == "score":
        q = {"type": "score", "instructions": question["instructions"], "criteria": [opts[n] for n in names]}
    elif qtype == "noul":
        q = {"type": "noul", "instructions": question["instructions"]}
    else:
        q = {"type": "choice", "instructions": question["instructions"], "criteria": opts}
    total = n_per_option * len(names)
    seeds = personas(max(1, total // batch + len(names)), seed=seed)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    k = 0
    others = "\n".join(f"- {n}: {d}" for n, d in opts.items())
    for i, name in enumerate(names):
        got, attempts = 0, 0
        while got < n_per_option and attempts < n_per_option // batch + 4:
            attempts += 1
            want = min(batch, n_per_option - got)
            persona = seeds[k % len(seeds)]
            k += 1
            prompt = f"""Use case: {use_case}
Decision question: {question['instructions']}
All options:
{others}

Write {want} distinct, realistic inputs ("states") whose correct answer is clearly **{name}**.
Language: {language}. Voice: {persona}. {style}
Vary length (1-6 sentences), structure and vocabulary. Include some realistic hard cases that
are near the boundary with other options but still clearly {name}. Never mention the option name.
Return JSON: {{"states": ["...", "..."]}}"""
            try:
                data = llm.json(prompt)
                states = data.get("states", data) if isinstance(data, dict) else data
            except Exception as e:  # noqa: BLE001
                if progress:
                    progress(len(out), total, f"retrying {name}: {e}")
                continue
            for s in states:
                s = str(s).strip()
                if len(s) < 3 or s in seen:
                    continue
                seen.add(s)
                ans: Any = name if qtype == "choice" else (i if qtype == "score" else name == "yes")
                out.append({"state": s, "question": q, "answer": ans})
                got += 1
            if progress:
                progress(len(out), total, f"{name}: {got}/{n_per_option}")
    return out


def generate_dataset(name: str, use_case: str, question: dict[str, Any], n_per_option: int = 40,
                     language: str = "English", style: str = "", provider: str | None = None,
                     model: str | None = None, val: float = 0.1, test: float = 0.1, seed: int = 42,
                     progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    llm = LLM(provider, model)
    recs = generate(use_case, question, n_per_option, language, style, llm, seed=seed, progress=progress)
    if not recs:
        raise RuntimeError("The LLM returned no usable examples.")
    return save_dataset(name, split_records(recs, val, test, seed),
                        {"kind": "synthetic", "provider": llm.provider, "model": llm.model,
                         "use_case": use_case, "language": language, "n_per_option": n_per_option})
