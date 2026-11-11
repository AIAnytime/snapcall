"""Faker in three roles:

1. Template generation: no API key needed. Write a few templates per option with
   {placeholders}; Faker fills them with realistic names, amounts, dates, products.
2. PII-safe augmentation: swap emails, phones, URLs, amounts and IDs in real
   data with fake ones so the model learns the decision, not the entity.
3. Persona seeds: diversify LLM synthetic generation (who is writing, from where).
"""

from __future__ import annotations

import random
import re
from typing import Any

from . import save_dataset, split_records

PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*|[^{}]*\|[^{}]*)\}")


def _faker(locale: str | list[str] = "en_US", seed: int = 42):
    from faker import Faker
    f = Faker(locale)
    Faker.seed(seed)
    return f


def _custom(f, key: str, rng: random.Random) -> str | None:
    custom = {
        "amount": lambda: f"${rng.choice([rng.randint(5, 99), rng.randint(100, 999), rng.randint(1000, 9999)])}.{rng.randint(0, 99):02d}",
        "order_id": lambda: f"#{rng.choice(['ORD', 'A', 'INV', ''])}{rng.randint(10000, 999999)}",
        "product": lambda: rng.choice(["wireless earbuds", "running shoes", "laptop stand", "coffee grinder",
                                        "phone case", "yoga mat", "desk lamp", "backpack", "smart watch",
                                        "blender", "monitor", "keyboard", "jacket", "air fryer"]),
        "days": lambda: str(rng.randint(2, 30)),
        "number": lambda: str(rng.randint(1, 500)),
        "plan": lambda: rng.choice(["Basic", "Pro", "Team", "Enterprise", "Starter", "Premium"]),
        "weekday": lambda: f.day_of_week(),
        "date": lambda: f.date_this_year().strftime(rng.choice(["%b %d", "%d/%m/%Y", "%Y-%m-%d", "%B %d"])),
    }
    fn = custom.get(key)
    return fn() if fn else None


def fill(template: str, f, rng: random.Random) -> str:
    def rep(m: re.Match) -> str:
        key = m.group(1)
        if "|" in key:  # {a|b|c} -> random alternative
            return rng.choice(key.split("|"))
        v = _custom(f, key, rng)
        if v is not None:
            return v
        attr = getattr(f, key, None)
        if callable(attr):
            try:
                return str(attr())
            except TypeError:
                pass
        return m.group(0)
    return PLACEHOLDER.sub(rep, template)


FAKER_EXAMPLE = {
    "instructions": "Route this support message to the right team.",
    "type": "choice",
    "options": {
        "billing": {"description": "Charges, refunds, invoices, payment methods.",
                    "templates": ["Hi, I'm {first_name}. I was charged {amount} twice for order {order_id}.",
                                  "{Can I|Could you|Please} get a refund for {order_id}? It was {amount}.",
                                  "Why does my {plan} invoice from {date} show {amount}?"]},
        "shipping": {"description": "Delivery status, delays, wrong address.",
                     "templates": ["My {product} (order {order_id}) still hasn't arrived after {days} days.",
                                   "Can you ship order {order_id} to {address} instead?",
                                   "Tracking for my {product} hasn't updated since {weekday}."]},
        "technical": {"description": "Bugs, login problems, errors in the app.",
                      "templates": ["I get an error when logging in with {email}.",
                                    "The app crashes every time I open {product} settings.",
                                    "Password reset emails never reach {email}."]},
    },
}


def generate(spec: dict[str, Any], n_per_option: int = 50, locale: str | list[str] = "en_US",
             seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    f = _faker(locale, seed)
    qtype = spec.get("type", "choice")
    opts: dict[str, Any] = spec["options"]
    names = list(opts)
    if qtype == "score":
        question = {"type": "score", "instructions": spec["instructions"],
                    "criteria": [opts[n].get("description", n) for n in names]}
    elif qtype == "noul":
        question = {"type": "noul", "instructions": spec["instructions"]}
    else:
        question = {"type": "choice", "instructions": spec["instructions"],
                    "criteria": {n: opts[n].get("description", n) for n in names}}
    out, seen = [], set()
    for i, n in enumerate(names):
        tpls = opts[n].get("templates") or []
        if not tpls:
            continue
        tries = 0
        made = 0
        while made < n_per_option and tries < n_per_option * 20:
            tries += 1
            s = fill(rng.choice(tpls), f, rng)
            if s in seen:
                continue
            seen.add(s)
            made += 1
            ans: Any = n if qtype == "choice" else (i if qtype == "score" else n.lower() in ("yes", "true"))
            out.append({"state": s, "question": question, "answer": ans})
    rng.shuffle(out)
    return out


def generate_dataset(name: str, spec: dict[str, Any], n_per_option: int = 50, locale: str = "en_US",
                     val: float = 0.1, test: float = 0.1, seed: int = 42) -> dict[str, Any]:
    recs = generate(spec, n_per_option, locale, seed)
    return save_dataset(name, split_records(recs, val, test, seed),
                        {"kind": "faker", "locale": locale, "n_per_option": n_per_option, "spec": spec})


# --- augmentation -------------------------------------------------------------

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "email"),
    (re.compile(r"https?://\S+"), "url"),
    (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "phone_number"),
    (re.compile(r"[$€£₹]\s?\d[\d,]*(?:\.\d{2})?"), "amount"),
    (re.compile(r"#\w*\d{4,}"), "order_id"),
]


class Augmenter:
    def __init__(self, prob: float = 0.3, locale: str = "en_US", seed: int = 42):
        self.prob = prob
        self.rng = random.Random(seed)
        self.f = _faker(locale, seed)

    def __call__(self, text: str) -> str:
        if self.prob <= 0 or self.rng.random() > self.prob:
            return text
        for pat, kind in _PATTERNS:
            text = pat.sub(lambda _m, k=kind: _custom(self.f, k, self.rng) or str(getattr(self.f, k)()), text)
        return text


def personas(n: int, locale: str | list[str] = "en_US", seed: int = 42) -> list[str]:
    f = _faker(locale, seed)
    rng = random.Random(seed)
    tones = ["terse", "polite", "frustrated", "rambling", "formal", "casual with typos", "anxious", "sarcastic"]
    return [f"{f.first_name()}, a {f.job().lower()} from {f.city()}, writing in a {rng.choice(tones)} tone"
            for _ in range(n)]
