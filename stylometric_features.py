"""
Stylometric feature extraction for cross-lingual phishing/fraud detection.
"""

from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import spacy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_MODEL_MAP = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "pt": "pt_core_news_sm",
}

_NLP_CACHE = {}


def get_nlp(lang: str):
    if lang not in _NLP_CACHE:
        model_name = _MODEL_MAP.get(lang, _MODEL_MAP["en"])
        _NLP_CACHE[lang] = spacy.load(model_name, disable=["ner"])
    return _NLP_CACHE[lang]


URGENCY_MARKERS = {
    "en": ["urgent", "immediately", "act now", "verify", "suspended", "24 hours", "final notice"],
    "es": ["urgente", "inmediatamente", "verificar", "suspendida", "24 horas", "aviso final"],
    "pt": ["urgente", "imediatamente", "verificar", "suspensa", "24 horas", "aviso final"],
}

AUTHORITY_MARKERS = {
    "en": ["your bank", "irs", "security team", "official", "administrator", "support team"],
    "es": ["su banco", "hacienda", "equipo de seguridad", "oficial", "administrador"],
    "pt": ["seu banco", "receita federal", "equipe de segurança", "oficial", "administrador"],
}


def _count_markers(text_lower: str, markers: list[str]) -> int:
    return sum(text_lower.count(m) for m in markers)


@dataclass
class StyloFeatures:
    pos_unigrams: Counter = field(default_factory=Counter)
    pos_bigrams: Counter = field(default_factory=Counter)
    dep_labels: Counter = field(default_factory=Counter)
    morph_tags: Counter = field(default_factory=Counter)
    punct_counts: dict = field(default_factory=dict)
    numeric: dict = field(default_factory=dict)


def extract_doc_features(text: str, lang: str) -> StyloFeatures:
    nlp = get_nlp(lang)
    doc = nlp(text)

    f = StyloFeatures()
    pos_seq = [tok.pos_ for tok in doc if not tok.is_space]

    f.pos_unigrams.update(pos_seq)
    f.pos_bigrams.update(zip(pos_seq, pos_seq[1:]))
    f.dep_labels.update(tok.dep_ for tok in doc if not tok.is_space)
    f.morph_tags.update(str(tok.morph) for tok in doc if str(tok.morph))

    text_lower = text.lower()
    n_chars = max(len(text), 1)
    n_tokens = max(len(doc), 1)
    n_sents = max(len(list(doc.sents)), 1)

    f.punct_counts = {
        "exclam_ratio": text.count("!") / n_chars,
        "question_ratio": text.count("?") / n_chars,
        "caps_ratio": sum(1 for c in text if c.isupper()) / n_chars,
        "digit_ratio": sum(1 for c in text if c.isdigit()) / n_chars,
        "url_count": len(re.findall(r"https?://|www\.", text_lower)),
    }

    f.numeric = {
        "avg_sent_len": n_tokens / n_sents,
        "type_token_ratio": len(set(t.lower_ for t in doc)) / n_tokens,
        "urgency_markers": _count_markers(text_lower, URGENCY_MARKERS.get(lang, URGENCY_MARKERS["en"])),
        "authority_markers": _count_markers(text_lower, AUTHORITY_MARKERS.get(lang, AUTHORITY_MARKERS["en"])),
    }
    return f


class StylometricTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, top_k_pos_bigrams: int = 100, top_k_dep: int = 30, top_k_morph: int = 50):
        self.top_k_pos_bigrams = top_k_pos_bigrams
        self.top_k_dep = top_k_dep
        self.top_k_morph = top_k_morph

    def fit(self, X: Iterable[tuple[str, str]], y=None):
        docs = [extract_doc_features(t, l) for t, l in X]
        self._pos_bigram_vocab = self._top_vocab(docs, "pos_bigrams", self.top_k_pos_bigrams)
        self._dep_vocab = self._top_vocab(docs, "dep_labels", self.top_k_dep)
        self._morph_vocab = self._top_vocab(docs, "morph_tags", self.top_k_morph)
        return self

    @staticmethod
    def _top_vocab(docs: list[StyloFeatures], attr: str, k: int) -> list:
        total = Counter()
        for d in docs:
            total.update(getattr(d, attr))
        return [item for item, _ in total.most_common(k)]

    def transform(self, X: Iterable[tuple[str, str]]) -> np.ndarray:
        rows = []
        for text, lang in X:
            d = extract_doc_features(text, lang)
            row = []
            total_bigrams = sum(d.pos_bigrams.values()) or 1
            row += [d.pos_bigrams.get(v, 0) / total_bigrams for v in self._pos_bigram_vocab]
            total_dep = sum(d.dep_labels.values()) or 1
            row += [d.dep_labels.get(v, 0) / total_dep for v in self._dep_vocab]
            total_morph = sum(d.morph_tags.values()) or 1
            row += [d.morph_tags.get(v, 0) / total_morph for v in self._morph_vocab]
            row += list(d.punct_counts.values())
            row += list(d.numeric.values())
            rows.append(row)
        return np.array(rows, dtype=float)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return (
            [f"posbi_{a}_{b}" for a, b in self._pos_bigram_vocab]
            + [f"dep_{d}" for d in self._dep_vocab]
            + [f"morph_{m}" for m in self._morph_vocab]
            + ["exclam_ratio", "question_ratio", "caps_ratio", "digit_ratio", "url_count"]
            + ["avg_sent_len", "type_token_ratio", "urgency_markers", "authority_markers"]
        )


def build_stylometric_pipeline():
    from lightgbm import LGBMClassifier
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("stylo", StylometricTransformer()),
        ("clf", LGBMClassifier(
            n_estimators=100,
            num_leaves=15,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
        )),
    ])
