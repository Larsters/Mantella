"""Pure-python lore retrieval: alias phrase matching + BM25 over keywords, with location awareness.

Two entry points:
  * ``scene_notes``  - notes about where the conversation happens and who is in it
                       (location chain, factions tied to that place, NPC/player race,
                       factions named at the start of an NPC's bio). Stable for a conversation.
  * ``topic_notes``  - notes about what was just said (player utterance, last NPC line),
                       boosted when they belong to the current place.
Both return at most ``max_notes`` notes within ``max_tokens`` (estimated), excluding given ids.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from src.lore.corpus import LoreNote

_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren as at be because been before being below
between both but by can cannot could did do does doing don down during each few for from further had has
have having he her here hers herself him himself his how i if in into is isn it its itself just know let
me more most my myself no nor not now of off on once only or other ought our ours ourselves out over own
really same say said she should so some such tell than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom
why will with would yes you your yours yourself yourselves think thing things like well oh hey ok okay
going go get got one also much many ever way see look need want something anything heard hear
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_tokens(text: str) -> list[str]:
    """Lowercase, drop apostrophes, split on non-alphanumerics, light plural stemming."""
    text = text.lower().replace("’", "").replace("'", "")
    return [_stem(t) for t in _TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [t for t in normalize_tokens(text) if t not in _STOPWORDS and len(t) > 1]


_RACE_RE = re.compile(r"(nord|imperial|breton|redguard|high\s*elf|wood\s*elf|dark\s*elf|orc|khajiit|argonian|altmer|bosmer|dunmer|orsimer)", re.I)


def normalize_race(race: str) -> str:
    """Maps game race strings like '<Race DarkElfRace (00013742)>' or 'NordRaceVampire' to 'Dark Elf' / 'Nord'."""
    m = _RACE_RE.search(race or "")
    if not m:
        return race or ""
    word = re.sub(r"\s+", "", m.group(1).lower())
    return {"highelf": "High Elf", "woodelf": "Wood Elf", "darkelf": "Dark Elf"}.get(word, word.capitalize())


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 characters per token for English)."""
    return max(1, math.ceil(len(text) / 4))


@dataclass
class LoreQuery:
    location: str = ""
    utterance: str = ""
    previous_line: str = ""
    npc_names: list[str] = field(default_factory=list)
    npc_races: list[str] = field(default_factory=list)
    npc_bios: list[str] = field(default_factory=list)
    player_race: str = ""


class LoreRetriever:
    # scoring weights
    W_LOCATION_DIRECT = 10.0
    W_LOCATION_ANCESTOR = 2.5
    W_PLACE_AFFINITY = 4.0
    W_BIO_FACTION = 4.5
    W_PLAYER_RACE = 4.5
    W_NPC_RACE = 2.0
    W_ALIAS = 5.0
    W_PREVIOUS_LINE = 0.4
    AFFINITY_BOOST = 1.6
    MIN_TOPIC_SCORE = 3.5
    BIO_HEAD_CHARS = 300
    K1, B = 1.2, 0.5

    def __init__(self, notes: dict[str, LoreNote], allow_spoilers: bool = False):
        self.allow_spoilers = allow_spoilers
        self.notes: dict[str, LoreNote] = {
            nid: n for nid, n in notes.items() if allow_spoilers or not n.spoiler
        }
        # alias phrases as normalized token strings, padded with spaces for whole-phrase matching
        self._aliases: dict[str, list[str]] = {
            nid: sorted({" ".join(normalize_tokens(a)) for a in n.aliases if normalize_tokens(a)}, key=len, reverse=True)
            for nid, n in self.notes.items()
        }
        # BM25 documents: title + aliases + keywords only. The note body is deliberately not indexed:
        # it is prose full of incidental words ("city", "rule", "gold") that caused false matches.
        self._doc_tf: dict[str, Counter] = {}
        self._doc_len: dict[str, int] = {}
        df: Counter = Counter()
        for nid, n in self.notes.items():
            tf = Counter(content_tokens(" ".join((n.title, *n.aliases, *n.keywords))))
            self._doc_tf[nid] = tf
            self._doc_len[nid] = sum(tf.values())
            df.update(tf.keys())
        n_docs = max(1, len(self.notes))
        self._avg_len = sum(self._doc_len.values()) / n_docs if self.notes else 1.0
        self._idf = {t: math.log(1 + (n_docs - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    # ------------------------------------------------------------------ helpers
    def _alias_hits(self, text: str) -> dict[str, list[tuple[int, int]]]:
        """Finds whole-phrase alias matches. Longer matches win over shorter overlapping ones
        (so "Whiterun Hold" does not also count as "Whiterun")."""
        padded = " " + " ".join(normalize_tokens(text)) + " "
        if not padded.strip():
            return {}
        candidates: list[tuple[int, int, str]] = []
        for nid, aliases in self._aliases.items():
            for alias in aliases:
                needle = " " + alias + " "
                start = padded.find(needle)
                while start != -1:
                    candidates.append((start, start + len(needle), nid))
                    start = padded.find(needle, start + 1)
        candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
        taken: list[tuple[int, int]] = []
        hits: dict[str, list[tuple[int, int]]] = {}
        for s, e, nid in candidates:
            span = (s + 1, e - 1)
            overlaps_longer = any(span[0] < te and ts < span[1] and (te - ts) > (span[1] - span[0]) for ts, te in taken)
            if overlaps_longer:
                continue
            taken.append(span)
            hits.setdefault(nid, []).append(span)
        return hits

    def _bm25(self, query_tokens: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not query_tokens:
            return scores
        q = Counter(query_tokens)
        for nid, tf in self._doc_tf.items():
            s = 0.0
            dl = self._doc_len[nid]
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                idf = self._idf.get(term, 0.0)
                s += idf * (f * (self.K1 + 1)) / (f + self.K1 * (1 - self.B + self.B * dl / self._avg_len))
            if s:
                scores[nid] = s
        return scores

    def location_chain(self, location: str) -> list[str]:
        """Place note ids for the location, most specific first, followed by their parents."""
        hits = self._alias_hits(location or "")
        direct = [nid for nid in hits if self.notes[nid].scope == "place"]
        # most specific first: a place whose parent is also matched comes before that parent
        direct.sort(key=lambda nid: -self._depth(nid))
        chain: list[str] = []
        for nid in direct:
            cur: str | None = nid
            while cur and cur not in chain:
                chain.append(cur)
                parent = self.notes[cur].parent if cur in self.notes else None
                cur = parent if parent in self.notes else None
        return chain

    def _depth(self, nid: str) -> int:
        depth, cur, seen = 0, self.notes[nid].parent, set()
        while cur and cur in self.notes and cur not in seen:
            seen.add(cur)
            depth += 1
            cur = self.notes[cur].parent
        return depth

    def _self_note_ids(self, npc_names: list[str]) -> set[str]:
        """Person notes about characters who are in the conversation (their bio already covers them)."""
        result: set[str] = set()
        for name in npc_names:
            for nid, spans in self._alias_hits(name).items():
                if self.notes[nid].scope == "person":
                    result.add(nid)
        return result

    @staticmethod
    def _pick(scored: dict[str, float], notes: dict[str, LoreNote], exclude: set[str], max_notes: int, max_tokens: int) -> list[LoreNote]:
        picked: list[LoreNote] = []
        used = 0
        for nid, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(picked) >= max_notes:
                break
            if nid in exclude:
                continue
            cost = estimate_tokens(format_note(notes[nid]))
            if used + cost > max_tokens:
                continue
            picked.append(notes[nid])
            used += cost
        return picked

    # ------------------------------------------------------------------ public API
    def score_scene(self, query: LoreQuery) -> dict[str, float]:
        scores: Counter = Counter()
        chain = self.location_chain(query.location)
        for depth, nid in enumerate(chain):
            scores[nid] += self.W_LOCATION_DIRECT if depth == 0 else self.W_LOCATION_ANCESTOR / depth
        chain_set = set(chain)
        if chain:
            for nid, n in self.notes.items():
                if n.scope in ("faction", "event") and chain_set.intersection(n.places):
                    # closer places weigh more: the first matching place in the chain decides
                    rank = min(chain.index(p) for p in n.places if p in chain_set)
                    scores[nid] += self.W_PLACE_AFFINITY * n.priority / (1 + rank)
        for bio in query.npc_bios:
            for nid in self._alias_hits((bio or "")[: self.BIO_HEAD_CHARS]):
                if self.notes[nid].scope == "faction":
                    scores[nid] += self.W_BIO_FACTION
        for race, weight in [(query.player_race, self.W_PLAYER_RACE)] + [(r, self.W_NPC_RACE) for r in query.npc_races]:
            if not race:
                continue
            for nid in self._alias_hits(normalize_race(race)):
                if self.notes[nid].scope == "race" and nid != "race_nord":
                    scores[nid] += weight
        return dict(scores)

    def score_topic(self, query: LoreQuery) -> dict[str, float]:
        chain_set = set(self.location_chain(query.location))
        scores: Counter = Counter()
        for text, weight in ((query.utterance, 1.0), (query.previous_line, self.W_PREVIOUS_LINE)):
            if not text:
                continue
            for nid, spans in self._alias_hits(text).items():
                scores[nid] += weight * self.W_ALIAS * min(len(spans), 2)
            for nid, s in self._bm25(content_tokens(text)).items():
                scores[nid] += weight * s
        for nid in list(scores):
            n = self.notes[nid]
            if chain_set and (chain_set.intersection(n.places) or n.parent in chain_set):
                scores[nid] *= self.AFFINITY_BOOST
        return {nid: s for nid, s in scores.items() if s >= self.MIN_TOPIC_SCORE}

    def scene_notes(self, query: LoreQuery, max_notes: int = 3, max_tokens: int = 200, exclude: set[str] | None = None) -> list[LoreNote]:
        excl = set(exclude or ()) | self._self_note_ids(query.npc_names)
        return self._pick(self.score_scene(query), self.notes, excl, max_notes, max_tokens)

    def topic_notes(self, query: LoreQuery, max_notes: int = 3, max_tokens: int = 200, exclude: set[str] | None = None) -> list[LoreNote]:
        excl = set(exclude or ()) | self._self_note_ids(query.npc_names)
        return self._pick(self.score_topic(query), self.notes, excl, max_notes, max_tokens)


def format_note(note: LoreNote) -> str:
    return f"{note.title}: {note.text}"


def format_notes(notes: list[LoreNote], header: str = "", bullet: str = "- ") -> str:
    if not notes:
        return ""
    body = "\n".join(bullet + format_note(n) for n in notes)
    return f"{header}\n{body}" if header else body
