"""Generates a bio once per NPC (world + ref-id), caches it on disk and applies it to the Character.

Timing: generation starts when the actors of a conversation are loaded. We wait at most
`bio_generation_wait_seconds` (sync with timeout). If the model is slower, the conversation starts with the
original bio and the generation finishes in the background: the result is written to disk (used from the next
conversation on) and also set on the Character so any later prompt rebuild in this conversation picks it up.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Iterable

import src.utils as utils
from src.bio_generation.npc_facts import NpcFacts, facts_from_character, is_creature_race
from src.bio_generation.prompt import build_messages, parse_response
from src.bio_generation.quality import BioQuality, BioScope, TemplateIndex, classify
from src.bio_generation.store import BioStore, entry_matches, entry_text, make_entry

logger = utils.get_logger()

LLMCall = Callable[[list[dict[str, str]]], "str | None"]
RETRY_AFTER_FAILURE_SECONDS = 600


def compose_bio(original: str, generated: str, quality: BioQuality, name: str) -> str:
    original = str(original or "").strip()
    if quality == BioQuality.MISSING or not original:
        return generated
    if quality == BioQuality.SHARED:
        return f"{original}\n\nAbout this particular {name}: {generated}"
    return f"{original}\n{generated}"


class BioGenerator:
    def __init__(self, llm_call: LLMCall, model_name: str, store: BioStore, template_index: TemplateIndex,
                 scope: BioScope = BioScope.ALL, min_words: int = 40, wait_seconds: float = 6.0, max_workers: int = 2):
        self.__llm_call = llm_call
        self.__model_name = model_name
        self.__store = store
        self.__index = template_index
        self.__scope = scope
        self.__min_words = min_words
        self.__wait_seconds = max(0.0, float(wait_seconds))
        self.__executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bio_generation")
        self.__lock = threading.Lock()
        self.__inflight: dict[tuple[str, str, str], Future] = {}
        self.__failed_at: dict[tuple[str, str, str], float] = {}

    @property
    def store(self) -> BioStore:
        return self.__store

    def apply(self, characters: Iterable[Any], world_id: str, location: str | None = None) -> None:
        """Replace / extend the bios of `characters` in place. Never raises."""
        try:
            self.__apply(list(characters), world_id or "default", location or "")
        except Exception as e:
            logger.warning(f"Bio generation skipped: {e}")

    def __apply(self, characters: list[Any], world_id: str, location: str) -> None:
        pending: list[tuple[Any, Future, BioQuality, str, bool]] = []
        for character in characters:
            if getattr(character, "is_player_character", False) or is_creature_race(getattr(character, "race", "")):
                continue
            original = str(character.bio or "")
            quality = classify(original, bool(getattr(character, "is_generic_npc", False)), self.__index, self.__min_words)
            if not self.__scope.covers(quality):
                continue
            key = (world_id, str(character.name), str(character.ref_id))
            facts = facts_from_character(character, location, original, seed_key="|".join(key))

            entry = self.__store.load(*key)
            if entry is not None and entry_matches(entry, facts.name, facts.race, facts.gender):
                if entry.get("enabled", True) is False:
                    continue
                text = entry_text(entry)
                if text:
                    character.bio = compose_bio(original, text, quality, facts.name)
                    logger.log(23, f"Using generated bio for {facts.name} ({facts.ref_id}) from {self.__store.path_for(*key)}")
                    continue

            future, is_new = self.__submit(key, facts, quality, world_id)
            if future is not None:
                pending.append((character, future, quality, original, is_new))

        if not pending:
            return
        new_futures = [f for _, f, _, _, is_new in pending if is_new]
        if self.__wait_seconds > 0 and new_futures:  # only a new NPC may delay a request, and only once
            wait(new_futures, timeout=self.__wait_seconds)
        for character, future, quality, original, _ in pending:
            if future.done():
                text = future.result()
                if text:
                    character.bio = compose_bio(original, text, quality, character.name)
            else:
                logger.log(23, f"Bio for {character.name} not ready after {self.__wait_seconds:.0f}s; "
                               f"starting with the original bio, the generated one will be used when ready.")
                future.add_done_callback(self.__late_setter(character, quality, original))

    @staticmethod
    def __late_setter(character: Any, quality: BioQuality, original: str):
        def _set(future: Future):
            try:
                text = future.result()
                if text and character.bio == original:
                    character.bio = compose_bio(original, text, quality, character.name)
            except Exception:
                pass
        return _set

    def __submit(self, key: tuple[str, str, str], facts: NpcFacts, quality: BioQuality, world_id: str) -> tuple[Future | None, bool]:
        with self.__lock:
            existing = self.__inflight.get(key)
            if existing is not None and not existing.done():
                return existing, False
            failed_at = self.__failed_at.get(key)
            if failed_at and time.time() - failed_at < RETRY_AFTER_FAILURE_SECONDS:
                return None, False
            future = self.__executor.submit(self.__generate, key, facts, quality, world_id)
            self.__inflight[key] = future
            return future, True

    def __generate(self, key: tuple[str, str, str], facts: NpcFacts, quality: BioQuality, world_id: str) -> str | None:
        started = time.time()
        try:
            messages = build_messages(facts, quality)
            response = self.__llm_call(messages)
            generated = parse_response(response)
            if generated is None:
                raise ValueError(f"unusable response: {str(response)[:200]!r}")
            mode = "replace" if quality == BioQuality.MISSING else "append"
            entry = make_entry(world_id=world_id, name=facts.name, ref_id=facts.ref_id, base_id=facts.base_id,
                               race=facts.race, gender=facts.gender, location=facts.location, quality=quality.value,
                               mode=mode, model=self.__model_name, given_name=generated.given_name,
                               bio=generated.bio, personality_and_speech=generated.personality_and_speech)
            path = self.__store.save(*key, entry)
            logger.log(23, f"Generated bio for {facts.name} ({facts.ref_id}) in {time.time() - started:.1f}s -> {path}")
            return entry_text(entry)
        except Exception as e:
            with self.__lock:
                self.__failed_at[key] = time.time()
            logger.warning(f"Bio generation for {facts.name} ({facts.ref_id}) failed, keeping the original bio: {e}")
            return None
        finally:
            with self.__lock:
                self.__inflight.pop(key, None)

    def shutdown(self, wait_for_pending: bool = False) -> None:
        self.__executor.shutdown(wait=wait_for_pending)
