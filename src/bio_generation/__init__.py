"""Generated bios for generic / weak-bio NPCs (W7). Entry point: create_bio_generator(); hook: BioGenerator.apply()."""
from __future__ import annotations

import os
from typing import Any

import src.utils as utils
from src.bio_generation.generator import BioGenerator, LLMCall, compose_bio
from src.bio_generation.quality import BioQuality, BioScope, TemplateIndex, classify
from src.bio_generation.store import BioStore

logger = utils.get_logger()

GENERATION_MAX_TOKENS = 450
GENERATION_TEMPERATURE = 0.9
HTTP_TIMEOUT_SECONDS = 30.0

__all__ = ["BioGenerator", "BioQuality", "BioScope", "BioStore", "TemplateIndex", "classify", "compose_bio",
           "create_bio_generator", "make_llm_call", "location_from_request"]


def make_llm_call(client: Any, model: str) -> LLMCall:
    """One chat completion through the client's endpoint/key, without taking the client's generation lock
    (so a slow bio never blocks the NPC's reply) and with its own HTTP timeout."""
    def _call(messages: list[dict[str, str]]) -> str | None:
        params: dict[str, Any] = {"max_tokens": GENERATION_MAX_TOKENS, "temperature": GENERATION_TEMPERATURE}
        try:  # W3: rename / drop parameters the model family does not support (optional module)
            from src.llm.model_capabilities import adapt_request_params  # type: ignore
            params = adapt_request_params(model, params)
        except ImportError:
            pass
        sync_client = client.generate_sync_client()
        try:
            completion = sync_client.chat.completions.create(model=model, messages=messages,
                                                             timeout=HTTP_TIMEOUT_SECONDS, **params)
        finally:
            sync_client.close()
        if not completion or not completion.choices:
            return None
        return completion.choices[0].message.content
    return _call


def create_bio_generator(game: Any, config: Any, client: Any) -> BioGenerator | None:
    """Returns None when the feature is off (default) or not applicable, so callers can do `if gen: gen.apply(...)`."""
    try:
        if getattr(config, "bio_generation_enabled", False) is not True:
            return None
        if getattr(game, "game_name_in_filepath", "") != "skyrim":
            logger.info("Bio generation is only implemented for Skyrim; skipping.")
            return None
        root = os.path.join(os.path.dirname(game.conversation_folder_path), "generated_bios")
        model = str(getattr(config, "bio_generation_model", "") or "").strip() or client.model_name
        generator = BioGenerator(
            llm_call=make_llm_call(client, model),
            model_name=model,
            store=BioStore(root),
            template_index=TemplateIndex(game.character_df),
            scope=BioScope.from_config(getattr(config, "bio_generation_scope", BioScope.ALL.value)),
            min_words=int(getattr(config, "bio_generation_min_words", 40)),
            wait_seconds=float(getattr(config, "bio_generation_wait_seconds", 6)),
        )
        logger.info(f"Bio generation enabled (model {model}); bios are saved in {root}")
        return generator
    except Exception as e:
        logger.warning(f"Bio generation could not be set up and is disabled: {e}")
        return None


def location_from_request(request_json: dict[str, Any], fallback: str | None) -> str | None:
    """The location arrives in the same request as the actors but is applied after them; read it early."""
    try:
        from src.http.communication_constants import communication_constants as comm_consts
        context = request_json.get(comm_consts.KEY_CONTEXT) or {}
        return context.get(comm_consts.KEY_CONTEXT_LOCATION) or fallback
    except Exception:
        return fallback
