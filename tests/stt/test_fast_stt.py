import sys
import types
import numpy as np
import pytest
from src.stt.fast_stt import MLX_REPOS, MlxWhisper, clean_transcript

PROMPT = "This is a conversation with Adrianne Avenicci and Alfhild Battle-Born in Whiterun."


@pytest.mark.parametrize("text", [
    "This is a conversation with Adrianne Avenicci and Alfhild Battle-Born in Whiterun.",  # prompt echoed back
    "conversation with Adrianne Avenicci and Alfhild Battle-Born",
    "Adrianne Avenicci, Alfhild Battle-Born, Whiterun.",  # just the names from the prompt
    "Vaska, Vaska, Vaska, Vaska, Vaska, Vaska, Vaska, Vaska, Vaska",  # repetition loop (seen with large-v3-turbo)
    "and the Vaska-Mane-Mane-Mane-Mane-Mane.",
    "Окей, Адриана Веничи.",  # wrong script for English (seen with gpt-4o-mini-transcribe)
    "Брак, с вами Редактор субтитров А.Синецкая, Корректор А.Егорова",  # classic Whisper subtitle hallucination
    "그렇지 않아.",
])
def test_drops_whisper_failures(text):
    assert clean_transcript(text, PROMPT, 'en') == ''


@pytest.mark.parametrize("text", [
    "Hello, Adrianne, how are you doing?",
    "Mr. Whiterun Guard, what is your name?",
    "Can we go somewhere private?",
    "Okay, Adrianne Avenicci, I'll be right back.",
    "No, no, no, no.",  # short real repetition is fine
    "Yes.",
    "I think it's time for you to call the guards, because if you don't, I will destroy the whole city.",
])
def test_keeps_real_speech(text):
    assert clean_transcript(text, PROMPT, 'en') == text


def test_other_scripts_allowed_for_non_latin_languages():
    assert clean_transcript("Привет, Адриана.", PROMPT, 'ru') == "Привет, Адриана."


def test_empty_and_no_prompt():
    assert clean_transcript('', PROMPT, 'en') == ''
    assert clean_transcript('Hello there.', '', 'en') == 'Hello there.'


def test_mlx_whisper_uses_greedy_decoding_and_repo_mapping(monkeypatch):
    calls = []
    fake = types.ModuleType('mlx_whisper')
    fake.transcribe = lambda audio, **kwargs: calls.append(kwargs) or {'text': ' Hello there. '}
    monkeypatch.setitem(sys.modules, 'mlx_whisper', fake)

    model = MlxWhisper('small')
    assert model.repo == MLX_REPOS['small']
    assert len(calls) == 1  # warm-up at load time

    text = model.transcribe(np.zeros(1600, dtype=np.float32), language='en', task='transcribe', prompt=PROMPT)
    assert text == 'Hello there.'
    kwargs = calls[-1]
    assert kwargs['path_or_hf_repo'] == 'mlx-community/whisper-small-mlx'
    assert kwargs['temperature'] == 0.0  # no temperature fallback: that is what loops on noisy clips
    assert kwargs['condition_on_previous_text'] is False
    assert kwargs['initial_prompt'] == PROMPT


def test_mlx_whisper_accepts_custom_repo(monkeypatch):
    fake = types.ModuleType('mlx_whisper')
    fake.transcribe = lambda audio, **kwargs: {'text': ''}
    monkeypatch.setitem(sys.modules, 'mlx_whisper', fake)
    assert MlxWhisper('someone/whisper-custom-mlx').repo == 'someone/whisper-custom-mlx'
