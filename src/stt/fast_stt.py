"""Faster, cleaner speech-to-text helpers.

- MlxWhisper: runs Whisper on the Apple Silicon GPU via mlx-whisper (select `process_device = mlx`).
- clean_transcript: drops common Whisper failure modes (echoed prompt, repetition loops,
  output in the wrong script) so they are not sent to the NPC as player speech.
"""
import difflib
import re
import unicodedata
import numpy as np
import src.utils as utils

logger = utils.get_logger()

# whisper_model_size values -> mlx-community repos. Anything else is used as a Hugging Face repo id as is.
MLX_REPOS = {
    'tiny': 'mlx-community/whisper-tiny-mlx', 'tiny.en': 'mlx-community/whisper-tiny.en-mlx',
    'base': 'mlx-community/whisper-base-mlx', 'base.en': 'mlx-community/whisper-base.en-mlx',
    'small': 'mlx-community/whisper-small-mlx', 'small.en': 'mlx-community/whisper-small.en-mlx',
    'medium': 'mlx-community/whisper-medium-mlx', 'medium.en': 'mlx-community/whisper-medium.en-mlx',
    'large-v3': 'mlx-community/whisper-large-v3-mlx', 'large-v3-turbo': 'mlx-community/whisper-large-v3-turbo',
    'turbo': 'mlx-community/whisper-large-v3-turbo',
}

# Languages written in Latin script: output in another script means Whisper misdetected the language
LATIN_LANGUAGES = {'en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'pl', 'cs', 'sv', 'da', 'no', 'fi', 'tr', 'ro', 'hu'}


class MlxWhisper:
    """Whisper on the Apple Silicon GPU. Decoding is greedy without temperature fallback:
    the fallback is what makes short or noisy clips loop for seconds."""

    def __init__(self, model_size: str):
        import mlx_whisper  # only needed on Apple Silicon, so import lazily
        self.__mlx_whisper = mlx_whisper
        self.repo = MLX_REPOS.get(model_size, model_size)
        logger.log(27, f'Loading Whisper on Apple GPU (MLX): {self.repo}')
        # Warm up once so the first real transcription does not pay for model loading and kernel compilation
        self.transcribe(np.zeros(16000, dtype=np.float32), language='en', task='transcribe', prompt='')

    def transcribe(self, audio: np.ndarray, language: str, task: str, prompt: str) -> str:
        result = self.__mlx_whisper.transcribe(
            audio, path_or_hf_repo=self.repo, language=language, task=task,
            temperature=0.0, condition_on_previous_text=False,
            initial_prompt=prompt or None, no_speech_threshold=0.6, fp16=True, verbose=None)
        return result.get('text', '').strip()


def _normalise(text: str) -> str:
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.lower()).split())


def _is_prompt_echo(text: str, prompt: str) -> bool:
    """Whisper (local and API) sometimes returns its context prompt when the audio is silence or noise."""
    text_n, prompt_n = _normalise(text), _normalise(prompt)
    if not text_n or not prompt_n:
        return False
    if text_n in prompt_n or difflib.SequenceMatcher(None, text_n, prompt_n).ratio() > 0.75:
        return True
    words = text_n.split()
    prompt_words = set(prompt_n.split())
    # a list of names from the prompt (eg "Vaska, Adrianne Avenicci, Alfhild Battle-Born") with nothing else
    return len(words) >= 3 and sum(word in prompt_words for word in words) / len(words) > 0.8


def _is_repetition_loop(text: str) -> bool:
    """The same word 5+ or phrase 4+ times in a row ("Vaska, Vaska, Vaska, ...", "Mane-Mane-Mane-Mane")."""
    words = _normalise(text).split()
    for n in (1, 2, 3):
        for start in range(len(words)):
            gram = words[start:start + n]
            repeats = 1
            while words[start + repeats * n:start + (repeats + 1) * n] == gram:
                repeats += 1
            if len(gram) == n and repeats >= (5 if n == 1 else 4):  # "no, no, no, no" is real speech
                return True
    return False


def _is_wrong_script(text: str, language: str) -> bool:
    if language not in LATIN_LANGUAGES:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    non_latin = sum('LATIN' not in unicodedata.name(c, '') for c in letters)
    return non_latin / len(letters) > 0.3


def clean_transcript(text: str, prompt: str, language: str) -> str:
    """Returns the transcript, or '' if it is a known Whisper failure rather than real player speech."""
    if not text:
        return ''
    if _is_prompt_echo(text, prompt):
        logger.debug(f"Dropped transcript that echoes the STT prompt: '{text}'")
        return ''
    if _is_repetition_loop(text):
        logger.debug(f"Dropped transcript with a repetition loop: '{text}'")
        return ''
    if _is_wrong_script(text, language):
        logger.debug(f"Dropped transcript in the wrong script for language '{language}': '{text}'")
        return ''
    return text
