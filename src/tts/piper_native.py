"""In-process Piper (the `piper-tts` package) as a drop-in for Mantella's piper.exe.

Loads the same `.onnx` voice models and writes the voiceline WAV directly, without the stdin
protocol or a separate process. On macOS this also removes Wine from the speech path:
about 0.05 s per voiceline instead of about 0.5 s.
"""
from collections import OrderedDict
from pathlib import Path
import wave
import src.utils as utils

logger = utils.get_logger()


def is_available() -> bool:
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


class NativePiper:
    MAX_LOADED_VOICES = 4  # about 65 MB each; conversations rarely switch between more voices than this

    def __init__(self, models_path: Path):
        from piper import PiperVoice  # imported lazily: optional dependency
        self.__piper_voice = PiperVoice
        self.__models_path = Path(models_path)
        self.__voices: OrderedDict[str, object] = OrderedDict()
        self.__current: str | None = None

    def load_voice(self, voice: str):
        if voice not in self.__voices:
            self.__voices[voice] = self.__piper_voice.load(self.__models_path / f'{voice}.onnx')
            if len(self.__voices) > self.MAX_LOADED_VOICES:
                self.__voices.popitem(last=False)
        self.__voices.move_to_end(voice)
        self.__current = voice

    def synthesize(self, text: str, wav_path: str):
        if not self.__current:
            raise RuntimeError('No Piper voice loaded')
        with wave.open(wav_path, 'wb') as wav_file:
            self.__voices[self.__current].synthesize_wav(text, wav_file)
