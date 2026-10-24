import os
import wave
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from src.tts import piper_native
from src.tts.piper import Piper
from src.tts.synthesization_options import SynthesizationOptions
from tests.tts.test_piper import _make_mock_piper

# Real Mantella voice model, if this machine has the game installed (skipped elsewhere)
MODELS = Path(os.path.expanduser('~/Games/skyrim/versions/1.6.1170/Data/SKSE/Plugins/MantellaSoftware/piper/models/skyrim/low'))


def test_native_synthesize_writes_wav_with_cleaned_text(tmp_path):
    piper = _make_mock_piper()
    native = MagicMock()
    piper._Piper__native = native
    out = tmp_path / 'out.wav'

    piper.tts_synthesize('Hello!\n*She waves.*', str(out), SynthesizationOptions(False, False))

    native.synthesize.assert_called_once_with('Hello. She waves.', str(out))
    piper.process.stdin.write.assert_not_called()  # piper.exe is not used


def test_native_change_voice_loads_model_without_piper_exe():
    piper = _make_mock_piper()
    native = MagicMock()
    piper._Piper__native = native
    piper._select_voice_type = MagicMock(return_value='femaleeventoned')
    piper._current_actor_gender = piper._current_actor_race = None

    piper.change_voice('FemaleEvenToned')

    native.load_voice.assert_called_once_with('femaleeventoned')
    assert piper._last_voice == 'femaleeventoned'
    assert piper._Piper__waiting_for_voice_load is False
    piper.process.stdin.write.assert_not_called()


def test_native_failure_falls_back_to_piper_exe(tmp_path):
    piper = _make_mock_piper()
    native = MagicMock()
    native.synthesize.side_effect = RuntimeError('boom')
    piper._Piper__native = native
    piper._run_piper = MagicMock()
    piper._check_voice_changed = MagicMock(return_value=True)
    out = tmp_path / 'out.wav'
    def write_wav(*_):  # piper.exe "writes" the file after the synthesize command
        with wave.open(str(out), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b'\0\0' * 1600)
    piper.process.stdin.write.side_effect = write_wav

    piper.tts_synthesize('Hello.', str(out), SynthesizationOptions(False, False))

    assert piper._Piper__native is None
    piper._run_piper.assert_called_once()


def test_voice_cache_is_bounded(monkeypatch):
    loads = []
    class FakeVoice:
        @staticmethod
        def load(path):
            loads.append(Path(path).stem); return object()
    import types, sys
    monkeypatch.setitem(sys.modules, 'piper', types.SimpleNamespace(PiperVoice=FakeVoice))
    native = piper_native.NativePiper(Path('/models'))
    for voice in ['a', 'b', 'a', 'c', 'd', 'e', 'a']:
        native.load_voice(voice)
    assert loads == ['a', 'b', 'c', 'd', 'e']  # 'a' stays cached because it was used recently


@pytest.mark.skipif(not (piper_native.is_available() and (MODELS / 'femaleeventoned.onnx').exists()), reason='needs piper-tts and Mantella voice models')
def test_real_model_synthesis(tmp_path):
    native = piper_native.NativePiper(MODELS)
    native.load_voice('femaleeventoned')
    out = tmp_path / 'out.wav'
    native.synthesize('I am sworn to carry your burdens.', str(out))
    with wave.open(str(out), 'rb') as w:
        assert w.getnframes() / w.getframerate() > 1.0
