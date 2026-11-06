"""Tests for Parakeet TDT integration."""

import ast
import inspect
import sys
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import numpy as np
import pytest

from debora_whisper.dictation_engine import (
    MODEL_REGISTRY, DictationApp, DEFAULT_CONFIG,
    create_model, ParakeetNPU, WhisperNPU,
    LANGUAGES, PARAKEET_UPSTREAM_LANGUAGES,
    get_models_for_language, is_model_downloaded, MODEL_DIR,
)


class TestParakeetRegistry:
    """Verify parakeet is correctly registered."""

    def test_parakeet_in_registry(self):
        assert "parakeet" in MODEL_REGISTRY

    def test_parakeet_has_required_fields(self):
        info = MODEL_REGISTRY["parakeet"]
        assert info["backend"] == "parakeet"
        assert info["preferred_device"] == "NPU"
        assert "ov_repo" in info
        assert "local_dir" in info
        assert "description" in info

    def test_all_models_have_backend_field(self):
        for name, info in MODEL_REGISTRY.items():
            assert "backend" in info, f"{name} missing backend field"
            assert info["backend"] in ("whisper", "parakeet"), f"{name} has invalid backend"

    def test_all_models_have_local_dir_field(self):
        for name, info in MODEL_REGISTRY.items():
            assert "local_dir" in info, f"{name} missing local_dir field"


class TestCreateModel:
    """Verify factory function dispatches correctly."""

    @patch("debora_whisper.dictation_engine.WhisperNPU")
    def test_creates_whisper_for_whisper_backend(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = create_model(Path("/fake"), device="NPU", backend="whisper")
        mock_cls.assert_called_once_with(Path("/fake"), device="NPU")

    @patch("debora_whisper.dictation_engine.ParakeetNPU")
    def test_creates_parakeet_for_parakeet_backend(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = create_model(Path("/fake"), device="NPU", backend="parakeet")
        mock_cls.assert_called_once_with(Path("/fake"), device="NPU")


class TestEnsureModelDispatch:
    """Verify ensure_model uses factory pattern."""

    @patch("debora_whisper.dictation_engine.create_model")
    @patch("debora_whisper.dictation_engine.setup_model")
    def test_ensure_model_parakeet(self, mock_setup, mock_create):
        mock_setup.return_value = Path("/fake/parakeet")
        mock_create.return_value = MagicMock()

        config = {**DEFAULT_CONFIG, "model_size": "parakeet", "device": "NPU"}
        app = DictationApp(config)
        app.ensure_model()

        mock_create.assert_called_once_with(
            Path("/fake/parakeet"), device="NPU", backend="parakeet",
            model_size="parakeet"
        )

    @patch("debora_whisper.dictation_engine.create_model")
    @patch("debora_whisper.dictation_engine.setup_model")
    def test_ensure_model_whisper(self, mock_setup, mock_create):
        mock_setup.return_value = Path("/fake/whisper")
        mock_create.return_value = MagicMock()

        config = {**DEFAULT_CONFIG, "model_size": "base", "device": "NPU"}
        app = DictationApp(config)
        app.ensure_model()

        mock_create.assert_called_once_with(
            Path("/fake/whisper"), device="NPU", backend="whisper",
            model_size="base"
        )


class TestParakeetConfig:
    """Verify parakeet works with config validation."""

    def test_parakeet_config_validates(self):
        from debora_whisper.dictation_engine import validate_config
        config = {**DEFAULT_CONFIG, "model_size": "parakeet", "device": "NPU"}
        # Should not raise
        validate_config(config)

    def test_invalid_model_still_rejected(self):
        from debora_whisper.dictation_engine import validate_config
        config = {**DEFAULT_CONFIG, "model_size": "nonexistent"}
        with pytest.raises(ValueError, match="model_size"):
            validate_config(config)


class TestLanguageFiltering:
    """Verify language-based model filtering."""

    def test_whisper_tiny_is_registered_for_all_languages(self):
        info = MODEL_REGISTRY["tiny"]
        assert info["repo"] == "openai/whisper-tiny"
        assert info["ov_repo"] == "OpenVINO/whisper-tiny-int8-ov"
        assert info["backend"] == "whisper"
        assert info["preferred_device"] == "NPU"
        assert info["languages"] == "all"

    def test_all_models_have_languages_field(self):
        for name, info in MODEL_REGISTRY.items():
            assert "languages" in info, f"{name} missing languages field"

    def test_english_returns_all_models(self):
        models = get_models_for_language("en")
        assert len(models) == len(MODEL_REGISTRY)
        assert "parakeet" in models

    def test_non_european_excludes_parakeet(self):
        # Parakeet's upstream checkpoint covers 25 European languages; the
        # app also exposes ja/zh/ko/tr/ar which the checkpoint does not
        # support, so those must stay excluded.
        for lang in ("ja", "zh", "ko", "tr", "ar"):
            models = get_models_for_language(lang)
            assert "parakeet" not in models, f"parakeet should not be in {lang} models"
            assert "base" in models
            assert "small" in models

    def test_non_english_includes_parakeet_for_supported_languages(self):
        # Regression test for issue #1: Parakeet must be offered for at
        # least one non-English language it actually supports.
        for lang in ("ru", "es", "fr", "de", "pt", "it", "nl", "pl", "uk"):
            models = get_models_for_language(lang)
            assert "parakeet" in models, f"parakeet should be offered for {lang}"

    def test_whisper_models_support_all_languages(self):
        for name, info in MODEL_REGISTRY.items():
            if info["backend"] == "whisper":
                assert info["languages"] == "all", f"{name} should support all languages"

    def test_parakeet_is_multilingual(self):
        info = MODEL_REGISTRY["parakeet"]
        assert isinstance(info["languages"], list)
        assert "en" in info["languages"]
        assert len(info["languages"]) > 1, "parakeet should no longer be English-only"

    def test_parakeet_languages_subset_of_upstream_checkpoint(self):
        info = MODEL_REGISTRY["parakeet"]
        for lang in info["languages"]:
            assert lang in PARAKEET_UPSTREAM_LANGUAGES, (
                f"{lang} is declared for parakeet but the upstream checkpoint "
                f"does not support it"
            )

    def test_parakeet_languages_selectable_in_ui(self):
        # Regression test for issue #1: the registry must never declare a
        # language the UI's LANGUAGES map (and therefore the model picker)
        # cannot select.
        info = MODEL_REGISTRY["parakeet"]
        for lang in info["languages"]:
            assert lang in LANGUAGES, (
                f"{lang} is declared for parakeet but is not in LANGUAGES "
                f"(the UI cannot select it)"
            )

    def test_languages_dict_has_entries(self):
        assert len(LANGUAGES) >= 10
        assert "en" in LANGUAGES
        assert LANGUAGES["en"] == "English"


class TestParakeetTdtConstants:
    """Verify BLANK_IDX / VOCAB_SIZE are derived from the loaded vocab
    rather than trusted blindly (issue #1)."""

    def _bare_parakeet(self, tmp_path):
        # Bypass __init__ (which loads onnxruntime/OpenVINO models) to unit
        # test _load_vocab() in isolation.
        instance = ParakeetNPU.__new__(ParakeetNPU)
        instance.model_path = tmp_path
        instance.vocab = {}
        return instance

    def test_derives_constants_matching_current_checkpoint_vocab(self, tmp_path):
        # 8192 tokens (indices 0-8191) matches the shipped
        # goodsmileduck/parakeet-tdt-0.6b-v3-onnx vocab.txt.
        lines = [f"tok{i} {i}" for i in range(8192)]
        (tmp_path / "vocab.txt").write_text("\n".join(lines), encoding="utf-8")
        instance = self._bare_parakeet(tmp_path)
        instance._load_vocab()
        assert instance.BLANK_IDX == 8192
        assert instance.VOCAB_SIZE == 8193

    def test_derives_constants_for_a_different_sized_vocab(self, tmp_path):
        # A hypothetical re-export with a smaller vocab must not silently
        # keep using the old checkpoint's hardcoded constants.
        lines = [f"tok{i} {i}" for i in range(100)]
        (tmp_path / "vocab.txt").write_text("\n".join(lines), encoding="utf-8")
        instance = self._bare_parakeet(tmp_path)
        instance._load_vocab()
        assert instance.BLANK_IDX == 100
        assert instance.VOCAB_SIZE == 101

    def test_empty_vocab_raises_instead_of_silently_falling_back(self, tmp_path):
        # An empty vocab.txt must not silently keep the hardcoded 8192/8193
        # defaults paired with an empty self.vocab dict -- that would "load"
        # successfully and then render every token as "?" at inference time.
        (tmp_path / "vocab.txt").write_text("", encoding="utf-8")
        instance = self._bare_parakeet(tmp_path)
        with pytest.raises(ValueError, match="zero valid entries"):
            instance._load_vocab()

    def test_unparseable_vocab_raises_instead_of_silently_falling_back(self, tmp_path):
        # Every line fails the "token index" pair format (e.g. a corrupted
        # download truncated mid-line) -- self.vocab ends up empty exactly
        # like the fully-empty-file case above.
        (tmp_path / "vocab.txt").write_text(
            "this is not a valid vocab line\nneither is this",
            encoding="utf-8",
        )
        instance = self._bare_parakeet(tmp_path)
        with pytest.raises(ValueError, match="zero valid entries"):
            instance._load_vocab()


class TestParakeetVocabDecoderValidation:
    """Verify the vocab-derived VOCAB_SIZE is cross-checked against the
    decoder's real compiled output width (fast follow-up to issue #1's
    review: `@sec` medium finding)."""

    def _bare_parakeet_with_decoder(self, tmp_path, decoder_output_width, vocab_size, blank_idx):
        instance = ParakeetNPU.__new__(ParakeetNPU)
        instance.model_path = tmp_path
        instance.vocab = {i: str(i) for i in range(vocab_size - 1)}
        instance.VOCAB_SIZE = vocab_size
        instance.BLANK_IDX = blank_idx

        mock_output = MagicMock()
        mock_output.get_partial_shape.return_value = [MagicMock(get_length=lambda: decoder_output_width)]
        instance.dec_compiled = MagicMock()
        instance.dec_compiled.output.return_value = mock_output
        return instance

    def test_matching_decoder_output_width_passes(self, tmp_path):
        # VOCAB_SIZE=8193 (8192 vocab tokens + blank) plus 5 duration bins
        # matches decoder_joint-model.onnx's real output width for the
        # shipped checkpoint.
        instance = self._bare_parakeet_with_decoder(
            tmp_path, decoder_output_width=8198, vocab_size=8193, blank_idx=8192,
        )
        instance._validate_vocab_against_decoder()  # should not raise

    def test_vocab_larger_than_decoder_output_raises(self, tmp_path):
        # A truncated/stale vocab.txt derives a VOCAB_SIZE that leaves no
        # room for duration logits in the decoder's real output -- this must
        # fail loudly at load time instead of crashing argmax() on an empty
        # slice mid-transcription.
        instance = self._bare_parakeet_with_decoder(
            tmp_path, decoder_output_width=8198, vocab_size=8300, blank_idx=8299,
        )
        with pytest.raises(RuntimeError, match="inconsistent with decoder_joint-model.onnx"):
            instance._validate_vocab_against_decoder()

    def test_vocab_size_equal_to_decoder_output_raises(self, tmp_path):
        # Leaves zero duration logits (empty slice) -- must also be rejected.
        instance = self._bare_parakeet_with_decoder(
            tmp_path, decoder_output_width=8193, vocab_size=8193, blank_idx=8192,
        )
        with pytest.raises(RuntimeError, match="inconsistent with decoder_joint-model.onnx"):
            instance._validate_vocab_against_decoder()


class TestModelDownloadStatus:
    """Verify download status detection."""

    def test_nonexistent_model_not_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("debora_whisper.dictation_engine.MODEL_DIR", tmp_path)
        assert is_model_downloaded("base") is False

    def test_empty_dir_not_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("debora_whisper.dictation_engine.MODEL_DIR", tmp_path)
        (tmp_path / "whisper-base-openvino").mkdir()
        assert is_model_downloaded("base") is False

    def test_dir_with_xml_is_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("debora_whisper.dictation_engine.MODEL_DIR", tmp_path)
        model_dir = tmp_path / "whisper-base-openvino"
        model_dir.mkdir()
        (model_dir / "model.xml").write_text("")
        (model_dir / "model.bin").write_bytes(b"\0")
        assert is_model_downloaded("base") is True

    def test_interrupted_download_is_not_downloaded(self, tmp_path, monkeypatch):
        """The .xml files arrive first; missing or empty weights mean the
        download stopped and must resume, not load."""
        monkeypatch.setattr("debora_whisper.dictation_engine.MODEL_DIR", tmp_path)
        model_dir = tmp_path / "whisper-base-openvino"
        model_dir.mkdir()
        for name in ("encoder", "decoder"):
            (model_dir / f"{name}.xml").write_text("<net/>")
        (model_dir / "encoder.bin").write_bytes(b"\0")
        assert is_model_downloaded("base") is False
        (model_dir / "decoder.bin").write_bytes(b"")
        assert is_model_downloaded("base") is False

    def test_dir_with_onnx_is_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("debora_whisper.dictation_engine.MODEL_DIR", tmp_path)
        model_dir = tmp_path / "parakeet-tdt-openvino"
        model_dir.mkdir()
        (model_dir / "encoder-model.onnx").write_text("")
        assert is_model_downloaded("parakeet") is True


class TestShapeBucketing:
    """Regression tests for issue #3 shape-bucket selection (no NPU needed —
    select_bucket() and the padding logic are pure/deterministic)."""

    def test_bucket_set_is_ascending_and_nonempty(self):
        buckets = ParakeetNPU.MEL_BUCKETS
        assert len(buckets) >= 1
        assert list(buckets) == sorted(buckets)
        assert len(set(buckets)) == len(buckets)

    def test_smallest_audio_gets_smallest_bucket(self):
        bucket, truncated = ParakeetNPU.select_bucket(1)
        assert bucket == ParakeetNPU.MEL_BUCKETS[0]
        assert truncated is False

    def test_exact_boundary_picks_that_bucket_not_the_next(self):
        for bucket in ParakeetNPU.MEL_BUCKETS:
            chosen, truncated = ParakeetNPU.select_bucket(bucket)
            assert chosen == bucket, f"exact-length input {bucket} should pick bucket {bucket}, got {chosen}"
            assert truncated is False

    def test_one_frame_over_boundary_picks_next_bucket(self):
        buckets = ParakeetNPU.MEL_BUCKETS
        for smaller, larger in zip(buckets, buckets[1:]):
            chosen, truncated = ParakeetNPU.select_bucket(smaller + 1)
            assert chosen == larger, f"{smaller + 1} frames should round up to {larger}, got {chosen}"
            assert truncated is False

    def test_over_length_audio_falls_back_to_largest_bucket_and_flags_truncation(self):
        largest = ParakeetNPU.MEL_BUCKETS[-1]
        chosen, truncated = ParakeetNPU.select_bucket(largest + 1)
        assert chosen == largest
        assert truncated is True

        chosen, truncated = ParakeetNPU.select_bucket(largest * 10)
        assert chosen == largest
        assert truncated is True

    def test_zero_frames_picks_smallest_bucket(self):
        chosen, truncated = ParakeetNPU.select_bucket(0)
        assert chosen == ParakeetNPU.MEL_BUCKETS[0]
        assert truncated is False


class TestMelPaddingToBucket:
    """Verify mel features end up padded to exactly the selected bucket's
    frame count, via ParakeetNPU.pad_to_bucket() — the actual method
    transcribe() and the benchmark harness call, not a reimplementation.
    """

    def test_short_audio_padded_to_exact_bucket_size(self):
        actual_frames = 50
        mel = np.ones((1, 128, actual_frames), dtype=np.float32)
        bucket, truncated = ParakeetNPU.select_bucket(actual_frames)
        assert truncated is False
        padded = ParakeetNPU.pad_to_bucket(mel, bucket)
        assert padded.shape == (1, 128, bucket)
        # original content preserved
        assert np.array_equal(padded[:, :, :actual_frames], mel)
        # padding is zeros
        assert np.all(padded[:, :, actual_frames:] == 0)

    def test_exact_bucket_length_unchanged(self):
        bucket = ParakeetNPU.MEL_BUCKETS[1]
        mel = np.ones((1, 128, bucket), dtype=np.float32)
        chosen, truncated = ParakeetNPU.select_bucket(bucket)
        assert chosen == bucket
        assert truncated is False
        padded = ParakeetNPU.pad_to_bucket(mel, chosen)
        assert padded.shape == (1, 128, bucket)
        assert np.array_equal(padded, mel)

    def test_over_length_audio_truncated_to_largest_bucket_not_dropped_silently(self):
        largest = ParakeetNPU.MEL_BUCKETS[-1]
        actual_frames = largest + 500
        mel = np.arange(actual_frames, dtype=np.float32).reshape(1, 1, -1)
        mel = np.broadcast_to(mel, (1, 128, actual_frames)).copy()
        bucket, truncated = ParakeetNPU.select_bucket(actual_frames)
        assert bucket == largest
        assert truncated is True  # caller (transcribe()) must log this, not swallow it
        result = ParakeetNPU.pad_to_bucket(mel, bucket)
        assert result.shape == (1, 128, largest)
        # the leading `largest` frames of real speech are kept, not zeroed/dropped
        assert np.array_equal(result, mel[:, :, :largest])


class TestParakeetMissingConstantsRegression:
    """Regression test for issue #9: commit 2dc9cd4 deleted ENC_DIM,
    LSTM_DIM and DECODE_SPACE from ParakeetNPU's class body while their
    uses in _load_pipeline() (encoder/decoder compile) and
    _tdt_greedy_decode() survived, crashing real-hardware loads with
    'ParakeetNPU' object has no attribute 'ENC_DIM'.

    This exercises the real _load_pipeline (compile) and
    _tdt_greedy_decode (decode) code paths with mocked OpenVINO /
    onnxruntime handles -- not hasattr(ParakeetNPU, "ENC_DIM"), which
    would pass against a wrong value and would never touch DECODE_SPACE.
    """

    def _fake_model_dir(self, tmp_path):
        # ▁ (U+2581) is NeMo/SentencePiece's word-start marker; _load_vocab
        # turns it into a literal leading space.
        (tmp_path / "vocab.txt").write_text(
            "▁hello 0\n▁world 1\n▁c 2\n", encoding="utf-8",
        )
        (tmp_path / "nemo128.onnx").write_bytes(b"")
        (tmp_path / "encoder-model.onnx").write_bytes(b"")
        (tmp_path / "decoder_joint-model.onnx").write_bytes(b"")
        return tmp_path

    def test_load_pipeline_and_decode_use_real_enc_lstm_decode_space_constants(
        self, tmp_path, monkeypatch,
    ):
        model_dir = self._fake_model_dir(tmp_path)
        # One bucket is enough to exercise the compile path; MEL_BUCKETS
        # itself is unrelated to this regression.
        monkeypatch.setattr(ParakeetNPU, "MEL_BUCKETS", (10,))

        fake_encoder_model = MagicMock(name="encoder_model")
        fake_decoder_model = MagicMock(name="decoder_model")

        def fake_read_model(path):
            return fake_decoder_model if "decoder_joint" in str(path) else fake_encoder_model

        fake_enc_compiled = MagicMock(name="enc_compiled")
        fake_enc_compiled.side_effect = lambda inputs: {
            "outputs": np.zeros((1, 1024, 3), dtype=np.float32)
        }

        fake_dec_compiled = MagicMock(name="dec_compiled")
        vocab_size = 4  # 3 real tokens (indices 0-2) + derived blank at 3
        duration_bins = 2
        mock_output = MagicMock()
        mock_output.get_partial_shape.return_value = [
            MagicMock(get_length=lambda: vocab_size + duration_bins)
        ]
        fake_dec_compiled.output.return_value = mock_output

        def fake_compile_model(model, device, cfg=None):
            return fake_dec_compiled if model is fake_decoder_model else fake_enc_compiled

        fake_core = MagicMock(name="core")
        fake_core.read_model.side_effect = fake_read_model
        fake_core.compile_model.side_effect = fake_compile_model

        fake_ov = MagicMock(name="openvino_module")
        fake_ov.Core.return_value = fake_core

        with patch.dict(sys.modules, {"openvino": fake_ov}), \
             patch("onnxruntime.InferenceSession", return_value=MagicMock()):
            instance = ParakeetNPU(model_dir, device="CPU")

        # --- compile path: ENC_DIM / LSTM_DIM must have been used with
        # their real pre-2dc9cd4 values, not raised AttributeError. ---
        fake_decoder_model.reshape.assert_called_once_with({
            "encoder_outputs": [1, 1024, 1],
            "targets": [1, 1],
            "target_length": [1],
            "input_states_1": [2, 1, 640],
            "input_states_2": [2, 1, 640],
        })

        # --- decode path: LSTM_DIM (state shape) and DECODE_SPACE (final
        # text cleanup) must both be exercised through the real method. ---
        step_outputs = [
            {  # t=0: emit token 0 ("▁hello"), duration=1
                "outputs": np.array([[10, 0, 0, 0, 0, 10]], dtype=np.float32),
                "output_states_1": np.zeros((2, 1, 640), dtype=np.float32),
                "output_states_2": np.zeros((2, 1, 640), dtype=np.float32),
            },
            {  # t=1: emit token 1 ("▁world"), duration=1
                "outputs": np.array([[0, 10, 0, 0, 0, 10]], dtype=np.float32),
                "output_states_1": np.zeros((2, 1, 640), dtype=np.float32),
                "output_states_2": np.zeros((2, 1, 640), dtype=np.float32),
            },
        ]
        instance.dec_compiled = MagicMock(side_effect=step_outputs)

        enc_out = np.zeros((1, 1024, 2), dtype=np.float32)
        text = instance._tdt_greedy_decode(enc_out, enc_len=2)

        # A missing/wrong DECODE_SPACE would leave the leading/duplicated
        # sentencepiece-marker spaces in the output instead of "hello world".
        assert text == "hello world"

    def test_mutation_delete_enc_dim_breaks_the_regression_test(self, tmp_path, monkeypatch):
        """Companion assertion proving the test above is load-bearing: with
        ENC_DIM removed from the class (simulating the 2dc9cd4 regression),
        constructing ParakeetNPU raises AttributeError instead of silently
        passing."""
        model_dir = self._fake_model_dir(tmp_path)
        monkeypatch.setattr(ParakeetNPU, "MEL_BUCKETS", (10,))
        monkeypatch.delattr(ParakeetNPU, "ENC_DIM")

        fake_ov = MagicMock(name="openvino_module")
        fake_core = MagicMock(name="core")
        fake_core.read_model.return_value = MagicMock()
        fake_enc_compiled = MagicMock()
        fake_enc_compiled.side_effect = lambda inputs: {
            "outputs": np.zeros((1, 1024, 3), dtype=np.float32)
        }
        fake_core.compile_model.return_value = fake_enc_compiled
        fake_ov.Core.return_value = fake_core

        with patch.dict(sys.modules, {"openvino": fake_ov}), \
             patch("onnxruntime.InferenceSession", return_value=MagicMock()):
            with pytest.raises(AttributeError, match="ENC_DIM"):
                ParakeetNPU(model_dir, device="CPU")


class TestParakeetClassAttributeGuard:
    """Cheap static guard for issue #9's defect class: a class attribute
    (self.X / cls.X) that is loaded somewhere in ParakeetNPU but never
    defined at class level nor assigned on the instance anywhere in the
    class. Walks the class body with `ast` so this stays true even as the
    class grows, without needing OpenVINO/onnxruntime installed."""

    def _undefined_self_cls_attributes(self):
        source = inspect.getsource(ParakeetNPU)
        tree = ast.parse(source)
        class_node = tree.body[0]
        assert isinstance(class_node, ast.ClassDef)

        def _targets(node):
            if isinstance(node, ast.Assign):
                return node.targets
            if isinstance(node, ast.AnnAssign):
                return [node.target]
            if isinstance(node, ast.AugAssign):
                return [node.target]
            return []

        class_level_names = set()
        for stmt in class_node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                class_level_names.add(stmt.name)
                continue
            for target in _targets(stmt):
                if isinstance(target, ast.Name):
                    class_level_names.add(target.id)
                elif isinstance(target, ast.Tuple):
                    class_level_names.update(
                        elt.id for elt in target.elts if isinstance(elt, ast.Name)
                    )

        instance_assigned_names = set()
        loaded_names = set()
        for node in ast.walk(class_node):
            for target in _targets(node):
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
                        and target.value.id in ("self", "cls"):
                    instance_assigned_names.add(target.attr)
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) \
                    and isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
                loaded_names.add(node.attr)

        known_names = class_level_names | instance_assigned_names
        return loaded_names - known_names

    def test_no_undefined_self_or_cls_attribute_loads(self):
        undefined = self._undefined_self_cls_attributes()
        assert not undefined, (
            f"ParakeetNPU reads self./cls. attribute(s) never defined at "
            f"class level or assigned on the instance: {sorted(undefined)} "
            f"-- this is exactly the class of bug that made ENC_DIM/"
            f"LSTM_DIM/DECODE_SPACE disappear in 2dc9cd4 while their uses "
            f"survived."
        )
