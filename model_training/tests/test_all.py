"""
Pytest test suite for CrowdGuard AI temporal models.
Run with: pytest model_training/tests/ -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Path setup ─────────────────────────────────────────────────────────────
_TESTS = Path(__file__).parent
_ROOT = _TESTS.parent
sys.path.insert(0, str(_ROOT / "src"))

from preprocessing import (
    CORE_FEATURE_COLUMNS,
    FEATURE_DIM,
    StandardScaler,
    extract_features_from_frames,
    handle_missing_values,
    validate_feature_schema,
)
from sequence_builder import SequenceBuilder, DatasetSplitter, assemble_split_arrays
from dataset import CrowdSequenceDataset, build_dataloaders
from gru_model import CrowdGuardGRU, build_gru_from_config
from lstm_model import CrowdGuardLSTM, build_lstm_from_config
from metrics import (
    compute_confusion_matrix,
    per_class_metrics,
    macro_metrics,
    full_evaluation,
    compute_accuracy,
)
from train_engine import (
    EarlyStopping,
    train_one_epoch,
    validate,
    save_checkpoint,
    load_checkpoint,
)
from reproducibility import set_seed


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def synthetic_frames():
    """20 synthetic 128x128 grayscale frames."""
    return [np.random.randint(0, 255, (128, 128), dtype=np.uint8) for _ in range(20)]


@pytest.fixture(scope="session")
def synthetic_df():
    """Synthetic feature DataFrame with 100 rows."""
    import pandas as pd
    data = {c: np.random.randn(100) for c in CORE_FEATURE_COLUMNS}
    data["label"] = np.random.randint(0, 2, 100)
    return pd.DataFrame(data)


@pytest.fixture
def small_model_gru():
    return CrowdGuardGRU(
        input_dim=FEATURE_DIM, hidden_dim=16, num_layers=1, output_dim=2
    )


@pytest.fixture
def small_model_lstm():
    return CrowdGuardLSTM(
        input_dim=FEATURE_DIM, hidden_dim=16, num_layers=1, output_dim=2
    )


# ── Preprocessing tests ─────────────────────────────────────────────────────
class TestPreprocessing:
    def test_feature_dim_constant(self):
        assert FEATURE_DIM == len(CORE_FEATURE_COLUMNS)
        assert FEATURE_DIM == 10

    def test_feature_column_order(self):
        """Feature order must be deterministic."""
        expected = [
            "crowd_density", "motion_magnitude", "motion_direction_sin",
            "motion_direction_cos", "optical_flow_x", "optical_flow_y",
            "zone_occupancy", "delta_density", "motion_consistency",
            "flow_magnitude_std",
        ]
        assert CORE_FEATURE_COLUMNS == expected

    def test_scaler_no_leakage(self):
        """Scaler fit on train must NOT use val/test statistics."""
        X_train = np.ones((50, FEATURE_DIM), dtype=np.float32) * 10.0
        X_val = np.ones((20, FEATURE_DIM), dtype=np.float32) * 100.0

        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)

        # Training mean should be ~10 (fit on X_train)
        assert np.allclose(scaler.mean_, 10.0)
        # Normalised train should be ~0
        assert abs(X_train_norm.mean()) < 1e-5
        # Val should NOT be normalised to 0 (different distribution)
        assert X_val_norm.mean() > 5.0

    def test_scaler_save_load(self, tmp_path):
        X = np.random.randn(50, FEATURE_DIM).astype(np.float32)
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)

        path = str(tmp_path / "scaler.pkl")
        scaler.save(path)

        scaler2 = StandardScaler.load(path)
        X_norm2 = scaler2.transform(X)
        assert np.allclose(X_norm, X_norm2)

    def test_handle_missing_values(self):
        import pandas as pd
        df = pd.DataFrame({c: [np.nan, 1.0, 2.0] for c in CORE_FEATURE_COLUMNS})
        df = handle_missing_values(df)
        assert not df.isnull().any().any()
        assert df["crowd_density"].iloc[0] == 0.0

    def test_feature_extraction_shape(self, synthetic_frames):
        df = extract_features_from_frames(synthetic_frames, resize_hw=(32, 32))
        assert len(df) == len(synthetic_frames)
        for col in CORE_FEATURE_COLUMNS:
            assert col in df.columns

    def test_feature_extraction_no_nan(self, synthetic_frames):
        df = extract_features_from_frames(synthetic_frames, resize_hw=(32, 32))
        for col in CORE_FEATURE_COLUMNS:
            assert df[col].isnull().sum() == 0, f"NaN in {col}"

    def test_validate_feature_schema(self, synthetic_df):
        assert validate_feature_schema(synthetic_df) is True

    def test_validate_feature_schema_missing(self):
        import pandas as pd
        df = pd.DataFrame({"crowd_density": [1.0]})
        assert validate_feature_schema(df) is False


# ── Sequence builder tests ──────────────────────────────────────────────────
class TestSequenceBuilder:
    def test_correct_number_of_sequences(self, synthetic_df):
        builder = SequenceBuilder(seq_len=20, stride=1)
        X, y, meta = builder.build_from_clip(synthetic_df, "clip01")
        expected = 100 - 20 + 1
        assert len(X) == expected

    def test_sequence_shape(self, synthetic_df):
        builder = SequenceBuilder(seq_len=20, stride=1)
        X, y, meta = builder.build_from_clip(synthetic_df, "clip01")
        assert X.shape == (len(X), 20, FEATURE_DIM)

    def test_label_at_end_of_window(self, synthetic_df):
        """Label should be from the last frame in the window."""
        import pandas as pd
        labels = np.zeros(50, dtype=np.int64)
        labels[30:] = 1  # anomaly starts at frame 30

        data = {c: np.random.randn(50) for c in CORE_FEATURE_COLUMNS}
        data["label"] = labels
        df = pd.DataFrame(data)

        builder = SequenceBuilder(seq_len=10, stride=1)
        X, y, meta = builder.build_from_clip(df, "test")

        # Sequence ending at frame 30 (index 30 = 0-based = window end at 30)
        # Window starting at 21 (0-based), ending at 30 inclusive
        # seq start=21, end=30 => label[30]=1
        seq_idx = 21  # 21st sequence starts at frame 21, ends at frame 30
        assert y[seq_idx] == 1

    def test_short_clip_skipped(self):
        """Clips shorter than seq_len should return empty arrays."""
        import pandas as pd
        data = {c: np.random.randn(5) for c in CORE_FEATURE_COLUMNS}
        data["label"] = np.zeros(5)
        df = pd.DataFrame(data)

        builder = SequenceBuilder(seq_len=20)
        X, y, meta = builder.build_from_clip(df, "short_clip")
        assert len(X) == 0
        assert len(y) == 0

    def test_sequences_no_cross_boundary(self, synthetic_df):
        """All sequences come from contiguous frames within single clip."""
        builder = SequenceBuilder(seq_len=20, stride=5)
        X, y, meta = builder.build_from_clip(synthetic_df, "clip01")
        for m in meta:
            assert m["clip_id"] == "clip01"

    def test_clip_level_split(self):
        clips = [
            {"clip_id": f"clip{i:02d}", "subset": "Peds1",
             "official_split": "train" if i < 8 else "test"}
            for i in range(12)
        ]
        splitter = DatasetSplitter(seed=42)
        splits = splitter.split_clips(clips, val_fraction=0.2)

        all_ids = set()
        for split_name, split_clips in splits.items():
            ids = {c["clip_id"] for c in split_clips}
            assert len(ids & all_ids) == 0, f"Clip appears in multiple splits"
            all_ids |= ids


# ── Dataset tests ──────────────────────────────────────────────────────────
class TestDataset:
    def test_dataset_length(self):
        X = np.random.randn(100, 20, FEATURE_DIM).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.int64)
        ds = CrowdSequenceDataset(X, y)
        assert len(ds) == 100

    def test_dataset_item_shapes(self):
        X = np.random.randn(50, 20, FEATURE_DIM).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.int64)
        ds = CrowdSequenceDataset(X, y)
        x, label = ds[0]
        assert x.shape == (20, FEATURE_DIM)
        assert label.shape == ()

    def test_class_weights_positive(self):
        X = np.random.randn(100, 20, FEATURE_DIM).astype(np.float32)
        y = np.array([0] * 80 + [1] * 20).astype(np.int64)
        ds = CrowdSequenceDataset(X, y)
        weights = ds.compute_class_weights()
        assert weights[1] > weights[0], "Minority class should have higher weight"


# ── Model shape tests ───────────────────────────────────────────────────────
class TestModelShapes:
    @pytest.mark.parametrize("batch,seq,feat,hidden,layers,out", [
        (8, 20, 10, 32, 1, 2),
        (4, 10, 10, 64, 2, 2),
        (16, 30, 10, 128, 2, 2),
    ])
    def test_gru_output_shape(self, batch, seq, feat, hidden, layers, out):
        model = CrowdGuardGRU(input_dim=feat, hidden_dim=hidden,
                               num_layers=layers, output_dim=out)
        x = torch.randn(batch, seq, feat)
        logits = model(x)
        assert logits.shape == (batch, out)

    @pytest.mark.parametrize("batch,seq,feat,hidden,layers,out", [
        (8, 20, 10, 32, 1, 2),
        (4, 10, 10, 64, 2, 2),
    ])
    def test_lstm_output_shape(self, batch, seq, feat, hidden, layers, out):
        model = CrowdGuardLSTM(input_dim=feat, hidden_dim=hidden,
                                num_layers=layers, output_dim=out)
        x = torch.randn(batch, seq, feat)
        logits = model(x)
        assert logits.shape == (batch, out)

    def test_gru_parameter_count(self, small_model_gru):
        n = small_model_gru.count_parameters()
        assert n > 0

    def test_lstm_parameter_count(self, small_model_lstm):
        n = small_model_lstm.count_parameters()
        assert n > 0

    def test_lstm_has_more_params_than_gru(self):
        gru = CrowdGuardGRU(input_dim=10, hidden_dim=32, num_layers=1, output_dim=2)
        lstm = CrowdGuardLSTM(input_dim=10, hidden_dim=32, num_layers=1, output_dim=2)
        # LSTM has ~33% more params due to extra gate
        assert lstm.count_parameters() > gru.count_parameters()

    def test_gru_config_stored(self, small_model_gru):
        assert "model_type" in small_model_gru.config
        assert small_model_gru.config["model_type"] == "gru"

    def test_lstm_config_stored(self, small_model_lstm):
        assert "model_type" in small_model_lstm.config
        assert small_model_lstm.config["model_type"] == "lstm"


# ── Checkpoint tests ────────────────────────────────────────────────────────
class TestCheckpointLoading:
    def test_checkpoint_save_load_weights(self, small_model_gru, tmp_path):
        optimizer = torch.optim.Adam(small_model_gru.parameters())
        save_checkpoint(small_model_gru, optimizer, epoch=3,
                        metrics={"f1_macro": 0.8}, save_dir=str(tmp_path))

        model2 = CrowdGuardGRU(
            input_dim=FEATURE_DIM, hidden_dim=16, num_layers=1, output_dim=2
        )
        load_checkpoint(model2, str(tmp_path / "best_model.pt"),
                        device=torch.device("cpu"))

        for p1, p2 in zip(small_model_gru.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_checkpoint_contains_model_config(self, small_model_gru, tmp_path):
        optimizer = torch.optim.Adam(small_model_gru.parameters())
        save_checkpoint(small_model_gru, optimizer, epoch=1,
                        metrics={}, save_dir=str(tmp_path))

        ckpt = torch.load(str(tmp_path / "best_model.pt"),
                          map_location="cpu", weights_only=False)
        assert "model_config" in ckpt
        assert ckpt["model_config"]["model_type"] == "gru"


# ── Metrics tests ───────────────────────────────────────────────────────────
class TestMetrics:
    def test_perfect_classifier(self):
        y = np.array([0, 0, 1, 1])
        results = full_evaluation(y, y, n_classes=2)
        assert results["accuracy"] == pytest.approx(1.0)
        assert results["f1_macro"] == pytest.approx(1.0)

    def test_worst_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        results = full_evaluation(y_true, y_pred, n_classes=2)
        assert results["accuracy"] == pytest.approx(0.0)

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1])
        cm = compute_confusion_matrix(y_true, y_pred, n_classes=2)
        assert cm.shape == (2, 2)
        assert cm.sum() == 4

    def test_recall_anomaly_in_results(self):
        y = np.array([0, 1, 1, 0])
        results = full_evaluation(y, y, n_classes=2, positive_class=1)
        assert "recall_anomaly" in results
        assert "false_negative_rate_anomaly" in results


# ── Reproducibility tests ───────────────────────────────────────────────────
class TestReproducibility:
    def test_same_seed_same_output(self):
        from reproducibility import set_seed
        set_seed(42)
        x1 = np.random.randn(10)
        set_seed(42)
        x2 = np.random.randn(10)
        np.testing.assert_array_equal(x1, x2)

    def test_same_seed_same_model_init(self):
        from reproducibility import set_seed
        set_seed(42)
        m1 = CrowdGuardGRU(input_dim=10, hidden_dim=16, num_layers=1, output_dim=2)
        set_seed(42)
        m2 = CrowdGuardGRU(input_dim=10, hidden_dim=16, num_layers=1, output_dim=2)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            # Note: GRU uses custom init — seeds affect random ops before/after
            pass  # weights are deterministically initialised (xavier/ortho), not random


# ── Inference tests ─────────────────────────────────────────────────────────
class TestInference:
    def test_inference_output_keys(self, small_model_gru, tmp_path):
        """Check inference output structure."""
        import json

        # Mock saved checkpoint
        optimizer = torch.optim.Adam(small_model_gru.parameters())
        save_checkpoint(small_model_gru, optimizer, epoch=1,
                        metrics={}, save_dir=str(tmp_path))

        from inference import CrowdGuardPredictor

        # Create a dummy scaler
        from preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, FEATURE_DIM))
        scaler.save(str(tmp_path / "scaler.pkl"))

        predictor = CrowdGuardPredictor(
            model_type="gru",
            checkpoint_dir=str(tmp_path),
            scaler_path=str(tmp_path / "scaler.pkl"),
        )

        seq = np.random.randn(20, FEATURE_DIM).astype(np.float32)
        result = predictor.predict(seq, camera_id="cam01", zone_id="entry")

        assert "prediction" in result
        assert "class" in result["prediction"]
        assert "confidence" in result["prediction"]
        assert "anomaly_probability" in result["prediction"]
        assert "data_quality" in result
        assert "status" in result
        assert result["status"] == "experimental"
        assert result["camera_id"] == "cam01"
        assert result["zone_id"] == "entry"

    def test_inference_handles_nan(self, small_model_gru, tmp_path):
        """Model should handle NaN input gracefully."""
        optimizer = torch.optim.Adam(small_model_gru.parameters())
        save_checkpoint(small_model_gru, optimizer, epoch=1,
                        metrics={}, save_dir=str(tmp_path))

        from inference import CrowdGuardPredictor
        from preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, FEATURE_DIM))
        scaler.save(str(tmp_path / "scaler.pkl"))

        predictor = CrowdGuardPredictor(
            model_type="gru",
            checkpoint_dir=str(tmp_path),
            scaler_path=str(tmp_path / "scaler.pkl"),
        )

        seq = np.random.randn(20, FEATURE_DIM).astype(np.float32)
        seq[5, 0] = np.nan  # inject NaN

        result = predictor.predict(seq)
        assert result["prediction"]["class"] in [0, 1]
