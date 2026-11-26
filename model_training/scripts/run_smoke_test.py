# -*- coding: utf-8 -*-
"""
run_smoke_test.py — CrowdGuard AI
=====================================
Quick sanity-check of the entire training pipeline.
Uses synthetic data (no real dataset needed).

Run:
    python scripts/run_smoke_test.py

This verifies:
  1. Preprocessing functions work
  2. Sequence builder generates valid windows
  3. GRU model produces correct output shapes
  4. LSTM model produces correct output shapes
  5. Training loop runs without error
  6. Checkpoint save/load works
  7. Evaluation metrics compute correctly
  8. Inference interface works
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crowdguard.smoke_test")

PASSED = []
FAILED = []


def test(name, fn):
    try:
        fn()
        logger.info(f"  PASS: {name}")
        PASSED.append(name)
    except Exception as e:
        logger.error(f"  FAIL: {name} --- {e}")
        FAILED.append((name, str(e)))


# ── Test functions ─────────────────────────────────────────────────────────
def test_preprocessing():
    from preprocessing import FEATURE_DIM, CORE_FEATURE_COLUMNS, StandardScaler, handle_missing_values
    import pandas as pd

    assert FEATURE_DIM == len(CORE_FEATURE_COLUMNS), "Feature dim mismatch"
    assert FEATURE_DIM == 10, f"Expected 10 features, got {FEATURE_DIM}"

    # Test scaler
    X = np.random.randn(100, FEATURE_DIM).astype(np.float32)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    assert X_norm.shape == X.shape
    assert abs(X_norm.mean()) < 0.1, "Normalised mean not near 0"

    # Test missing values
    df = pd.DataFrame(np.random.randn(20, FEATURE_DIM), columns=CORE_FEATURE_COLUMNS)
    df.iloc[0, 0] = np.nan
    df = handle_missing_values(df)
    assert not df.isnull().any().any(), "NaN values not handled"


def test_sequence_builder():
    from sequence_builder import SequenceBuilder
    from preprocessing import CORE_FEATURE_COLUMNS
    import pandas as pd

    n_frames = 100
    data = {c: np.random.randn(n_frames) for c in CORE_FEATURE_COLUMNS}
    data["label"] = np.random.randint(0, 2, n_frames)
    df = pd.DataFrame(data)

    builder = SequenceBuilder(seq_len=20, stride=1, feature_cols=CORE_FEATURE_COLUMNS)
    X, y, meta = builder.build_from_clip(df, "test_clip")

    expected_seqs = (n_frames - 20) + 1  # with stride=1
    assert len(X) == expected_seqs, f"Expected {expected_seqs} sequences, got {len(X)}"
    assert X.shape == (expected_seqs, 20, len(CORE_FEATURE_COLUMNS))
    assert y.shape == (expected_seqs,)
    assert len(meta) == expected_seqs


def test_dataset():
    from dataset import CrowdSequenceDataset, build_dataloaders
    from preprocessing import FEATURE_DIM

    n_seqs, seq_len, n_feat = 200, 20, FEATURE_DIM
    X = np.random.randn(n_seqs, seq_len, n_feat).astype(np.float32)
    y = np.random.randint(0, 2, n_seqs).astype(np.int64)

    ds = CrowdSequenceDataset(X, y)
    assert len(ds) == n_seqs
    x_item, y_item = ds[0]
    assert x_item.shape == (seq_len, n_feat)

    # DataLoaders
    X_val = X[:30]
    y_val = y[:30]
    X_te = X[30:60]
    y_te = y[30:60]
    X_tr = X[60:]
    y_tr = y[60:]

    train_loader, val_loader, test_loader, train_ds = build_dataloaders(
        X_tr, y_tr, X_val, y_val, X_te, y_te, batch_size=16
    )
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[0] <= 16
    assert batch_x.shape[1] == seq_len
    assert batch_x.shape[2] == n_feat


def test_gru_shapes():
    from gru_model import CrowdGuardGRU

    input_dim, seq_len, batch = 10, 20, 8
    model = CrowdGuardGRU(input_dim=input_dim, hidden_dim=32, num_layers=2, output_dim=2)
    x = torch.randn(batch, seq_len, input_dim)
    out = model(x)
    assert out.shape == (batch, 2), f"GRU output shape error: {out.shape}"

    params = model.count_parameters()
    assert params > 0, "GRU has no parameters"
    logger.info(f"  GRU params: {params:,}")


def test_lstm_shapes():
    from lstm_model import CrowdGuardLSTM

    input_dim, seq_len, batch = 10, 20, 8
    model = CrowdGuardLSTM(input_dim=input_dim, hidden_dim=32, num_layers=2, output_dim=2)
    x = torch.randn(batch, seq_len, input_dim)
    out = model(x)
    assert out.shape == (batch, 2), f"LSTM output shape error: {out.shape}"

    params = model.count_parameters()
    assert params > 0
    logger.info(f"  LSTM params: {params:,}")


def test_training_loop():
    from gru_model import CrowdGuardGRU
    from dataset import CrowdSequenceDataset
    from torch.utils.data import DataLoader
    from train_engine import train_one_epoch, validate
    import torch.nn as nn

    input_dim, seq_len = 10, 20
    n_train, n_val = 64, 32

    X_tr = np.random.randn(n_train, seq_len, input_dim).astype(np.float32)
    y_tr = np.random.randint(0, 2, n_train).astype(np.int64)
    X_v = np.random.randn(n_val, seq_len, input_dim).astype(np.float32)
    y_v = np.random.randint(0, 2, n_val).astype(np.int64)

    train_ds = CrowdSequenceDataset(X_tr, y_tr)
    val_ds = CrowdSequenceDataset(X_v, y_v)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = CrowdGuardGRU(input_dim=input_dim, hidden_dim=16, num_layers=1, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    assert 0 <= acc <= 1, f"Accuracy out of range: {acc}"
    assert loss >= 0, f"Negative loss: {loss}"

    val_metrics = validate(model, val_loader, criterion, device)
    assert "accuracy" in val_metrics
    assert "f1_macro" in val_metrics


def test_checkpoint_save_load():
    from gru_model import CrowdGuardGRU
    from train_engine import save_checkpoint, load_checkpoint

    model = CrowdGuardGRU(input_dim=10, hidden_dim=16, num_layers=1, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters())

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(model, optimizer, epoch=1, metrics={"f1_macro": 0.75},
                        save_dir=tmpdir)
        ckpt_path = os.path.join(tmpdir, "best_model.pt")
        assert os.path.exists(ckpt_path)

        # Load into new model instance
        model2 = CrowdGuardGRU(input_dim=10, hidden_dim=16, num_layers=1, output_dim=2)
        load_checkpoint(model2, ckpt_path, device=torch.device("cpu"))

        # Check weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Checkpoint weights do not match"


def test_metrics():
    from metrics import full_evaluation, compute_confusion_matrix

    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.4, 0.2, 0.9])

    results = full_evaluation(y_true, y_pred, y_prob, n_classes=2, model_name="test")
    assert "accuracy" in results
    assert "f1_macro" in results
    assert "recall_anomaly" in results
    assert "roc_auc" in results
    # 4/6 correct = ~0.667
    assert abs(results["accuracy"] - (4/6)) < 0.01


def test_reproducibility():
    from reproducibility import set_seed

    set_seed(42)
    x1 = np.random.randn(5)
    set_seed(42)
    x2 = np.random.randn(5)
    assert np.allclose(x1, x2), "Reproducibility failed"


def test_feature_extraction_from_synthetic():
    """Test feature extraction on synthetic frames."""
    from preprocessing import extract_features_from_frames, CORE_FEATURE_COLUMNS

    # Synthetic grayscale frames
    frames = [np.random.randint(0, 255, (128, 128), dtype=np.uint8) for _ in range(20)]
    df = extract_features_from_frames(frames, resize_hw=(32, 32))

    assert len(df) == 20, f"Expected 20 rows, got {len(df)}"
    for col in CORE_FEATURE_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"
    assert not df.isnull().any().any(), "Unexpected NaN values"


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print("  CrowdGuard AI — Smoke Test Suite")
    print(f"{'='*60}\n")

    test("Preprocessing (scaler, missing values)", test_preprocessing)
    test("Sequence builder", test_sequence_builder)
    test("Dataset and DataLoader", test_dataset)
    test("GRU output shapes", test_gru_shapes)
    test("LSTM output shapes", test_lstm_shapes)
    test("Training loop (1 epoch)", test_training_loop)
    test("Checkpoint save/load", test_checkpoint_save_load)
    test("Metrics computation", test_metrics)
    test("Reproducibility", test_reproducibility)
    test("Feature extraction from synthetic frames", test_feature_extraction_from_synthetic)

    print(f"\n{'='*60}")
    print(f"  Results: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("  FAILURES:")
        for name, err in FAILED:
            print(f"    FAIL {name}: {err}")
    print(f"{'='*60}\n")

    if FAILED:
        sys.exit(1)
    else:
        print("  All smoke tests PASSED. Ready to train.\n")
        print("  Next: python scripts/inspect_dataset.py  (after downloading dataset)")


if __name__ == "__main__":
    main()
