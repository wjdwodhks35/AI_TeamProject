from apartment_ml.config import load_settings
from apartment_ml.pipeline import run_smoke_pipeline


def test_settings_load():
    s = load_settings()
    assert s.classification_training_enabled is False
    assert abs(s.split.train_ratio + s.split.valid_ratio + s.split.test_ratio - 1) < 1e-9


def test_smoke_pipeline_on_sample_data():
    res = run_smoke_pipeline(load_settings())
    assert (res.data["source"] == "synthetic_sample").all()
    assert res.report.duplicate_rows == 1
    assert res.report.invalid_rows == {"non_positive_price": 1}
    assert res.data["base_rate"].notna().all()
    assert res.data["anomaly_label"].isna().all()
    summary = res.split.summary()
    assert summary["rows"].sum() == len(res.data)
    assert summary.loc["train", "end"] < summary.loc["valid", "start"]
    assert summary.loc["valid", "end"] < summary.loc["test", "start"]
