import csv
import math

from src.utils.logging import CSVLogger, build_csv_logger_schema


def test_train_csv_schema_keeps_fixed_columns_aligned(tmp_path):
    losses = {
        "loss": 1.25,
        "predictor_loss": 0.75,
    }
    total_stats = {
        "act_max": 0.9,
        "act_mean": 0.1,
        "act_min": -0.2,
    }
    columns, csv_columns = build_csv_logger_schema(
        losses,
        total_stats,
        leading_columns=[("%d", "epoch"), ("%d", "itr")],
        fixed_columns=[
            ("%.5f", "loss"),
            ("%.5f", "gpu-time(ms)"),
            ("%.5f", "iter-time(ms)"),
        ],
    )

    assert columns[:5] == ["epoch", "itr", "loss", "gpu-time(ms)", "iter-time(ms)"]
    assert columns.count("loss") == 1
    assert [name for _, name in csv_columns] == columns

    log_values = [1, 30, losses["loss"], 226.0, 233.0]
    for key in columns[5:]:
        if key in losses:
            log_values.append(losses[key])
        elif key in total_stats:
            log_values.append(total_stats[key])
        else:
            log_values.append(0.0)

    path = tmp_path / "log_r0.csv"
    logger = CSVLogger(path, *csv_columns)
    logger.log(*log_values)

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    row = rows[0]
    assert math.isclose(float(row["loss"]), 1.25)
    assert math.isclose(float(row["gpu-time(ms)"]), 226.0)
    assert math.isclose(float(row["iter-time(ms)"]), 233.0)
    assert math.isclose(float(row["act_max"]), 0.9)
    assert math.isclose(float(row["act_mean"]), 0.1)
    assert math.isclose(float(row["act_min"]), -0.2)
    assert math.isclose(float(row["predictor_loss"]), 0.75)
