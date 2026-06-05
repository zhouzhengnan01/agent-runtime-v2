from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prepare_yolo_dataset_module():
    path = Path("plugins/skills/data-auto-annotation/scripts/prepare_yolo_dataset.py").resolve()
    spec = importlib.util.spec_from_file_location("prepare_yolo_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_aware_split_keeps_positive_test_split_for_small_real_dataset() -> None:
    module = _load_prepare_yolo_dataset_module()
    coco = {
        "images": [
            {"id": image_id, "file_name": f"real_{image_id}.jpg", "source": "real"}
            for image_id in range(1, 15)
        ]
        + [
            {"id": image_id, "file_name": f"synthetic_{image_id}.jpg", "source": "synthetic"}
            for image_id in range(15, 23)
        ],
        "annotations": [],
        "categories": [],
    }

    split_ids, source_counts = module._split_image_ids_with_source(
        coco,
        {"train": 0.7, "val": 0.2, "test": 0.1},
        seed=42,
        synthetic_policy={
            "source_field": "source",
            "synthetic_values": ["synthetic", "generated", "gen"],
            "is_synthetic_fields": ["is_synthetic", "synthetic"],
            "val_real_only": True,
            "test_real_only": True,
            "synthetic_to_train_only": True,
        },
    )

    assert len(split_ids["test"]) == 1
    assert source_counts["test"] == {"real": 1, "synthetic": 0}
    assert source_counts["train"]["synthetic"] == 8
