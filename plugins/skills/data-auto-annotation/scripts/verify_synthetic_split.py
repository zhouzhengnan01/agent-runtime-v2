import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _count_synthetic_label_files(prepared_root: Path) -> dict:
    counts = {}
    for split in ("train", "val", "test"):
        image_dir = prepared_root / "images" / split
        synthetic_count = 0
        real_count = 0
        if image_dir.exists():
            for path in image_dir.rglob("*"):
                if not path.is_file():
                    continue
                normalized = path.as_posix().lower()
                if "/synthetic/" in normalized or "\\synthetic\\" in str(path).lower() or "synthetic" in path.name.lower():
                    synthetic_count += 1
                else:
                    real_count += 1
        counts[split] = {"real_like": real_count, "synthetic_like": synthetic_count}
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that val/test contain no synthetic images.")
    parser.add_argument("--run-summary", default=None, help="Path to run_summary.json from run_yolo_training.py")
    parser.add_argument("--prepared-root", default=None, help="Prepared dataset root containing images/train,val,test")
    args = parser.parse_args()

    result = {"ok": True, "errors": []}
    if args.run_summary:
        summary = _read_json(Path(args.run_summary).resolve())
        source_counts = summary.get("source_counts") or {}
        for split in ("val", "test"):
            synthetic = int((source_counts.get(split) or {}).get("synthetic", 0) or 0)
            if synthetic != 0:
                result["ok"] = False
                result["errors"].append(f"{split} has {synthetic} synthetic images according to run_summary.json")
        result["source_counts"] = source_counts

    if args.prepared_root:
        counts = _count_synthetic_label_files(Path(args.prepared_root).resolve())
        for split in ("val", "test"):
            if counts[split]["synthetic_like"] != 0:
                result["ok"] = False
                result["errors"].append(f"{split} has synthetic-looking files under prepared dataset")
        result["prepared_counts"] = counts

    if not args.run_summary and not args.prepared_root:
        raise ValueError("Provide --run-summary, --prepared-root, or both")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
