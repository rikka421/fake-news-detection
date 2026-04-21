from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_analysis_progress.serving import train_and_export_serving_model


if __name__ == "__main__":
    path = train_and_export_serving_model(
        dataset="balanced_medium",
        text_mode="title_content",
        mask_label_tokens=True,
        split_mode="grouped_title_content",
    )
    print(f"Saved serving model to: {path}")
