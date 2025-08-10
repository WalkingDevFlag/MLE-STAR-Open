import argparse
import os
import sys
import pandas as pd


def make_submission(output_dir: str, id_col: str = "id", id_start: int = 0,
                    pred_filename: str = "predictions.csv",
                    sub_filename: str = "submission.csv",
                    target_col: str = "median_house_value") -> str:
    pred_path = os.path.join(output_dir, pred_filename)
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    df = pd.read_csv(pred_path)
    if target_col not in df.columns:
        # Try to infer the target column if single column
        if df.shape[1] == 1:
            target_col = df.columns[0]
        else:
            raise ValueError(f"Target column '{target_col}' not found in {pred_path}")
    n = len(df)
    ids = list(range(id_start, id_start + n))
    sub = pd.DataFrame({id_col: ids, target_col: df[target_col].values})
    sub_path = os.path.join(output_dir, sub_filename)
    sub.to_csv(sub_path, index=False)
    return sub_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create Kaggle submission.csv from predictions.csv")
    parser.add_argument("--output-dir", required=True, help="Directory containing predictions.csv")
    parser.add_argument("--id-col", default="id", help="Name of the id column (default: id)")
    parser.add_argument("--id-start", type=int, default=0, help="Starting index for id (default: 0)")
    parser.add_argument("--pred-filename", default="predictions.csv")
    parser.add_argument("--sub-filename", default="submission.csv")
    parser.add_argument("--target-col", default="median_house_value")
    args = parser.parse_args(argv)

    out_path = make_submission(
        output_dir=args.output_dir,
        id_col=args.id_col,
        id_start=args.id_start,
        pred_filename=args.pred_filename,
        sub_filename=args.sub_filename,
        target_col=args.target_col,
    )
    print(f"Wrote submission to: {out_path}")


if __name__ == "__main__":
    main()
