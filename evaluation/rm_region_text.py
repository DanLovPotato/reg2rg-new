import pandas as pd
import re
import os

results_path = os.environ.get(
    "RESULTS_PATH", "../results/radgenome_valid_reports.csv"
)
source_df = pd.read_csv(results_path)
branches = {
    "full": ("GT_combined_report", "Full_pred_combined_report"),
    "mask": ("Mask_GT_combined_report", "Mask_pred_combined_report"),
}

for branch, (gt_column, pred_column) in branches.items():
    output_dir = os.path.join(os.path.dirname(results_path), "evaluation", branch)
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({
        "AccNum": source_df["AccNum"],
        "Question": source_df["Question"],
        "GT_combined_report": source_df[gt_column].fillna(""),
        "Pred_combined_report": source_df[pred_column].fillna(""),
    })
    raw_path = os.path.join(output_dir, "reports.csv")
    df.to_csv(raw_path, index=False)

    stripped_df = df.copy()
    for column in ("GT_combined_report", "Pred_combined_report"):
        stripped_df[column] = stripped_df[column].map(
            lambda text: re.sub(
                r'\s+', ' ', re.sub(r"The region \d+ is [^:]+: ?", "", text)
            ).strip()
        )
    stripped_df.to_csv(
        os.path.join(output_dir, "reports_rm_region_text.csv"), index=False
    )
    print(f"Prepared {branch} evaluation files in {output_dir}")
