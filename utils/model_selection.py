"""
SeedLab AI — Dynamic Model Selection Utility.
Evaluates multi-seed experiment CSVs and deterministically ranks algorithms.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


def normalize_experiment_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Standardize column headers from uploaded experiment CSVs.
    Supports variations in case, spaces, and abbreviations.
    """
    col_map = {}
    for col in df.columns:
        c = str(col).strip().lower().replace("_", " ")
        if c in ["algorithm", "model", "classifier", "algo", "model name"]:
            col_map[col] = "Algorithm"
        elif c in ["seed", "random seed", "random_seed", "seed id", "run"]:
            col_map[col] = "Seed"
        elif c in ["accuracy", "acc", "test accuracy", "val accuracy"]:
            col_map[col] = "Accuracy"
        elif c in ["precision", "macro precision", "prec", "test precision"]:
            col_map[col] = "Precision"
        elif c in ["recall", "macro recall", "rec", "test recall"]:
            col_map[col] = "Recall"
        elif c in ["f1 score", "f1", "macro f1", "f1-score", "macro f1 score"]:
            col_map[col] = "F1 Score"
        elif c in ["training time", "train time", "time", "time (s)", "training time (s)", "latency"]:
            col_map[col] = "Training Time"

    standardized = df.rename(columns=col_map).copy()
    return standardized, col_map


def canonical_model_name(name: str) -> str:
    """Normalize algorithm names to canonical SeedLab names."""
    n = str(name).strip().lower()
    if "forest" in n or "rf" in n:
        return "Random Forest"
    elif "mlp" in n or "neural" in n or "perceptron" in n:
        return "MLP Neural Network"
    elif "logistic" in n or "lr" in n or "linear" in n:
        return "Logistic Regression"
    return str(name).strip()


def evaluate_and_select_best_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes multi-seed experiment results and dynamically selects the best model.

    Selection Criteria:
    - Primary: Highest Mean Macro F1-Score across random seeds.
      (Fallback: Highest Mean Accuracy if F1-Score is not in CSV).
    - Tie-breaker 1: Lower Standard Deviation of Macro F1-Score (stability).
    - Tie-breaker 2: Higher Mean Accuracy.
    - Tie-breaker 3: Lower Mean Training Time (efficiency).

    Returns:
        dict containing:
            - 'summary_df': aggregated dataframe with means and std devs per algorithm
            - 'best_model': dict with winning model info and stats
            - 'ranking': list of ranked models
            - 'used_fallback': bool indicating if accuracy was used instead of F1
            - 'reason': detailed text explanation of why the winner was chosen
    """
    clean_df, _ = normalize_experiment_columns(df)

    # Validate essential columns
    if "Algorithm" not in clean_df.columns:
        clean_df["Algorithm"] = "Default Model"
    else:
        clean_df["Algorithm"] = clean_df["Algorithm"].apply(canonical_model_name)

    if "Seed" not in clean_df.columns:
        clean_df["Seed"] = range(len(clean_df))

    # Numeric conversion
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "Training Time"]
    for col in metric_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Detect if F1 is available and valid
    has_f1 = "F1 Score" in clean_df.columns and clean_df["F1 Score"].notnull().sum() > 0
    used_fallback = not has_f1

    # Ensure accuracy exists
    if "Accuracy" not in clean_df.columns or clean_df["Accuracy"].isnull().all():
        raise ValueError("The uploaded CSV does not contain a valid 'Accuracy' column.")

    # Convert accuracy / f1 / prec / rec to standard 0-1 scale internally if they were percentages > 1
    for m in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        if m in clean_df.columns and clean_df[m].max() > 1.0:
            clean_df[m] = clean_df[m] / 100.0

    # Ensure training time exists (fill with 0 if missing)
    if "Training Time" not in clean_df.columns:
        clean_df["Training Time"] = 0.0
    else:
        clean_df["Training Time"] = clean_df["Training Time"].fillna(0.0)

    # Aggregations per model
    records = []
    for algo, grp in clean_df.groupby("Algorithm"):
        n_seeds = len(grp)
        mean_acc = float(grp["Accuracy"].mean())
        std_acc = float(grp["Accuracy"].std(ddof=1)) if n_seeds > 1 else 0.0

        mean_prec = float(grp["Precision"].mean()) if "Precision" in grp.columns and grp["Precision"].notnull().any() else mean_acc
        mean_rec = float(grp["Recall"].mean()) if "Recall" in grp.columns and grp["Recall"].notnull().any() else mean_acc

        if has_f1:
            mean_f1 = float(grp["F1 Score"].mean())
            std_f1 = float(grp["F1 Score"].std(ddof=1)) if n_seeds > 1 else 0.0
        else:
            mean_f1 = mean_acc
            std_f1 = std_acc

        mean_time = float(grp["Training Time"].mean())
        std_time = float(grp["Training Time"].std(ddof=1)) if n_seeds > 1 else 0.0

        records.append({
            "Algorithm": algo,
            "Seeds_Tested": n_seeds,
            "Mean_Accuracy": mean_acc,
            "Std_Accuracy": std_acc,
            "Mean_Precision": mean_prec,
            "Mean_Recall": mean_rec,
            "Mean_F1": mean_f1,
            "Std_F1": std_f1,
            "Mean_Time": mean_time,
            "Std_Time": std_time
        })

    summary_df = pd.DataFrame(records)

    # Deterministic Ranking Function
    # We sort by:
    # 1. Primary score descending (Mean F1 or Mean Accuracy) rounded to 5 decimal places
    # 2. Std dev ascending (lower variance is better) rounded to 5 decimal places
    # 3. Accuracy descending (tie-breaker)
    # 4. Training Time ascending (faster is better)
    def ranking_key(row):
        f1_score_neg = -round(row["Mean_F1"], 5)
        std_f1_pos = round(row["Std_F1"], 5)
        acc_neg = -round(row["Mean_Accuracy"], 5)
        time_pos = round(row["Mean_Time"], 4)
        return (f1_score_neg, std_f1_pos, acc_neg, time_pos)

    sorted_indices = sorted(range(len(summary_df)), key=lambda idx: ranking_key(summary_df.iloc[idx]))
    ranked_df = summary_df.iloc[sorted_indices].reset_index(drop=True)

    winner_row = ranked_df.iloc[0]
    winning_model = winner_row["Algorithm"]

    # Formulate clear, human-readable reason
    n_tested = int(winner_row["Seeds_Tested"])
    f1_pct = winner_row["Mean_F1"] * 100
    acc_pct = winner_row["Mean_Accuracy"] * 100
    std_f1_val = winner_row["Std_F1"] * 100
    time_val = winner_row["Mean_Time"]

    if used_fallback:
        reason = (
            f"Highest mean test accuracy ({acc_pct:.2f}%) across {n_tested} random seeds. "
            f"(Note: Macro F1-score was not present in the CSV, so accuracy was used as the fallback criterion)."
        )
    else:
        # Check if runner up had comparable F1
        if len(ranked_df) > 1:
            runner_up = ranked_df.iloc[1]
            diff = (winner_row["Mean_F1"] - runner_up["Mean_F1"]) * 100
            if abs(diff) < 0.05:
                reason = (
                    f"Highest stability and accuracy among top contenders: achieved mean macro F1-score of {f1_pct:.2f}% "
                    f"across {n_tested} random seeds with low variability (±{std_f1_val:.3f}%), "
                    f"edging out {runner_up['Algorithm']} by tie-breaker metrics."
                )
            else:
                reason = (
                    f"Highest mean macro F1-score ({f1_pct:.2f}%) and mean accuracy ({acc_pct:.2f}%) "
                    f"across all {n_tested} tested random seeds with low variance (±{std_f1_val:.3f}%)."
                )
        else:
            reason = (
                f"Achieved mean macro F1-score of {f1_pct:.2f}% and accuracy of {acc_pct:.2f}% "
                f"across {n_tested} tested seeds."
            )

    best_model_info = {
        "name": winning_model,
        "mean_accuracy": winner_row["Mean_Accuracy"],
        "mean_precision": winner_row["Mean_Precision"],
        "mean_recall": winner_row["Mean_Recall"],
        "mean_f1": winner_row["Mean_F1"],
        "std_f1": winner_row["Std_F1"],
        "std_accuracy": winner_row["Std_Accuracy"],
        "mean_training_time": winner_row["Mean_Time"],
        "seeds_tested": n_tested,
        "reason": reason,
        "used_fallback": used_fallback
    }

    return {
        "summary_df": ranked_df,
        "best_model": best_model_info,
        "ranking": ranked_df["Algorithm"].tolist(),
        "used_fallback": used_fallback,
        "clean_experiment_df": clean_df
    }
