"""
Diagnose: for both_wrong examples, A is decodable but the model still fails.
Why? Check:
1. Are probe predictions actually close to true A? Or is R² misleading due to small n?
2. Is A+Y much worse than A? (addition bottleneck)
3. Are these examples harder — larger A values, larger sums?
4. What does the model actually output vs the correct answer?
"""

import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Ill-conditioned")

import numpy as np
from sklearn.metrics import r2_score


def load_metadata(extraction_dir, condition):
    cond_dir = extraction_dir / condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    meta = []
    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        meta.append({
            "problem_idx": data["problem_idx"],
            "fact_value": data["fact_value"],
            "x": data["x"],
            "answer": data["answer"],
            "model_correct": data["model_correct"],
            "model_answer": data.get("model_answer"),
        })
    return meta


def main():
    extraction_dir = Path("probing/extracted_states")
    results_dir = Path("probing/probe_results")

    with open(results_dir / "all_probe_results.pkl", "rb") as f:
        all_results = pickle.load(f)

    bl_meta = load_metadata(extraction_dir, "baseline")
    dt_meta = load_metadata(extraction_dir, "dots_250")

    n = len(bl_meta)
    bl_correct = np.array([m["model_correct"] for m in bl_meta], dtype=bool)
    dt_correct = np.array([m["model_correct"] for m in dt_meta], dtype=bool)

    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    split = int(0.75 * n)
    test_idx = indices[split:]

    bl_test = bl_correct[test_idx]
    dt_test = dt_correct[test_idx]

    both_wrong = ~bl_test & ~dt_test
    filler_helped = ~bl_test & dt_test
    both_correct = bl_test & dt_test

    # Get probe predictions at answer_prompt, best layer for A
    dots_A = all_results["dots_250"]["A"]
    pos = "answer_prompt"
    layers = sorted(dots_A[pos].keys())
    best_layer = max(layers, key=lambda l: dots_A[pos][l]["r2"])
    probe_A_pred = dots_A[pos][best_layer]["y_pred"]
    probe_A_true = dots_A[pos][best_layer]["y_test"]

    # Same for A+Y
    dots_AY = all_results["dots_250"]["A+Y"]
    best_layer_ay = max(sorted(dots_AY[pos].keys()), key=lambda l: dots_AY[pos][l]["r2"])
    probe_AY_pred = dots_AY[pos][best_layer_ay]["y_pred"]
    probe_AY_true = dots_AY[pos][best_layer_ay]["y_test"]

    # Compare categories
    print("=" * 80)
    print("CATEGORY COMPARISON")
    print("=" * 80)

    for cat_name, mask in [("filler_helped", filler_helped),
                            ("both_correct", both_correct),
                            ("both_wrong", both_wrong)]:
        if mask.sum() < 3:
            continue

        true_A = probe_A_true[mask]
        pred_A = probe_A_pred[mask]
        true_AY = probe_AY_true[mask]
        pred_AY = probe_AY_pred[mask]

        # Get metadata for these test examples
        test_meta_dt = [dt_meta[test_idx[i]] for i in range(len(test_idx)) if mask[i]]
        test_meta_bl = [bl_meta[test_idx[i]] for i in range(len(test_idx)) if mask[i]]
        answers = np.array([m["answer"] for m in test_meta_dt])
        As = np.array([m["fact_value"] for m in test_meta_dt])
        Ys = np.array([m["x"] for m in test_meta_dt])
        model_ans_dt = [m["model_answer"] for m in test_meta_dt]
        model_ans_bl = [m["model_answer"] for m in test_meta_bl]

        print(f"\n--- {cat_name} (n={mask.sum()}) ---")
        print(f"  A stats:   mean={As.mean():.1f}, median={np.median(As):.1f}, "
              f"std={As.std():.1f}, range=[{As.min()}, {As.max()}]")
        print(f"  Y stats:   mean={Ys.mean():.1f}, std={Ys.std():.1f}")
        print(f"  A+Y stats: mean={answers.mean():.1f}, std={answers.std():.1f}, "
              f"range=[{answers.min()}, {answers.max()}]")
        print(f"  Probe A:   R²={r2_score(true_A, pred_A):.3f}, "
              f"MAE={np.mean(np.abs(true_A - pred_A)):.2f}, "
              f"exact(±1)={np.mean(np.abs(np.round(pred_A) - true_A) <= 1):.1%}")
        print(f"  Probe A+Y: R²={r2_score(true_AY, pred_AY):.3f}, "
              f"MAE={np.mean(np.abs(true_AY - pred_AY)):.2f}, "
              f"exact(±1)={np.mean(np.abs(np.round(pred_AY) - true_AY) <= 1):.1%}")

        # Model error analysis
        errors_dt = []
        errors_bl = []
        for i in range(len(test_meta_dt)):
            if model_ans_dt[i] is not None:
                errors_dt.append(abs(model_ans_dt[i] - answers[i]))
            if model_ans_bl[i] is not None:
                errors_bl.append(abs(model_ans_bl[i] - answers[i]))

        if errors_dt:
            errors_dt = np.array(errors_dt)
            print(f"  Model error (dots): mean={errors_dt.mean():.1f}, "
                  f"median={np.median(errors_dt):.1f}, "
                  f"≤5: {np.mean(errors_dt <= 5):.1%}, "
                  f"≤10: {np.mean(errors_dt <= 10):.1%}")
        if errors_bl:
            errors_bl = np.array(errors_bl)
            print(f"  Model error (base): mean={errors_bl.mean():.1f}, "
                  f"median={np.median(errors_bl):.1f}, "
                  f"≤5: {np.mean(errors_bl <= 5):.1%}, "
                  f"≤10: {np.mean(errors_bl <= 10):.1%}")

    # Print individual both_wrong examples
    print("\n\n" + "=" * 80)
    print("INDIVIDUAL both_wrong EXAMPLES (dots_250)")
    print("=" * 80)
    bw_indices = np.where(both_wrong)[0]
    for i in bw_indices:
        global_idx = test_idx[i]
        m = dt_meta[global_idx]
        m_bl = bl_meta[global_idx]
        pred_a = probe_A_pred[i]
        pred_ay = probe_AY_pred[i]
        print(f"  prob_{m['problem_idx']:03d}: A={m['fact_value']:>4}, Y={m['x']:>4}, "
              f"A+Y={m['answer']:>4} | "
              f"probe_A={pred_a:>6.1f} (err={abs(pred_a - m['fact_value']):>5.1f}), "
              f"probe_A+Y={pred_ay:>6.1f} (err={abs(pred_ay - m['answer']):>5.1f}) | "
              f"model_dots={m['model_answer']}, model_bl={m_bl['model_answer']}")


if __name__ == "__main__":
    main()
