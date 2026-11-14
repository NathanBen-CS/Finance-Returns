"""
Charts for Phase 2 Task 2 (classification). Run ../data_phase2/classification_events.py first.

  27  ROC curves per model: one-vs-rest for the 5 classes plus the macro average
  28  the same curves compared: decision tree vs Naive Bayes, one panel per class plus the macro average

Run from inside charts2/:  ..\\..\\.venv\\Scripts\\python.exe classification_charts.py
"""
import matplotlib.pyplot as plt
import pandas as pd

from chart_style import BLUE, BLUE_RAMP, DATA, INK, MUTED, ORANGE, RED_RAMP, save, subtitle

CLASSES = ["Strong Negative", "Negative", "Neutral", "Positive", "Strong Positive"]
MACRO = "Macro average"
# The classes run from bad to good news, so they share the red-to-blue scale with a neutral grey middle.
CLASS_COLOURS = dict(zip(CLASSES, [RED_RAMP[4], RED_RAMP[2], MUTED, BLUE_RAMP[2], BLUE_RAMP[4]]))


def load():
    points = pd.read_csv(DATA / "classification_roc_points.csv")
    metrics = pd.read_csv(DATA / "classification_metrics.csv")
    auc = metrics[metrics["metric"] == "auc"].set_index(["model", "class"])["value"]
    macro = metrics[metrics["metric"] == "macro_auc"].set_index("model")["value"]
    return points, auc, macro


def chance_line(ax):
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")


def square_axes(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal")


def roc_by_model(points, auc, macro):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.6))
    for ax, (model, curves) in zip(axes, points.groupby("model", sort=False)):
        chance_line(ax)
        for name in CLASSES:
            curve = curves[curves["curve"] == name]
            ax.plot(curve["fpr"], curve["tpr"], color=CLASS_COLOURS[name], linewidth=2, label=f"{name}: AUC {auc[model, name]:.3f}")
        curve = curves[curves["curve"] == MACRO]
        ax.plot(curve["fpr"], curve["tpr"], color=INK, linewidth=2.6, label=f"{MACRO}: AUC {macro[model]:.3f}")
        square_axes(ax)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(frameon=False, loc="lower right", fontsize=9.5)
        ax.set_title(model, loc="left", pad=20)
        subtitle(ax, "One-vs-rest on the test rows. Dashed line: guessing.")
    fig.suptitle("Both models beat guessing on every class, but not by much", x=0.012, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.1, top=0.84, wspace=0.16)
    save(fig, "27_classification_roc_by_model.png", tight=False)


def roc_comparison(points, auc, macro):
    tree, bayes = points["model"].unique()
    colours = {tree: BLUE, bayes: ORANGE}
    fig, axes = plt.subplots(2, 3, figsize=(12, 9.6), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, [*CLASSES, MACRO]):
        chance_line(ax)
        for model, colour in colours.items():
            curve = points[(points["model"] == model) & (points["curve"] == name)]
            ax.plot(curve["fpr"], curve["tpr"], color=colour, linewidth=2)
        values = {model: macro[model] if name == MACRO else auc[model, name] for model in colours}
        square_axes(ax)
        ax.set_title(name, loc="left", pad=20, fontsize=12)
        subtitle(ax, f"AUC: tree {values[tree]:.3f}, Naive Bayes {values[bayes]:.3f}")
    handles = [plt.Line2D([], [], color=colour, linewidth=2.4, label=model) for model, colour in colours.items()]
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=1, linestyle="--", label="Guessing"))
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center")
    fig.supxlabel("False positive rate", fontsize=11, y=0.055)
    fig.supylabel("True positive rate", fontsize=11, x=0.015)
    fig.suptitle("The decision tree edges out Naive Bayes on all five classes", x=0.012, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.89, wspace=0.12, hspace=0.26)
    save(fig, "28_classification_roc_comparison.png", tight=False)


def main():
    points, auc, macro = load()
    roc_by_model(points, auc, macro)
    roc_comparison(points, auc, macro)


if __name__ == "__main__":
    main()
