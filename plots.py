import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import streamlit as st
import numpy as np

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "#F9F9F9"
})

def plot_confusion_matrix(y_true, y_pred, model_name, labels=None):
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))  

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                     xticklabels=labels, yticklabels=labels,
                     cbar_kws={"shrink": 0.8}, linewidths=0.5, linecolor='gray')
    plt.xlabel('Predicted', fontsize=13)
    plt.ylabel('Actual', fontsize=13)
    plt.title(f'Confusion Matrix: {model_name}', fontsize=13, weight='bold')
    st.pyplot(plt.gcf())
    plt.clf()


def plot_regression_scatter(y_true, y_pred, model_name="Model"):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=60, color="steelblue", edgecolor="white")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--', color='red', linewidth=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Regression Scatter Plot: {model_name}", fontsize=13, weight='bold')
    sns.despine()
    st.pyplot(plt.gcf())
    plt.clf()


def plot_model_comparison(results, task_type):
    labels = []
    scores = {"accuracy": [], "f1": [], "r2": []}

    for res in results:
        labels.append(res["model_key"])
        metrics = res["metrics"]
        if task_type == "Classification":
            scores["accuracy"].append(metrics.get("accuracy", 0))
            scores["f1"].append(metrics.get("f1 score", 0))
        else:
            scores["r2"].append(metrics.get("r2 score", 0))

    x = np.arange(len(labels))
    plt.figure(figsize=(12, 8))

    if task_type == "Classification":
        best_idx = int(np.argmax(scores["f1"]))

        bar_width = 0.35
        plt.bar(x - bar_width/2, scores["accuracy"], width=bar_width,
                color=["royalblue" if i != best_idx else "darkgreen" for i in range(len(labels))],
                label="Accuracy")
        plt.bar(x + bar_width/2, scores["f1"], width=bar_width,
                color=["orange" if i != best_idx else "gold" for i in range(len(labels))],
                label="F1 Score")

        plt.text(best_idx, max(scores["f1"][best_idx], scores["accuracy"][best_idx]) + 0.01,
                 f"{labels[best_idx]}\nF1: {scores['f1'][best_idx]:.3f}",
                 ha='center', va='bottom', fontweight='bold', color='black')

        plt.xticks(x, labels)
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title("Model Performance Comparison", fontsize=12, weight='bold')
        plt.legend()

    else:
        best_idx = int(np.argmax(scores["r2"]))
        plt.bar(labels, scores["r2"],
                color=["seagreen" if i != best_idx else "gold" for i in range(len(labels))],
                label="R² Score")

        plt.text(best_idx, scores["r2"][best_idx] + 0.01,
                 f"{labels[best_idx]}\nR²: {scores['r2'][best_idx]:.3f}",
                 ha='center', va='bottom', fontweight='bold', color='black')

        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title("Model Performance Comparison", fontsize=12, weight='bold')
        plt.legend()

    sns.despine()
    st.pyplot(plt.gcf())
    plt.clf()