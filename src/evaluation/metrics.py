import json
import os

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate(
    y_true,
    y_pred,
    out="results/metrics.json"
):

    results={}


    results["accuracy"]=accuracy_score(
        y_true,
        y_pred
    )


    results["macro_f1"]=f1_score(
        y_true,
        y_pred,
        average="macro"
    )


    results["report"]=classification_report(
        y_true,
        y_pred,
        output_dict=True
    )


    results["confusion_matrix"]=(
        confusion_matrix(
            y_true,
            y_pred
        ).tolist()
    )


    os.makedirs(
        "results",
        exist_ok=True
    )


    with open(
        out,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


    return results