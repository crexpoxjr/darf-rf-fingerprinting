from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(
        y_true,
        y_pred
):

    metrics = {}


    metrics["accuracy"] = (
        accuracy_score(
            y_true,
            y_pred
        )
    )


    metrics["macro_f1"] = (
        f1_score(
            y_true,
            y_pred,
            average="macro"
        )
    )


    metrics["precision"] = (
        precision_score(
            y_true,
            y_pred,
            average=None
        ).tolist()
    )


    metrics["recall"] = (
        recall_score(
            y_true,
            y_pred,
            average=None
        ).tolist()
    )


    metrics["confusion_matrix"] = (
        confusion_matrix(
            y_true,
            y_pred
        ).tolist()
    )


    metrics["classification_report"] = (
        classification_report(
            y_true,
            y_pred,
            output_dict=True
        )
    )


    return metrics