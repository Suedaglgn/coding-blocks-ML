import os
from sklearn.metrics import classification_report, precision_recall_fscore_support


def report(y_true, y_pred, model_name, dir_output):
    """
    A method that reports model evaluation
    :param y_true: ground truth
    :param y_pred: prediction list
    :param model_name: name of the saved model
    :param dir_output: desired directory for saving report
    :return:
    """
    clf_report = classification_report(y_true, y_pred)
    f1_report = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    dir_all_report = os.path.join(dir_output, "all_report.txt")
    with open(dir_all_report, "a") as all_report:
        all_report.write("\n################## " + model_name + " ##################\n" + clf_report + "\n"
                         + "Precision\t Recall\t, F1 score\n" + str(f1_report))
    dir_report = os.path.join(dir_output, model_name, f"{model_name}_report.txt")
    with open(dir_report, "w") as report:
        report.write("\n################## " + model_name + " #####################\n" + clf_report + "\n"
                     + "Precision\t\t\t\t Recall\t\t\t\t F1 score\n" + str(f1_report))

