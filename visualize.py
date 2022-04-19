import os
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


def visualize(clf, X_test, y_test, model_name, dir_output):
    """
    :param dir_output: desired directory for saving report png
    :param model_name: name of the saved model
    :param clf: estimator
    :param X_test: content column of data
    :param y_test: label column of data
    """
    plot_confusion_matrix(clf, X_test, y_test, include_values=False, cmap='Blues')
    plt.show()

    dir_report = os.path.join(dir_output, model_name)
    plt.savefig(f'{dir_report}/{model_name}.png', bbox_inches='tight')
