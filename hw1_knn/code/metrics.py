import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    """
    YOUR CODE IS HERE
    """
    y_true = y_true.astype(int)

    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0

    for actual, predicted in zip(y_true, y_pred):
        if predicted == actual:
            if predicted == 1: # tp
                true_pos += 1
            else: # tn
                true_neg += 1
        else:
            if predicted == 1: # fp
                false_pos += 1
            else: # fn
                false_neg += 1

    confusion_matrix = np.array([
        [true_pos, false_pos],
        [false_neg, true_neg]
    ])
    accuracy = (true_pos + true_neg)/np.sum(confusion_matrix)
    precision = (true_pos/(true_pos + false_pos))
    recall = (true_pos/(true_pos + false_neg))
    f1_score = 2*(precision * recall)/(precision+recall)
    return f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}'


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    """
    YOUR CODE IS HERE
    """
    pass


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    """
    YOUR CODE IS HERE
    """
    pass


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    """
    YOUR CODE IS HERE
    """
    pass


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    """
    YOUR CODE IS HERE
    """
    pass
    