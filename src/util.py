import tracemalloc
from contextlib import contextmanager

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn import metrics


@contextmanager
def trace_memory():
    # Yield a reference to a dictionary when creating a new context. The
    # dictionary is empty initially until the execution within the context is
    # finished.
    stats = {}

    if not tracemalloc.is_tracing():
        tracemalloc.start()

    try:
        yield stats
    finally:
        current, peak = tracemalloc.get_traced_memory()
        stats["current"] = current
        stats["peak"] = peak

        tracemalloc.stop()


def evaluate_classification(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    p_pos = tp / (tp + fp)
    r_pos = tp / (tp + fn)
    f1_pos = (2 * p_pos * r_pos) / (p_pos + r_pos)

    p_neg = tn / (tn + fn)
    r_neg = tn / (tn + fp)
    f1_neg = (2 * p_neg * r_neg) / (p_neg + r_neg)

    p_macro = (p_pos + p_neg) / 2
    r_macro = (r_pos + r_neg) / 2
    f1_macro = (f1_pos + f1_neg) / 2

    sup_pos = tp + fn
    sup_neg = tn + fp

    p_weighted = (p_pos * sup_pos + p_neg * sup_neg) / (sup_pos + sup_neg)
    r_weighted = (r_pos * sup_pos + r_neg * sup_neg) / (sup_pos + sup_neg)
    f1_weighted = (f1_pos * sup_pos + f1_neg * sup_neg) / (sup_pos + sup_neg)

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision_1": p_pos,
        "recall_1": r_pos,
        "f1_1": f1_pos,
        "precision_0": p_neg,
        "recall_0": r_neg,
        "f1_0": f1_neg,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "support_1": sup_pos,
        "support_0": sup_neg,
        "roc_auc": roc_auc_score(y_true, y_pred),
    }


def calculate_mape(y_true, y_pred):
    actual    = np.array(y_true)
    predicted = np.array(y_pred.flatten()) 
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def evaluate_regression(model,X_train,y_train,X_test,y_test,y_pred):
    cv_score = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 6)
    r2 = r2_score(y_test, y_pred)
    n = X_test.shape[0]
    p = X_test.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    R2 = r2_score(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    MSE = metrics.mean_squared_error(y_test, y_pred)
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MAPE = calculate_mape(y_test, y_pred)
    CV_R2 = cv_score.mean()
        
    print('RMSE:', round(RMSE,4))
    print('MSE:', round(MSE,4))
    print('MAE:', round(MAE,4))
    print('MAPE:', round(MAPE,4))
    print('R2:', round(R2,4))
    print('Adjusted R2:', round(adjusted_r2, 4) )
    print("Cross Validated R2: ", round(cv_score.mean(),4) )
    
    return {
        "R2": round(R2,4),
        "Adjusted R2": round(adjusted_r2, 4),
        "Cross Validated R2": round(cv_score.mean(),4),
        "RMSE": round(RMSE,4),
        "MSE": round(MSE,4),
        "MAE": round(MAE,4),
        "MAPE": round(MAPE,4)
        }

