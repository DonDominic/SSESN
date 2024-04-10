import numpy as np

def initialize_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values
    Returns
    -------
    dict
        a dictionary of metrics
    """
    metrics = {
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': []
    }

    return metrics


def get_mean_metrics(metric_dict):
    """takes a dictionary of lists for metrics and returns dict of mean values
    Parameters
    ----------
    metric_dict : dict
        A dictionary of metrics
    Returns
    -------
    dict
        dict of floats that reflect mean metric value
    """
    return {k: np.mean(v) for k, v in metric_dict.items()}


def set_metrics(metric_dict, cd_report):
    """Updates metric dict with batch metrics
    Parameters
    ----------
    metric_dict : dict
        dict of metrics
    cd_loss : dict(?)
        loss value
    cd_corrects : dict(?)
        number of correct results (to generate accuracy
    cd_report : list
        precision, recall, f1 values
    Returns
    -------
    dict
        dict of  updated metrics
    """
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])

    return metric_dict

def set_test_metrics(metric_dict, cd_report):

    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])

    return metric_dict

