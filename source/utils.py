import os, sys
from libs import *

def load_weights(weight_file):
    weights = []
    data = pd.read_csv(weight_file)
    classes = [{data} for data in data.columns[1:].tolist()]
    for i in range(data.shape[0]):
        weights.append(data.loc[i][1:].tolist())
    weights = np.array(weights)
    
    return classes, weights

def compute_modified_confusion_matrix(labels, outputs):
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))
    for i in range(num_recordings):
        normalization = float(max(np.sum(np.any((labels[i][:], outputs[i][:]), axis=0)), 1))
        for j in range(num_classes):
            if labels[i][j]:
                for k in range(num_classes):
                    if outputs[i][k]:
                        A[j, k] += 1.0/normalization

    return A

def compute_metrics(weights_file, labels, outputs):
    classes, weights = load_weights(weights_file)
    sinus_rhythm = set(['426783006'])
    num_recordings, num_classes = np.shape(labels)
    if sinus_rhythm in classes:
        sinus_rhythm_index = classes.index(sinus_rhythm)
    else:
        raise ValueError('The sinus rhythm class is not available.')

    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool_)
    inactive_outputs[:, sinus_rhythm_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score