# scorer

from sklearn.metrics import f1_score, classification_report, confusion_matrix
from data_load import label_coder
import numpy as np

# SCORER_FILE="SemEval2010_task8_all_data\SemEval2010_task8_scorer-v1.2\semeval2010_task8_scorer-v1.2.pl"
# SCORER_FILE=os.path.abspath(SCORER_FILE)

def report(num_labels, num_preds):
    num_to_label, _ = label_coder()
    return classification_report(num_labels, num_preds, target_names = num_to_label)
def cm(num_labels, num_preds):
    # avoid matrix output ugly since 1 row could be divided into 2
    np.set_printoptions(linewidth=np.inf, precision=0, suppress=True)
    return confusion_matrix(num_labels, num_preds)

def my_f1_score(num_labels, num_preds):
    num_to_label, _ = label_coder()
    true_labels = [num_to_label[num] for num in num_labels]
    true_preds = [num_to_label[num] for num in num_preds]
    class_labels = num_to_label
    class_labels.remove("Other")
    macro_f1 = f1_score(
        true_labels,
        true_preds,
        labels=class_labels,  # without "Other"
        average="macro",
        zero_division=0)# avoid warnings
    return macro_f1 * 100
