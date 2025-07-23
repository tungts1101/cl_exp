import torch
import numpy as np
import random


def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def accuracy(y_pred, y_true, class_increments):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = []
    acc_total = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    for task_id, classes in enumerate(class_increments):
        idxes = np.where(
            np.logical_and(y_true >= classes[0], y_true <= classes[1])
        )[0]
        all_acc.append(np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        ))

    return acc_total, all_acc


def compute_metrics(accuracy_matrix):
    faa = np.mean(accuracy_matrix[-1])
    if accuracy_matrix.shape[0] == 1:
        return faa, 0.0
    final_acc_per_task = accuracy_matrix[-1]
    max_acc_per_task = np.max(accuracy_matrix, axis=0)
    ffm = np.mean(max_acc_per_task[:-1] - final_acc_per_task[:-1])
    
    return faa, ffm


if __name__ == '__main__':
    # accuracy_matrix = np.array([
    #     [80,  0,  0,  0],  # After task 1
    #     [75, 82,  0,  0],  # After task 2
    #     [70, 78, 85,  0],  # After task 3
    #     [68, 75, 80, 87]   # After task 4 (final)
    # ])
    accuracy_matrix = np.array([[85]])

    faa, ffm = compute_metrics(accuracy_matrix)
    print(f"Final Average Accuracy (FAA): {faa:.2f}")
    print(f"Final Forgetting Measure (FFM): {ffm:.2f}")