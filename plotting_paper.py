import json
import os
import numpy as np
import plotting


def first_figure(load_dir, save_dir):
    plotting.plot_loss_and_coverage_first(load_dir, save_dir)


def loss_and_coverage(load_dir, save_dir):
    plotting.plot_loss_and_coverage_paper(load_dir, save_dir)


def loss_compare(load_dir, save_dir):
    plotting.plot_loss_compare(load_dir, save_dir, [-3, 9], xgboost=True)


def calibration(load_dir_list, name_list, save_dir):
    prob_pred_list = []
    prob_true_list = []

    for load_dir in load_dir_list:
        prob_pred_part = np.load(os.path.join(load_dir, "calibration_prob_pred.npy"))
        prob_true_part = np.load(os.path.join(load_dir, "calibration_prob_true.npy"))

        prob_pred_list.extend(prob_pred_part)
        prob_true_list.extend(prob_true_part)
    with open(os.path.join(load_dir, 'config.json')) as f:
        args = json.load(f)

    plotting.plot_calibration_curves(save_dir, prob_true_list, prob_pred_list, name_list, name=args['name'])


if __name__ == '__main__':
    save_dir = "Add your path here"
    os.makedirs(save_dir, exist_ok=True)

    dir_popA = "Add your path here"
    dir_popB = "Add your path here"
    dir_popC = "Add your path here"
    dir_popD = "Add your path here"
    dir_first = "Add your path here"

    dir_compareA = "Add your path here"
    dir_compareB = "Add your path here"
    dir_compareC = "Add your path here"
    dir_compareD = "Add your path here"

    first_figure(dir_first, save_dir)
    loss_and_coverage(dir_popA, save_dir)
    loss_and_coverage(dir_popB, save_dir)
    loss_and_coverage(dir_popC, save_dir)
    loss_and_coverage(dir_popD, save_dir)

    loss_compare(dir_compareA, save_dir)
    loss_compare(dir_compareB, save_dir)
    loss_compare(dir_compareC, save_dir)
    loss_compare(dir_compareD, save_dir)

    name_list = ["Logistic", "XGBoost"]
    calibration([dir_popA], name_list, save_dir)
    calibration([dir_popB], name_list, save_dir)
    calibration([dir_popC], name_list, save_dir)
    calibration([dir_popD], name_list, save_dir)
