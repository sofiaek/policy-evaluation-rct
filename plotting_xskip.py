import json
import os
import numpy as np
import plotting


def loss_and_coverage(load_dir, save_dir):
    plotting.plot_loss_and_coverage_paper(load_dir, save_dir)


def weights(load_dir_missing_list, load_dir_all, save_dir, ignore_logistic=False):
    prob_missing_list = []
    prob_list = []
    name_list = []

    prob = np.load(os.path.join(load_dir_all, "prob_est.npy"))

    for load_dir in load_dir_missing_list:
        prob_missing = np.load(os.path.join(load_dir, "prob_est.npy"))

        with open(os.path.join(load_dir, 'config.json')) as f:
            args = json.load(f)

        for i, model in enumerate(args['weight_models']):
            if model == 'true':
                continue
            if ignore_logistic and model == 'logistic':
                continue
            if ignore_logistic:
                name = '$' + args['skip_x'] + '$ '
            else:
                name = '$' + args['skip_x'] + '$ ' + model
            name_list.append(name)

            prob_missing_list.extend(prob_missing[i - 1, :].reshape(1, -1))
            prob_list.extend(prob[i - 1, :].reshape(1, -1))

        # name_list.append(args['skip_x'])
    # name_list = [r'$' + label_i + '$' for label_i in name_list]
    # name_list = ['logistic', 'xgboost']

    plotting.plot_weight_curves(save_dir, prob_missing_list, prob_list, name_list, args['name'])


def weights_nhanes(load_dir_missing_list, load_dir_all, save_dir):
    prob_missing_list = []
    prob_list = []
    name_list = []

    prob = np.load(os.path.join(load_dir_all, "prob_est.npy"))

    for load_dir in load_dir_missing_list:
        prob_missing = np.load(os.path.join(load_dir, "prob_est.npy"))

        prob_missing_list.extend(prob_missing)
        prob_list.extend(prob)

        with open(os.path.join(load_dir, 'config.json')) as f:
            args = json.load(f)

        name_list.append(args['skip_x'])

    plotting.plot_weight_curves(save_dir, prob_missing_list, prob_list, name_list, args['name'])


if __name__ == '__main__':
    save_dir = "Add your path here"
    os.makedirs(save_dir, exist_ok=True)

    dir_all = "Add your path here"
    dir_missing_age = "Add your path here"
    dir_missing_income = "Add your path here"
    dir_missing_edu = "Add your path here"

    weights_nhanes([dir_missing_age,
                    dir_missing_income,
                    dir_missing_edu],
                   dir_all, save_dir)

    dir_all_popA = "Add your path here"
    dir_missing_x0_popA = "Add your path here"
    dir_missing_x1_popA = "Add your path here"

    dir_all_popB = "Add your path here"
    dir_missing_x0_popB = "Add your path here"
    dir_missing_x1_popB = "Add your path here"

    dir_all_popC = "Add your path here"
    dir_missing_x0_popC = "Add your path here"
    dir_missing_x1_popC = "Add your path here"

    dir_all_popD = "Add your path here"
    dir_missing_x0_popD = "Add your path here"
    dir_missing_x1_popD = "Add your path here"

    weights([dir_missing_x0_popA, dir_missing_x1_popA], dir_all_popA, save_dir)
    weights([dir_missing_x0_popB, dir_missing_x1_popB], dir_all_popB, save_dir)
    weights([dir_missing_x0_popC, dir_missing_x1_popC], dir_all_popC, save_dir)
    weights([dir_missing_x0_popD, dir_missing_x1_popD], dir_all_popD, save_dir)
