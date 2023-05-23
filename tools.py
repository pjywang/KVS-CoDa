import pickle
import warnings

import numpy as np
import pandas as pd
from kfda import Kfda
from scipy.spatial.distance import pdist
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC

import ccm_coda

warnings.filterwarnings("ignore")


def Sampling(n, dim, mean_a=0., mean_t=0., sigma_a=1., sigma_t=2., diff_ratio=0.1,
             b=np.log(5), seed=None, safe=0.3):
    """
    Synthetic data generation. See Section 5.1 of our paper.
    b: effect size
    """
    X = np.zeros((n, dim))

    # Local random generator
    Rand = np.random.RandomState(seed=seed)

    # Choosing taxa to give difference in abandance
    true_num = round(diff_ratio * dim)
    diff_taxa = np.sort(Rand.choice(dim, true_num, replace=False))

    # a_i and t_j <- Modified from Te Beest et al. (2021)
    # a, t = 0 <- 50% of zeros
    # a = -1.5, t = -0.5 <- 70% of zeros
    A = Rand.normal(mean_a, sigma_a, size=n)
    T = Rand.normal(mean_t, sigma_t, size=dim)

    # To avoid imposing effects on too deficient taxa / safe: threshold ratio of T
    # Recommended to use safe=0.3
    if safe:
        diff_taxa = np.sort(Rand.choice(np.argsort(T)[round(safe * dim):], true_num, replace=False))

    for i in range(n):
        # While loop to resample rows when it consists entirely of zeros
        while True:
            for j in range(dim):
                temp = A[i] + T[j]

                # Give difference for i >= n / 2
                if i >= n / 2:
                    if j in diff_taxa:
                        temp += b * Rand.choice([-1, 1])
                mu = np.exp(temp)

                # Sampling counts
                X[i, j] = Rand.negative_binomial(1, 1 / (mu + 1))

            # Break the while loop if there is a nonzero element
            if (X[i, :] != 0).any():
                break

    # Taxa present in less than 2 samples are removed (to use GBM zero replacement)
    # We observed that the true signal variables are not removed when safe > 0.1
    col_sum = np.sum(X > 0, axis=0)
    X_df = pd.DataFrame(X)
    removed_cols = X_df.columns[col_sum < 2]
    X_df.drop(removed_cols, inplace=True, axis=1)

    return X_df, diff_taxa


# Produces true positives and its number, False discovery, undetected
def summary_of_discovery(selected_feature, true):
    print("Num_feature:", len(selected_feature))

    # True positives
    mask = np.isin(selected_feature, true)
    true_pos = selected_feature[mask]
    print("\t{} features are properly selected:".format(len(true_pos)), true_pos)

    # False discovery
    print("\tFalse discoveries:", selected_feature[np.logical_not(mask)])

    # Undetected
    print("\tNot discovered:", np.setdiff1d(true, selected_feature), "\n")

    return len(true_pos)


# Compositional projection
def projection(X, selected_feats, method='compo', radial=False):
    X = X[:, selected_feats]
    if method == 'subcompo':
        X = X / np.sum(X, axis=1)[:, None]
    elif method == 'compo':
        X = np.c_[X, 1 - np.sum(X, axis=1)]

    if radial:
        X = X / np.linalg.norm(X, axis=1)[:, None]
    return X


# Summarizes variable selection results during CV; dim: number of selected variables
def selection_wrapper(selection_results, dim):
    selection_results = selection_results.astype(int)
    N_TRIALS, cvfolds, num_features = selection_results.shape

    counts_per_iteration = np.zeros((N_TRIALS, dim), dtype=int)
    feats_per_iteration = np.zeros((N_TRIALS, num_features), dtype=int)

    final_vote = np.zeros(dim, dtype=int)

    for i in range(N_TRIALS):
        for j in range(cvfolds):
            counts_per_iteration[i, selection_results[i, j, :]] += 1

        feats_per_iteration[i, :] = np.sort(np.argsort(counts_per_iteration[i])[:num_features])

        final_vote[feats_per_iteration[i, :]] += 1

    final_selection = np.sort(np.argsort(counts_per_iteration.sum(0))[::-1][:num_features])
    print("Selected features are:", final_selection)

    # Use .keys() method to check the keys of result..
    result = {'counts_per_iter': counts_per_iteration,
              'selfeats_per_iter': feats_per_iteration,
              'final_vote_per_iter': final_vote,
              'Selected_features': final_selection}
    return result


def get_model(model_name, **kwds):
    """
    Returns sklearn-based model based on the string clf, with its kwd parameters.
    To be added..
    """
    if model_name == 'svm':
        model = SVC(**kwds)
    elif model_name == 'kfda':
        model = Kfda(**kwds)
    elif model_name == 'krr':
        model = KernelRidge(**kwds)
    elif model_name == 'linearsvm':
        model = LinearSVC(**kwds)
    elif model_name == 'logistic':
        kwds.pop('gamma')
        model = LogisticRegressionCV(**kwds)

    return model


def get_params(model_name, do_alpha, no_gamma, sigma):
    parameters = [
        {"kernel": ['rbf'], "gamma": np.array([(2 ** i) / (2 * sigma ** 2) for i in range(-16, 17, 2)]),
         "C": np.array([10 ** i for i in np.arange(-5, 5, 1.)])}]
    if model_name not in {'svm', 'linearsvm', 'svr'}:
        parameters[0].pop('C')
    if do_alpha and model_name == 'krr':
        parameters[0]['alpha'] = np.array([10 ** i for i in np.arange(-5, 6, 1.)])
    if no_gamma:
        parameters[0]["gamma"] = [1 / (2 * sigma ** 2)]
    if model_name == 'linearsvm' or model_name == 'logistic':
        parameters[0].pop("gamma")
        parameters[0].pop("kernel")
    if not do_alpha and model_name == 'svm':
        parameters[0].pop('C')

    return parameters


def get_fold(folds, seed, type_Y="real-valued", shuffle=True, grouping=False):
    """
    grouping: to do next..
    """
    cv = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)
    if type_Y != "real-valued":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cv


def oneSE_model(model, model_name, do_alpha):
    inner_scores = model.cv_results_['mean_test_score']
    mask = inner_scores > (np.max(inner_scores) - np.std(inner_scores) / np.sqrt(len(inner_scores)))
    ga = model.cv_results_['param_gamma'].astype(float)
    ga_can = ga[mask]
    # (min_gamma, max_alpha) or (alpha_max, g_min)
    min_gamma = np.min(ga_can)
    min_gammas = (ga_can == min_gamma)
    if do_alpha:
        if model_name == 'svm':
            reg_name = 'param_C'
        else:
            reg_name = 'param_alpha'
        al = model.cv_results_[reg_name].astype(float)
        al_can = al[mask]
        # 1. (min_gamma, max_alpha)
        max_alpha = np.max(al_can[min_gammas])
        score1 = inner_scores[np.logical_and(ga == min_gamma, al == max_alpha)]

        # 2. (alpha_max, g_min)
        alpha_max = np.max(al_can)
        max_alphas = (al_can == alpha_max)
        g_min = np.min(ga_can[max_alphas])
        score2 = inner_scores[np.logical_and(ga == g_min, al == alpha_max)]

        if score1 < score2:
            min_gamma, max_alpha = g_min, alpha_max
    else:
        # max_alphas = ga_can > 0
        max_alpha = 1
    # min_gamma = np.min(ga_can[max_alphas])
    # Refit
    if model_name == 'svm':
        fitted_model = get_model(model_name, C=max_alpha, gamma=min_gamma, kernel='rbf')
    else:
        fitted_model = get_model(model_name, alpha=max_alpha, gamma=min_gamma, kernel='rbf')
    return fitted_model


def cv_score(X, Y, type_Y, num_features, model_name, savepath, proj_method='compo',  # data and classification model
             epsilon=0.001, learning_rate=0.001, iterations=1000,  # params for KFS
             outer_folds=5, inner_folds=5, N_TRIALS=10, scoring=None,
             do_alpha=False, no_gamma=False, radial=False, oneSE=False, verbose=False,
             **krgs):
    """
    type_Y: binary, categorical, real-valued
    proj_method: 'compo' and 'subcompo'
    model_name: 'svm', 'kfda', 'krr', 'linearsvm', 'logistic',... name of model to be fitted after feature selection
    savepath: path name to save results
    inner_folds: number of folds for inner cv process for parameter optimization.
                 if None, we do not implement inner CV for parameter choice
    scoring: sklearn scoring scheme; 'accuracy', 'roc_auc', 'f1', ..,
             Default value is 'accuracy'
    radial: do radial transform after variable selection
    **krgs: user-defined parameters for models

    Note: only rbf kernel is used in this code (will be revised soon)
    """
    score_all = list()
    selection_results = np.zeros((N_TRIALS, outer_folds, num_features))  # space to store all selected_feats

    '''
    Define get_default_scoring function
    '''
    if scoring is None:
        if type_Y == "real-valued":
            scoring = 'neg_mean_squared_error'
        else:
            scoring = 'accuracy'

    for _ in range(1, N_TRIALS + 1):
        print("\n", _, "th cv experiment...")
        score = list()

        # Outer cv folds
        Outer_cv = get_fold(outer_folds, seed=_, type_Y=type_Y, shuffle=True)

        # Outer cv loop
        for i, (train_idx, test_idx) in enumerate(Outer_cv.split(X, Y)):
            print(f"\n Fold {i}:")
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            # Feature selection
            rank, w = ccm_coda.ccm(X_train, Y_train, num_features, type_Y, epsilon, learning_rate=learning_rate,
                                   iterations=iterations, init=None, kernel='gaussian', verbose=verbose)
            selected_feats = np.sort(np.argsort(rank)[:num_features])
            selection_results[_ - 1, i, :] = selected_feats

            # theses prints will be modified for other datasets
            print('Selected features are: {}'.format(selected_feats))
            print('Selected weights are: {}'.format(w[selected_feats]))

            # -------------------------- Selection done -----------------------------------------

            sigma = np.median(pdist(X_train)) / np.sqrt(2)

            # projection with feature selection result
            X_proj = projection(X_train, selected_feats, method=proj_method, radial=radial)
            if radial:
                pd = pdist(X_proj)
                # take median among nonzero pd (there may be many zeros)
                sigma = np.median(pd[pd > 0]) / np.sqrt(2)

            if inner_folds is None:
                # No cross-validation on train set; one can input the model params via **krgs
                model = get_model(model_name, gamma=(2 * sigma ** 2), **krgs)
                model.fit(X_proj, Y_train)
                fitted_model = model
            else:
                # Inner CV to decide hyperparameters
                # Parameter grid
                parameters = get_params(model_name, do_alpha, no_gamma, sigma)

                # inner cv fold
                inner_cv = get_fold(inner_folds, seed=_ * N_TRIALS + i, type_Y=type_Y, shuffle=True)

                # Grid search
                estimator = get_model(model_name, **krgs)
                model = GridSearchCV(estimator=estimator, param_grid=parameters, scoring=scoring,
                                     cv=inner_cv, n_jobs=-1)
                model.fit(X_proj, Y_train)

                # Perform 1se rule to avoid overfitting (only use for svm, or krr)
                if oneSE:
                    fitted_model = oneSE_model(model, model_name, do_alpha)
                    fitted_model.fit(X_proj, Y_train)
                    print("Best params obtained by cv:", fitted_model.get_params())
                else:
                    fitted_model = model.best_estimator_
                    print("Best params obtained by cv:", model.best_params_)

            if scoring != 'roc_auc':
                scorehere = get_scorer(scoring)._score_func(Y_test,
                                                            fitted_model.predict(projection(X_test, selected_feats)))
            else:
                # roc_auc requires decision_function outputs
                scorehere = get_scorer(scoring)._score_func(Y_test,
                                                            fitted_model.decision_function(
                                                                projection(X_test, selected_feats)))
            score.append(scorehere)
            print("Score at this fold:", scorehere)

        score_all.append(score)
    print(np.mean(score_all), "\n")

    # Save
    # if inner_folds is not None:
    filename = "./results/real_data/{}/{}_{}_{}_1se{}".format(
        savepath, model_name, scoring, num_features, oneSE)
    if radial:
        filename += '_radial'
    filename += ".pickle"
    with open(filename, 'wb') as f:
        pickle.dump((score_all, selection_results), f)

    return np.mean(score_all)


# To avoid time-consuming feature selection for already selected data
def cv_with_selection(X, Y, type_Y, selection_results, savepath, proj_method='compo', model_name='svm',
                      inner_folds=5, scoring=None, do_alpha=False, no_gamma=False, radial=False,
                      oneSE=False, **krgs):
    score_all = list()
    N_TRIALS, outer_folds, num_features = selection_results.shape
    selection_results = selection_results.astype(int)

    if scoring is None:
        if type_Y == "real-valued":
            scoring = 'neg_mean_squared_error'
        else:
            scoring = 'accuracy'

    for _ in range(1, N_TRIALS + 1):
        print("\n", _, "th cv experiment...")
        score = list()

        # Outer cv folds
        Outer_cv = get_fold(outer_folds, seed=_, type_Y=type_Y, shuffle=True)

        # Outer cv loop
        for i, (train_idx, test_idx) in enumerate(Outer_cv.split(X, Y)):
            print(f"\n Fold {i}:")
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            sigma = np.median(pdist(X_train))

            selected_feats = selection_results[_ - 1, i, :]
            print('Selected features are: {}'.format(selected_feats))

            # projection with feature selection result
            X_proj = projection(X_train, selected_feats, method=proj_method, radial=radial)
            if radial:
                pd = pdist(X_proj)
                # take median among nonzero pd (there may be many zeros)
                sigma = np.median(pd[pd > 0]) / np.sqrt(2)

            if inner_folds is None:
                model = get_model(model_name, gamma=1 / (2 * sigma ** 2), **krgs)
                model.fit(X_proj, Y_train)
                fitted_model = model
            else:
                # Inner CV to decide hyperparameters
                # Parameter grid
                parameters = get_params(model_name, do_alpha, no_gamma, sigma)

                # inner cv fold
                inner_cv = get_fold(inner_folds, seed=_ * N_TRIALS + i, type_Y=type_Y, shuffle=True)

                # Grid search
                estimator = get_model(model_name, **krgs)
                model = GridSearchCV(estimator=estimator, param_grid=parameters, scoring=scoring,
                                     cv=inner_cv, n_jobs=-1)
                model.fit(X_proj, Y_train)

                # Perform 1se rule to avoid overfitting (only use for svm, or krr)
                if oneSE:
                    fitted_model = oneSE_model(model, model_name, do_alpha)
                    fitted_model.fit(X_proj, Y_train)
                    print("Best params obtained by cv:", fitted_model.get_params())
                else:
                    fitted_model = model.best_estimator_
                    print("Best params obtained by cv:", model.best_params_)

            if scoring != 'roc_auc':
                scorehere = get_scorer(scoring)._score_func(Y_test,
                                                            fitted_model.predict(projection(X_test, selected_feats)))
            else:
                # roc_auc requires decision_function outputs
                scorehere = get_scorer(scoring)._score_func(Y_test,
                                                            fitted_model.decision_function(
                                                                projection(X_test, selected_feats)))
            score.append(scorehere)
            print("Score at this fold:", scorehere)
        score_all.append(score)
    print(np.mean(score_all), "\n")

    # Save
    # if inner_folds is not None:
    filename = "./results/real_data/{}/{}_{}_{}_1se{}".format(
        savepath, model_name, scoring, num_features, oneSE)
    if radial:
        filename += '_radial'
    filename += ".pickle"
    with open(filename, 'wb') as f:
        pickle.dump((score_all, selection_results), f)

    return np.mean(score_all)


if __name__ == "__main__":
    print("Hello world!")
