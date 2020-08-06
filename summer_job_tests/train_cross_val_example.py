import numpy as np
import pandas as pd
import sklearn
import time
import scipy
import sys
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from joblib import Parallel, delayed
from sklearn.datasets import load_iris

#retrieve data (using iris as example)
features, labels = load_iris(return_X_y=True)


def train_classifier(features, labels, num_iterations=5, num_procs=1):
    """Trains and returns roc_auc_score and average_precision_score of the model.
       To use all processors, specify num_procs=-1."""

    #process data
    features = StandardScaler().fit_transform(features)

    #turn into DataFrame
    df_features = pd.DataFrame(features)
    df_labels = pd.DataFrame(labels)

    #Prevents binary label weirdness
    def custom_precision(y_true, y_score):
        """allows average precision score to be calculated with non-binary labels."""

        y_true = label_binarize(y_true, classes=np.unique(y_true))
        average_precision = average_precision_score(y_true,
                                                    y_score,
                                                    average='weighted')
        return average_precision

    #Create custom scoring dictionnary
    scoring = {
        'rocs':
        make_scorer(roc_auc_score,
                    average='weighted',
                    multi_class='ovr',
                    needs_proba=True),
        'custom_precision':
        make_scorer(custom_precision, needs_proba=True)
    }

    #Define estimator
    clf = OneVsRestClassifier(
        LogisticRegression(penalty='l1', tol=1e-8, solver='liblinear'))
    log_reg = LogisticRegression(penalty='l1', tol=1e-8, solver='liblinear')

    #define cv type
    sss = StratifiedShuffleSplit(n_splits=num_iterations,
                                 test_size=int(len(labels) / num_iterations),
                                 random_state=0)

    #Cross validate
    cv_results = cross_validate(log_reg,
                                df_features,
                                labels,
                                return_estimator=True,
                                cv=sss,
                                n_jobs=num_procs,
                                scoring=scoring)

    #access n_splits
    test_labels = []
    for i in range(num_iterations):
        train, test = tuple(list(sss.split(features, labels))[i])
    test_labels.sort()

    #show processing time
    print('Total fit time: ' + str(sum(cv_results['fit_time'])) + '\n'
          'Total score time: ' + str(sum(cv_results['score_time'])))

    #Retrieve results
    coef_array = [
        cv_results['estimator'][i].coef_.flatten()
        for i in range(num_iterations)
    ]
    all_coefficients = pd.DataFrame(coef_array).T
    #all_coefficients.index = df_features.columns.values
    all_probas = [
        estimator.predict_proba(df_features)
        for estimator in cv_results['estimator']
    ]
    all_rocs = cv_results['test_rocs']
    all_precisions = cv_results['test_custom_precision']

    print('Performance:')
    print(pd.DataFrame({'ROC': all_rocs, 'Precision Score': all_precisions}))

    #to fit with Greg's output
    results = (all_rocs, all_precisions, all_coefficients, all_probas)

    return results


def train_classifier_Carl(features,
                          labels,
                          numIterations=5,
                          num_procs=1,
                          test_size=0.5):
    """Trains and returns roc_auc_score and average_precision_score of the model.
       To use all processors, specify num_procs=-1. Test_size is useless."""

    print('training classifier...')

    #process data
    features = StandardScaler().fit_transform(features)

    #turn into DataFrame
    standardized_features = pd.DataFrame(features)
    df_labels = pd.DataFrame(labels)

    #Prevents binary label weirdness
    def custom_precision(y_true, y_score):
        """allows average precision score to be calculated with non-binary labels."""

        y_true = label_binarize(y_true, classes=np.unique(y_true))
        average_precision = average_precision_score(y_true,
                                                    y_score,
                                                    average='macro')
        return average_precision

    #Create custom scoring dictionnary
    scoring = {
        'rocs':
        make_scorer(roc_auc_score,
                    average='macro',
                    multi_class='ovr',
                    needs_proba=True),
        'custom_precision':
        make_scorer(custom_precision, needs_proba=True)
    }

    #Define estimator
    logreg = LogisticRegression(penalty='l1', tol=1e-8, solver='liblinear')

    #define cv type
    sss = StratifiedShuffleSplit(n_splits=num_iterations,
                                 test_size=int(len(labels) / num_iterations),
                                 random_state=0)

    #Cross validate
    print('cross validating')
    cv_results = cross_validate(logreg,
                                standardized_features,
                                labels,
                                return_estimator=True,
                                cv=sss,
                                n_jobs=num_procs,
                                scoring=scoring)

    #show processing time
    print('Train time: ' +
          str(sum(cv_results['fit_time']) + sum(cv_results['score_time'])))

    #Retrieve results
    coef_array = [
        cv_results['estimator'][i].coef_.flatten()
        for i in range(numIterations)
    ]
    all_coefficients = pd.DataFrame(coef_array).T
    all_coefficients.index = df_features.columns.values
    all_probas = [
        estimator.predict_proba(standardized_features)
        for estimator in cv_results['estimator']
    ]
    all_rocs = cv_results['test_rocs']
    all_precisions = cv_results['test_custom_precision']

    print('Performance:')
    print(pd.DataFrame({'ROC': all_rocs, 'Precision Score': all_precisions}))

    #to fit with Greg's output
    results = (all_rocs, all_precisions, all_coefficients, all_probas)
    return results


def calc_model_log_likelihood(probas, labels):
    log_likelihood = 0
    Y = labels.astype(float)
    for i in range(len(Y)):
        p_true = probas[i][1]
        p_false = probas[i][0]
        y = Y[i]
        prod = ((p_true)**y) * ((p_false)**(1 - y))
        log_prod = np.log(prod)
        log_likelihood += log_prod
    return log_likelihood


def calc_feature_pvals_Carl(features,
                            labels,
                            test_size=0.2,
                            num_iterations=5,
                            num_procs=4):
    pvals = []
    num_motifs = features.shape[1]

    #process data
    features = StandardScaler().fit_transform(features)

    #turn into DataFrame
    standardized_features = pd.DataFrame(features)
    df_labels = pd.DataFrame(labels)

    sss = StratifiedShuffleSplit(n_splits=num_iterations,
                                 test_size=int(len(labels) / num_iterations),
                                 random_state=0)

    results = train_classifier(features, labels, num_iterations=5)
    print(results[3])
    for i in range(num_iterations):
        probas = results[3][i]
        overall_log_likelihood = calc_model_log_likelihood(probas, labels)
        iter_pvals = []
        training_indices = list(sss.split(standardized_features, labels))[i][0]
        training_labels = [labels[index] for index in training_indices]
        standardized_training_features = pd.DataFrame(
            [standardized_features.iloc[index] for index in training_indices],
            index=training_indices)

        iter_pvals = Parallel(n_jobs=num_procs)(
            delayed(train_perturbed_classifier_Carl)(
                standardized_features, labels, standardized_training_features,
                training_labels, motif_to_drop, overall_log_likelihood)
            for motif_to_drop in standardized_features.columns.values)

        pvals.append(iter_pvals)

    return pvals


def train_perturbed_classifier_Carl(features, labels, training_features,
                                    training_labels, motif_to_drop,
                                    overall_log_likelihood):

    start = time.time()
    current_features = features.drop(motif_to_drop, axis=1, inplace=False)
    current_training_features = training_features.drop(motif_to_drop,
                                                       axis=1,
                                                       inplace=False)
    current_classifier = sklearn.linear_model.LogisticRegression(
        penalty='l1', solver='liblinear')
    current_classifier.fit(current_training_features, training_labels)
    current_probas = current_classifier.predict_proba(current_features)
    current_log_likelihood = calc_model_log_likelihood(current_probas, labels)
    stat = -2 * (current_log_likelihood - overall_log_likelihood)
    p = scipy.stats.chi2.sf(stat, df=1)
    print('tested', motif_to_drop, time.time() - start)
    return p


#print(calc_feature_pvals_Carl(features, labels, num_iterations=5))
if __name__ == '__main__':
    if sys.argv[1] == 'is_iris':
        features, labels = load_iris(return_X_y=True)
        results = train_classifier(features, labels)
        probas = results[3]
        print(probas)
    else:
        results = train_classifier(*sys.argv[2:])

