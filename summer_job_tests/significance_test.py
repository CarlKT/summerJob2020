### imports ###
import argparse
import numpy as np
import os
import time
import pandas as pd
from sklearn import preprocessing
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import scipy
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from joblib import Parallel, delayed
from sklearn.datasets import load_iris
#from train_cross_val_example import train_classifier_Carl
### functions ###


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


def calc_feature_pvals(features,
                       labels,
                       test_size=0.2,
                       num_iterations=5,
                       num_procs=4):
    pvals = []
    num_motifs = features.shape[1]
    # split data into training and test sets
    scaler = preprocessing.StandardScaler()

    # standardize features
    standardized_features = pd.DataFrame(scaler.fit_transform(features))
    standardized_features.columns = features.columns.values
    standardized_features.index = features.index.values

    for i in range(num_iterations):
        training_features, test_features, training_labels, test_labels = train_test_split(
            features, labels, test_size=test_size)

        # standardize training features
        standardized_training_features = pd.DataFrame(
            scaler.fit_transform(training_features))
        standardized_training_features.columns = training_features.columns.values
        standardized_training_features.index = training_features.index.values

        #  Train affinity classifier
        classifier = sklearn.linear_model.LogisticRegression(
            penalty='l1', solver='liblinear')

        classifier.fit(standardized_training_features, training_labels)
        # score predictions

        probas = classifier.predict_proba(
            standardized_features)  # [[p_false, p_true]...]
        overall_log_likelihood = calc_model_log_likelihood(probas, labels)
        iter_pvals = []

        iter_pvals = Parallel(n_jobs=num_procs)(
            delayed(train_perturbed_classifier)(
                standardized_features, labels, standardized_training_features,
                training_labels, motif_to_drop, overall_log_likelihood)
            for motif_to_drop in features.columns.values)

        pvals.append(iter_pvals)

    return pvals


def train_perturbed_classifier(features, labels, training_features,
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


def read_labels(label_path):
    '''
    reads label files created by create_features.py and returns a pandas Series representation
    '''
    indices = []
    vals = []
    with open(label_path) as f:
        data = f.readlines()
    for line in data:
        tokens = line.strip().split()
        indices.append(tokens[0])
        if tokens[1] == '1':
            vals.append(True)
        else:
            vals.append(False)
    to_return = pd.Series(vals, index=indices)
    return to_return


def write_test_results(features, pvals, output_path):
    pval_dict = dict(zip(range(len(pvals)), pvals))
    pval_frame = pd.DataFrame(data=pval_dict, index=features.columns.values)
    pval_frame.to_csv(output_path + '/significance.tsv', sep='\t')


def calc_feature_pvals_v2(features,
                          labels,
                          results,
                          num_iterations=5,
                          num_procs=1):

    pvals = []
    num_motifs = features.shape[1]
    scaler = preprocessing.StandardScaler()

    # standardize features
    standardized_features = pd.DataFrame(scaler.fit_transform(features))
    standardized_features.columns = features.columns.values
    standardized_features.index = features.index.values

    #define previously used split technique
    sss = StratifiedShuffleSplit(n_splits=num_iterations,
                                 test_size=int(len(labels) / num_iterations),
                                 random_state=0)

    #calculate pval for each iteration of train_classifier
    for i in range(num_iterations):

        print('Iteration %s' % (i + 1))
        #score predictions
        probas = results[3][i]
        overall_log_likelihood = calc_model_log_likelihood(probas, labels)
        iter_pvals = []

        #retrieve split information
        training_indices = list(sss.split(standardized_features, labels))[i][0]

        training_labels = [labels[index] for index in training_indices]
        #not sure about this
        training_standardized_features = pd.DataFrame(
            [standardized_features.iloc[index] for index in training_indices],
            index=training_indices)

        iter_pvals = Parallel(n_jobs=num_procs)(
            delayed(train_perturbed_classifier)(
                standardized_features, labels, training_standardized_features,
                training_labels, motif_to_drop, overall_log_likelihood)
            for motif_to_drop in standardized_features.columns.values)

        pvals.append(iter_pvals)

    return pvals


features, labels = load_iris(return_X_y=True)
results = train_classifier(features, labels)
print(
    calc_feature_pvals_v2(features,
                          labels,
                          results,
                          num_iterations=5,
                          num_procs=2))
