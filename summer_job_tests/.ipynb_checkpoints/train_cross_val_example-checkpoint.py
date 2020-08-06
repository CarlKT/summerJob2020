import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
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

    #Cross validate
    cv_results = cross_validate(log_reg,
                                df_features,
                                df_labels,
                                return_estimator=True,
                                cv=num_iterations,
                                n_jobs=num_procs,
                                scoring=scoring)

    #show processing time
    print('Total fit time: ' + str(sum(cv_results['fit_time'])) + '\n'
          'Total score time: ' + str(sum(cv_results['score_time'])))

    #Retrieve results
    results = {}

    results['all_coefficients'] = [
        cv_results['estimator'][i].coef_.flatten()
        for i in range(num_iterations)
    ]
    results['all_probas'] = [
        estimator.predict_proba(features)
        for estimator in cv_results['estimator']
    ]
    results['all_rocs'] = cv_results['test_rocs']
    results['all_precisions'] = cv_results['test_custom_precision']

    #results['all_test_labels'] = ???

    return results


print(train_classifier(features, labels, num_iterations=10))
