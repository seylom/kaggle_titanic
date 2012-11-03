'''
Created on Nov 1, 2012

@author: SeylomA
'''

import numpy as np
import loader
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble.forest import RandomForestClassifier

def create_rf_benchmark():
    # added for multiprocessor support in windows.
    
    train_X, train_Y, test = loader.load_data()
    
    n_samples = len(train_Y)
    
    ###############################################################################
    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators':[500, 1000], 'min_samples_split':[1, 5, 10, 20], 'oob_score':[False, True],
                        'max_depth':[ 4],
                        'min_samples_leaf':[5, 10]}]
    
    # Create a classifier: a support vector classifier
    # classifier = svm.SVC(C=2.82, cache_size=2000, coef0=0.0, gamma=0.0078, kernel='rbf',
    #                      probability=False, shrinking=True, tol=0.001, verbose=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_fraction=0.3, random_state=1)
    #
    # # We learn the digits on the first half of the digits
    # classifier.fit(train[:n_samples / 10], target[:n_samples / 10])
    
    ##########################################################
    # maximization of outcome
    print "# Tuning hyper-parameters for maximum outcone"
    print
    
    clf = GridSearchCV(RandomForestClassifier(verbose=True), tuned_parameters, n_jobs=4)
    clf.fit(X_train, y_train, cv=5)
    
    print "Best parameters set found on development set:"
    print
    print clf.best_estimator_
    print
    print "Grid scores on development set:"
    print
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (
            mean_score, scores.std() / 2, params)
    print
    
    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred)
    print
    
    # Now predict the value of the digit on the second half:
    # expected = target[n_samples / 10:]
    # predicted = classifier.predict(train[n_samples / 10:])
    
    preds = clf.predict(test)
    np.savetxt('Data/rf_benchmark_cv5_sub4.csv', preds, delimiter=',', fmt='%d')
    
    # print "Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted))
    # print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
    
if __name__ == "__main__":
    create_rf_benchmark()
