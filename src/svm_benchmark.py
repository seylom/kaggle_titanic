'''
Created on Nov 1, 2012

@author: SeylomA
'''

import numpy as np
import loader
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold

if __name__ == "__main__":
    # added for multiprocessor support in windows.
    
    train_X, train_Y, test = loader.load_data()
    
    n_samples = len(train_Y)
    
    ###############################################################################
#    # Set the parameters by cross-validation
#    tuned_parameters = [{'kernel': ['poly'], 'gamma': [2 ** -9, 2 ** -8.25, 2 ** -8.5, 2 ** -8.25, 2 ** -7],
#                         'C': [2 ** 1.8, 2 ** 2.2, 2 ** 2.4, 2 ** 2 ** 2.26, 2 ** 2 ** 2.8, 2 ** 3], 'degree':[3, 4]}]

#    # Set the parameters by cross-validation
    param_grid = [{'kernel': ['rbf'], 'gamma': [2 ** -15, 2 ** -13],
                         'C': [2 ** -4, 2 ** -2], 'degree':[3]}]

    
    X_train, X_test, y_train, y_test = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_fraction=0.3, random_state=1)
    #
    # # We learn the digits on the first half of the digits
    # classifier.fit(train[:n_samples / 10], target[:n_samples / 10])
    
    ##########################################################
    # maximization of outcome
    print "# Tuning hyper-parameters for maximum outcone"
    print
    
    clf = GridSearchCV(SVC(verbose=True), param_grid=param_grid, n_jobs=4, cv=StratifiedKFold(y=y_train, k=3))
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
    np.savetxt('Data/svm_benchmark_cv3_sub2.csv', preds, delimiter=',', fmt='%d')
    
    # print "Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted))
    # print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
    

