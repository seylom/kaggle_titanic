'''
Created on Nov 1, 2012

@author: SeylomA
'''

import numpy as np
import loader
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier

if __name__ == "__main__":
    # added for multiprocessor support in windows.
    
    train_X, train_Y, test = loader.load_data()
    
    n_samples = len(train_Y)
    
    ###############################################################################
    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators': [100, 200, 500, 1000, 2000, 3000], 'min_samples_split':[1, 2, 3, 4, 5, 10, 15, 20],
                         'max_depth':[2, 3, 4] ,
                        'max_depth':[0.1, 0.01, 0.05],
                         'min_samples_leaf':[1, 5, 10]}]
    
    # Create a classifier: a support vector classifier
    # classifier = svm.SVC(C=2.82, cache_size=2000, coef0=0.0, gamma=0.0078, kernel='rbf',
    #                      probability=False, shrinking=True, tol=0.001, verbose=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_fraction=0.3, random_state=0)
    #
    # # We learn the digits on the first half of the digits
    # classifier.fit(train[:n_samples / 10], target[:n_samples / 10])
    
    ##########################################################
    # maximization of outcome
    print "# Tuning hyper-parameters for maximum outcone"
    print
    
    clf = GridSearchCV(GradientBoostingClassifier(random_state=0, subsample=0.5), tuned_parameters, n_jobs=2)
    clf.fit(X_train, y_train, cv=10)
    
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
    np.savetxt('Data/gbr_benchmark_cv10_sub3.csv', preds, delimiter=',', fmt='%d')
    
    # print "Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted))
    # print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
    
