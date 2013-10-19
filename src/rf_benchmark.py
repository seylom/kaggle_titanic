'''
Created on Nov 1, 2012

@author: SeylomA
'''

import numpy as np
import loader
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble.forest import RandomForestClassifier
import matplotlib.pylab as pl

i_seed = 1234

def create_rf_benchmark():
    # added for multiprocessor support in windows.
    
    train_X, train_Y, test = loader.load_data_one_hot()
    
    n_samples = len(train_Y)
    
    ###############################################################################
    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators':[10,20,50], 'min_samples_split':[1], 'oob_score':[False, True],
                        'max_depth':[1,5,10],
                        'compute_importances' : [True],
                        'min_samples_leaf':[1,5]}]
    
    # Create a classifier: a support vector classifier
    # classifier = svm.SVC(C=2.82, cache_size=2000, coef0=0.0, gamma=0.0078, kernel='rbf',
    #                      probability=False, shrinking=True, tol=0.001, verbose=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        train_X[:n_samples], train_Y[:n_samples], test_size=0.3, random_state=1)
    #
    # # We learn the digits on the first half of the digits
    # classifier.fit(train[:n_samples / 10], target[:n_samples / 10])
    
    ##########################################################
    # maximization of outcome
    print "# Tuning hyper-parameters for maximum outcone"
    print
    
#    clf = RandomForestClassifier(bootstrap=True, compute_importances=True,
#            criterion='gini', max_depth=5, max_features='auto',
#            min_density=0.1, min_samples_leaf=5, min_samples_split=1,
#            n_estimators=150, n_jobs=1, oob_score=False, random_state=None,
#            verbose=True)
#    clf.fit(X_train, y_train)
    
    
    clf = GridSearchCV(RandomForestClassifier(verbose=True), tuned_parameters, n_jobs=6) #, score_func = f1_score)
    clf.fit(X_train, y_train, cv=3)
    
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
    print "Confusion matrix:\n%s" % confusion_matrix(y_true, y_pred)
    print
    
#    importances = clf.best_estimator_.feature_importances_
#    std = np.std([tree.feature_importances_ for tree in clf.best_estimator_.estimators_],
#                 axis=0)
#    indices = np.argsort(importances)[::-1]
#    
#    # Print the feature ranking
#    print "Feature ranking:"
#    
#    num_features = X_train.shape[1]
#    
#    for f in xrange(num_features):
#        print "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])
#    
#    # Plot the feature importances of the forest
# 
#    pl.figure()
#    pl.title("Feature importances")
#    pl.bar(xrange(num_features), importances[indices],
#           color="r", yerr=std[indices], align="center")
#    pl.xticks(xrange(num_features), indices)
#    pl.xlim([-1, 10])
#    pl.show()
    
    preds = clf.predict(test)
    np.savetxt('submissions/rf_benchmark_06_18_2013.csv', preds, delimiter=',', fmt='%d')
    
    # print "Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted))
    # print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
    
if __name__ == "__main__":
    create_rf_benchmark()
