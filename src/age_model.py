'''
Created on Nov 1, 2012

@author: SeylomA
'''
 
import loader
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble.forest import  RandomForestRegressor
from sklearn.ensemble import  GradientBoostingRegressor
import matplotlib.pylab as pl
import numpy as np
import csv

#from sklearn.ensemble import  GradientBoostingRegressor

i_seed = 1234

def compute_missing_age():
    # added for multiprocessor support in windows.
    
    train_X, train_Y, train_miss, test_miss = loader.load_age_data()
    
    n_samples = len(train_Y)
    
    ###############################################################################
    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators':[10,50,100,500], 'min_samples_split':[1,2,3,4, 5], 'learning_rate':[0.01, 0.1,1],
                        'max_depth':[1,5],
                        'loss': ['ls'],
                        'min_samples_leaf':[1, 5, 10]}]
    
    params = {'n_estimators': 300, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
    
    #clf = GradientBoostingRegressor(**params)
    
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
    
    clf = GridSearchCV(GradientBoostingRegressor(verbose=True), 
                        tuned_parameters, n_jobs=4, 
                        score_func= mean_squared_error)
    clf.fit(X_train, y_train, cv=3)
    
    clf.fit(X_train, y_train)
    
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)
    
    print "Best parameters set found on development set:"
    print
    print clf.best_estimator_
    
    # Plot training deviance
    
    # compute test set deviance
    test_score = np.zeros((clf.best_params_['n_estimators'],), dtype=np.float64)
    
    for i, y_pred in enumerate(clf.best_estimator_.staged_decision_function(X_test)):
        test_score[i] = clf.best_estimator_.loss_(y_test, y_pred)
    
    pl.figure(figsize=(12, 6))
    pl.subplot(1, 2, 1)
    pl.title('Deviance')
    pl.plot(np.arange(clf.best_params_['n_estimators']) + 1, clf.best_estimator_.train_score_, 'b-',
            label='Training Set Deviance')
    pl.plot(np.arange(clf.best_params_['n_estimators']) + 1, test_score, 'r-',
            label='Test Set Deviance')
    pl.legend(loc='upper right')
    pl.xlabel('Boosting Iterations')
    pl.ylabel('Deviance')
    
    pl.show()
    
    age_preds_train = clf.predict(train_miss)
    age_preds_train = np.around(age_preds_train,1)
    
    age_preds_test = clf.predict(test_miss)
    age_preds_test = np.around(age_preds_test,1)
    
    
    #writer = csv.writer(open('missing_age_predict.csv','w'),delimiter=',')
    #writer.writerows([x for x in age_preds]);
    np.savetxt("data/missing_age_train_predict.csv", age_preds_train,fmt='%10.1f', delimiter=",")
    np.savetxt("data/missing_age_test_predict.csv", age_preds_test,fmt='%10.1f', delimiter=",")
    
#     ###############################################################################
#     # Plot feature importance
#     feature_importance = clf.feature_importances_
#     # make importances relative to max importance
#     feature_importance = 100.0 * (feature_importance / feature_importance.max())
#     sorted_idx = np.argsort(feature_importance)
#     pos = np.arange(sorted_idx.shape[0]) + .5
#     pl.subplot(1, 2, 2)
#     pl.barh(pos, feature_importance[sorted_idx], align='center')
#     pl.yticks(pos, boston.feature_names[sorted_idx])
#     pl.xlabel('Relative Importance')
#     pl.title('Variable Importance')
#     pl.show() 
   
if __name__ == "__main__":
    compute_missing_age()
