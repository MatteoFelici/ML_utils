import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import cross_validate
from time import time

from hyperopt import fmin
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials



class SklearnHyperopt(object):
    """Wrapper to utilize the Bayesian hyperparameter optimization
    with a scikit-learn model. It uses the Tree-structured Parzen Estimator
    as surrogate function.
    """

    def __init__(self, n_jobs=1):
        """
        Parameters
        ----------
        n_jobs : int, optional (default 1)
            number of parallel cores that the optimization algorithm can use
        """

        self.alg = tpe.suggest
        self.n_jobs = n_jobs


    def define_loss(self, train_loss, test_loss, loss_type):
        """Definition of the loss. The possibilities are:
        - type 'test': the loss is the one calculated on the test set
        - type 'delta': the loss is calculated as
                test_loss * 10**(test_loss - train_loss)
          in order to penalize overfitting

        Parameters
        ----------
        train_loss : float
            loss on the train set
        test_loss : float
            loss on the test set
        loss_type : str - one from ['test', 'delta']
            type of loss to evaluate the hyperparameters sample
        Returns
        -------
        loss : float
            the loss based on train-test and type
        """

        if loss_type == 'test':
            loss = test_loss
        elif loss_type == 'delta':
            delta = test_loss - train_loss
            loss = test_loss * max(1, np.power(10, delta))

        return loss


    def objective(self, params):
        """Objective function used by the run_opt method. This method is
        repeatedly called at each hyperopt trial with different hyperparameters.

        Parameters
        ----------
        params : dict
            The current choice for hyperparameters.

        Returns
        -------
        res_dict : dict
            Dictionary with the results of the current trial. It contains:
                - iteration: number of trial
                - train_loss: loss on train data
                - test_loss: loss on test data
                - loss: loss calculated by chosen loss type
                - params: current choice of hyperparameters
                - run_time: elapsed time for model training and evaluation
                - status: status of the trial
        """

        # Advance iteration
        self.iteration += 1

        # For some particular hyperparameters, we have to operate a manual
        # conversion to integer
        for p_name in ['n_estimators', 'max_depth']:
            if p_name in params:
                params[p_name] = int(params[p_name])

        # Start time
        start = time()
        
        # Set current parameters to classifier
        self.clf.set_params(**params)
            
        # Perform n_folds cross validation
        cv_results = cross_validate(self.clf,
                                    self.X,
                                    self.y,
                                    scoring=self.scoring,
                                    cv=self.n_folds,
                                    n_jobs=self.n_jobs,
                                    return_train_score=True)

        # End time
        end = time()
        # Calculate run time
        run_time = end - start

        # Calculate train and test score
        train_score = np.mean(cv_results['train_score'])
        test_score = np.mean(cv_results['test_score'])

        # Calculate loss from score
        # If higher_better, loss must be minimized
        if self.higher_better:
            test_loss = 1 - test_score
            train_loss = 1 - train_score
        else:
            test_loss = test_score
            train_loss = train_score
        # Calculate loss from train-test and loss type
        loss = self.define_loss(train_loss, test_loss, self.loss_type)

        # If given an output file, write results
        if self.trials_file:
            of_connection = open(out_file, 'a')
            writer = csv.writer(of_connection)
            writer.writerow([self.iteration, train_loss, test_loss,
                             test_loss - train_loss, loss, params, run_time])
        
        # Return list of results
        res_dict = {
            'iteration': self.iteration,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'loss': loss,
            'params': params,
            'run_time': run_time,
            'status': STATUS_OK
        }

        return res_dict


    def run_opt(self, clf, space, X, y,
                scoring='accuracy',
                higher_better=True,
                n_folds=3,
                max_evals=10,
                type_loss='delta',
                trials_file=None,
                random_state=1123,
                verbosity=0):
        """Given a scikit-learn classifier, run the optimization task with the
        given hyperparameters space.

        Parameters
        ----------
        clf : scikit-learn transformer
            Model from scikit-learn. It must have both fit and predict methods
        space : dict
            Dictionary with the probability distribution for each
            hyperparameter; the form is
            {hyperparameter: hp.distribution}
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        scoring : str, optional (default: 'accuracy')
            The score to use in the cross validation to assess each instance of
            hyperparameters
        higher_better : bool, optional (default: True)
            Flag to set if the scoring function chosen is better if higher
            (like accuracy) or not (like Mean Absolute Error)
        n_folds : int, optional (default: 3)
            number of folds in cross validation
        max_evals : int, optional (default: 10)
            Number of samples taken from the hyperparameters space by the
            optimization algorithm
        type_loss : str - one from ['test', 'delta'], optional (default: 'test')
            Type of loss to evaluate the hyperparameters sample.
            The possibilities are:
                - type 'test': the loss is the one calculated on the test set
                - type 'delta': the loss is calculated as
                        test_loss * 10**(test_loss - train_loss)
                  in order to penalize overfitting
        trials_file : str, optional (default: None)
            Path to a file in which save all the hyperparameters tried with
            performance indicators; if None, the file is not created
        random_state : int, optional (default: 1123)
            The seed used by the random number generator
        verbosity : int, optional (default: 0)
            Controls the verbosity when fitting and predicting

        Returns
        -------
        best : dict
            hyperparameters of the best trial found
        bayes_trials : hyperopt.Trials
            container with all the info about hyperparameters trials
        """

        self.clf = clf
        self.X = X
        self.y = y
        self.scoring = scoring
        self.higher_better = higher_better
        self.n_folds = n_folds
        self.loss_type = type_loss
        self.trials_file = trials_file
        
        # File to save first results
        if self.trials_file:
            of_connection = open(self.trials_file, 'w')
            writer = csv.writer(of_connection)

            # Write the headers to the file
            writer.writerow(['iteration', 'train_loss', 'test_loss', 'delta',
                             'loss', 'params', 'train_time'])
            of_connection.close()

        bayes_trials = Trials()
        self.iteration = 0

        # Run optimization
        best = fmin(fn=self.objective,
                    space=space,
                    algo=self.alg,
                    max_evals=max_evals,
                    trials=bayes_trials,
                    rstate=np.random.RandomState(random_state),
                    verbose=verbosity)

        return best, bayes_trials


def pretty_bayes(t, is_file=False):
    if is_file:
        res = pd.read_csv(t)
        res['params'] = res['params'].map(eval)
        if 'iteration' in res.columns:
            res = pd.concat([res['iteration'], res['params'].apply(pd.Series),
                             res.drop(['iteration', 'params'], 1)], axis=1)
        else:
            res = pd.concat([res['params'].apply(pd.Series),
                             res.drop('params', 1)], axis=1)
    else:
        res = pd.DataFrame([x['params'] for x in t.results])
        for k in t.results[0]:
            if k == 'params':
                continue
            res[k] = [x[k] for x in t.results]
    res['delta'] = res['test_loss'] - res['train_loss']
    res.sort_values('loss', inplace=True)
    
    return res

