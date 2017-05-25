"""
@author: Matteo Felici - matteo.felici@quixa.it

Package of the Machine Learning utilities for feature transformation.
- grouper: creates groups of modalities
- imputer: impute missing values
- woe_transformation: transform a categorical variable into a continuous
                      one via Weight Of Evidence
"""



import pandas as pd
import numpy as np


class grouper():
    """
    Group low-percentage modalities of a variable: it creates N groups,
    N = 1 based on absolute frequency, N > 1 based also on impact on target.

    Parameters
    -----------
    top: group all the modalities except the top N (default None)
    perc: group all the modalities with frequency < perc; used if top == None
          (default 0.02)

    Methods
    -----------
    fit: calculates the groups from a training dataset
    trasform: creates the groups based on previous fitting
    fit_transform: fit and transform

    """

    def __init__(self, top=None, perc=0.02):
        self.top = top
        self.perc = perc


    def fit(self, data, columns, label='other', target=None,
            event_labels=['low', 'medium', 'high'], event_cuts=[0.6, 1.4]):
        """
        Parameters
        -----------
        data: training dataset
        columns: features to group
        label: label of the group if grouping is not based on target (default 'other')
        target: taget variable (default None); if not definded,
               the grouped modalities are put in one single bucket
        event_cuts: N-list of cuts for target impact on single modality, to create
                    N+1 buckets (only if target is defined)
                    (default [0.6, 1.4])
        event_labels: list with the N labels for the N buckets (only if target is defined),
                      its length must be 1 + length of event_cuts
                      (default ['low', 'medium', 'high'])

        """

        self.columns = columns
        self.transformations = {}

        if target == None:
            self.label = label
            self.target = None
        else:
            self.target = target
            self.event_labels = event_labels
            self.event_cuts = event_cuts
            self.event_perc = sum(data[self.target] == 1) / len(data)

            pairings = []
            pairings.append(([0, event_cuts[0]], event_labels[0]))
            for i in range(1, len(event_cuts)):
                pairings.append(([event_cuts[i-1], event_cuts[i]], event_labels[i]))
            pairings.append(([event_cuts[-1], 100], event_labels[-1]))


        for col in columns:
            a = data[col].value_counts()
            if self.top is not None:
                a = a.index.tolist()[self.top:]
            else:
                a = a[a < len(data) * self.perc].index.tolist()

            self.transformations[col] = {}

            if target == None:
                for elem in data[col].unique().tolist():
                    if elem in a:
                        self.transformations[col][elem] = self.label
                    else:
                        self.transformations[col][elem] = elem
            else:
                vals = data.groupby(col)[self.target].mean()
                for elem in data[col].unique().tolist():
                    if elem in a:
                        val = vals[elem]
                        for x in pairings:
                            if x[0][0] <= val < x[0][1]:
                                self.transformations[col][elem] = x[1]
                    else:
                        self.transformations[col][elem] = elem


    def transform(self, data):
        """
        Parameters
        -----------
        data: dataset to transform; it has to contain all the columns of the
              transformation (see self.columns)

        Output
        -----------
        data: transformed dataset
        """

        if hasattr(self, 'columns') == False:
            raise ValueError('Grouper has to be fitted yet.')
        for col in self.columns:
            if col not in data.columns:
                raise ValueError('Column %s not in input data features.' % col)
        if self.target == None:
            l = self.label
        else:
            l = self.event_labels[int(len(self.event_labels) / 2)]

        for col in self.columns:
            data[col] = data[col].apply(lambda x:
                self.transformations[col][x] if x in self.transformations[col].keys() else l)
        return data


    def fit_transform(self, data, columns, label='other', event=None,
            event_labels=['low', 'medium', 'high'], event_cuts=[0.6, 1.4]):
        """
        Parameters
        -----------
        data: training dataset to transform
        columns: features to group
        label: label of the group if grouping is not based on target (default 'other')
        event: level of the taget variable as the event (default None); if not definded,
               the grouped modalities are put in one single bucket
        event_cuts: N-list of cuts for target impact on single modality, to create
                    N+1 buckets (only if event is defined)
                    (default [0.6, 1.4])
        event_labels: list with the N labels for the N buckets (only if event is defined),
                      its length must be 1 + length of event_cuts
                      (default ['low', 'medium', 'high'])

        Output
        -----------
        data: transformed dataset
        """

        self.fit(data, columns, label=label, event=event,
                 event_labels=label, event_cuts=event_cuts)
        return self.transform(data)





def mean_imputer(data, col):
    return data[col].mean()

def median_imputer(data, col):
    return data[col].median()

def mode_imputer(data, col):
    return data[col].mode()[0]

class imputer():
    """
    Substitute missing values with the mean, median or mode of the feature.


    Parameters
    -----------
    method: default method to impute missing variables: choose between
            mean, median and mode (default)

    Methods
    -----------
    fit: calculates the groups from a training dataset
    trasform: creates the groups based on previous fitting
    fit_transform: fit and transform

    """


    def __init__(self, method='mode'):
        if method not in ('mode', 'median', 'mean'):
            raise ValueError('Method not recognised. Only available methods are'
                             ' mean, median and mode (default).')
        self.method = method
        self.imp_dict = {'mode': mode_imputer,
                         'mean': mean_imputer,
                         'median': median_imputer}
        self.fun = self.imp_dict[self.method]


    def fit(self, data, columns):
        """
        Parameters
        -----------
        data: training dataset
        columns: columns to impute. It can be a list (the method to impute is
                 self.method) or a dict in the form of {col: method, ...}
                 (method can be mean, median or mode)
        """

        self.columns = columns
        self.transformations = {}
        if type(columns) == dict:
            for col in columns:
                if data[col].dtype == 'O' and columns[col] != 'mode':
                    raise ValueError('Column %s is categoric, cannot apply'
                                     'method %s.' % (col, columns[col]))
                val = self.imp_dict(columns[col])(data, col)
                self.transformations[col] = val
        else:
            for col in columns:
                if data[col].dtype == 'O' and self.method != 'mode':
                    raise ValueError('Column %s is categoric, cannot apply'
                                     'method %s.' % (col, self.method))
                val = self.fun(data, col)
                self.transformations[col] = val


    def transform(self, data):
        """
        Parameters
        -----------
        data: data to impute; it has to contain all the columns of the
              transformation (see self.columns)

        Output
        -----------
        data: transformed dataset
        """

        if hasattr(self, 'columns') == False:
            raise ValueError('Imputer has to be fitted yet.')
        for col in self.columns:
            if col not in data.columns:
                raise ValueError('Column %s not in input data features.' % col)

        return data.fillna(self.transformations)


    def fit_transform(self, data, columns):
        """
        Parameters
        -----------
        data: training dataset to transform
        columns: columns to impute. It can be a list (the method to impute is
                 self.method) or a dict in the form of {col: method, ...}
                 (method can be mean, median or mode)

        Output
        -----------
        data: transformed dataset
        """

        self.fit(data, columns)
        return self.transform(data)



def sigmoid(x):
    if x == 0:
        return -9999
    else:
        return np.log(x / (1-x))


class woe_transformation():
    """
    Re-codes categorical features using their Weight Of Evidence
    based on target value.
    The impact of a modality is
    sum(feature == modality AND target == 1) / sum(feature == modality)
    The WOE is the sigmoid (log(x) / log(1-x)) of the impact.


    Methods
    -----------
    fit: calculates the groups from a training dataset
    trasform: creates the groups based on previous fitting
    fit_transform: fit and transform

    """


    def __init__(self):
        pass


    def fit(self, data, columns, target, event=1):
        """
        Parameters
        -----------
        data: training dataset
        columns: iterable of categorical columns to re-code
        target: name of the target variable
        event: event modality of the target (default 1)
        """

        self.transformations = {}
        self.columns = columns

        for col in self.columns:
            self.transformations[col] = {}
            for elem in data[col].unique():
                impact = (sum((data[target] == event) & (data[col] == elem)) /
                            sum(data[col] == elem))
                self.transformations[col][elem] = round(sigmoid(impact), 4)


    def transform(self, data, replace=False, prefix_old=None, prefix='woe'):
        """
        Parameters
        -----------
        data: dataset to transform; it has to contain all the columns of the
              transformation (see self.columns)
        replace: if True, the categorical columns are overwritten with the
                 transformed values (default False)
        prefix_old: prefix to put in front of the original variables (considered
                    only if replace == False)
        prefix: prefix to put in front of the transformed variables (considered
                only if replace == False and prefix_old == None)

        Output
        -----------
        data: transformed dataset
        """

        if hasattr(self, 'columns') == False:
            raise ValueError('Woe transformation has to be fitted yet.')
        for col in self.columns:
            if col not in data.columns:
                raise ValueError('Column %s not in input data features.' % col)

        for col in self.columns:
            if replace:
                old_c = col
                new_c = col
            elif prefix_old == None:
                old_c = col
                new_c = prefix + col
            else:
                new_c = col
                old_c = prefix_old + col
                data.rename(columns={col: old_c}, inplace=True)

            default = np.mean(list(self.transformations[col].values()))
            data[new_c] = data[old_c].apply(lambda x:
                self.transformations[col][x] if x in self.transformations[col].keys()
                else default)

        return data


    def fit_transform(self, data, columns, target, event=1, replace=False,
                      prefix_old=None, prefix='woe'):
        """
        Parameters
        -----------
        data: training dataset to transform
        columns: iterable of categorical columns to re-code
        target: name of the target variable
        event: event modality of the target (default 1)
        replace: if True, the categorical columns are overwritten with the
                 transformed values (default False)
        prefix_old: prefix to put in front of the original variables (considered
                    only if replace == False)
        prefix: prefix to put in front of the transformed variables (considered
                only if replace == False and prefix_old == None)

        Output
        -----------
        data: transformed dataset
        """

        self.fit(data, columns, target, event)
        return self.transform(data, replace=replace, prefix_old=prefix_old,
                              prefix=prefix)
