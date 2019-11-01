"""
@author: Matteo Felici - matteofelici87@gmail.com

Package of the Machine Learning utilities for data description.
- freq_tab: creates table of (relative) frequency
- description: create table with columns names, modalities, types,
               number of missing values and mode/mean
"""
import pandas as pd


def freq_tab(data, row_var, col_var, margins=True, perc=None):
    """
    Function that returns a frequency table between two columns of a
    pandas DataFrame. It can also calculate the row/column totals and
    the relative frequency.

    Parameters
    ----------
    data: pandas DataFrame
    row_var: feature of data to put in rows
    col_var: feature of data to put in columns
    margins: if True, it adds the row/column totals (default True)
    perc: if set to 'col' or 'row', it put the relative frequency by column
          or row (default None).

    Output
    -----------
    cr: pandas DataFrame result of a pandas.crosstab
    """

    cr = pd.crosstab(index=data[row_var],
                     columns=data[col_var],
                     margins=margins)
    if margins and perc in ('col', 'row'):
        cr.index = ['col_tot' if x == 'All' else x for x in cr.index.tolist()]
        cr.columns = ['row_tot' if x == 'All' else x for x in cr.columns.tolist()]

        if perc == 'col':
            cr = cr / cr.ix['col_tot']
        elif perc == 'row':
            cr = cr.div(cr['row_tot'], 0)

    return cr


def description(data, columns=None, complete=False, report=True, pr=True):
    """
    Function that calculates the basic informations on the selected features
    of a pandas DataFrame. It can print a report and/or create a DataFrame
    with all the infos.

    Parameters
    -----------
    data: pandas DataFrame
    columns: features to analyse; if None, takes all (default None)
    complete: if True, print all the modalities; if False, only the top 2
              (default False)
    report: if True, it returns a DataFrame with columns names, numbers of
            modalities, types, numbers of missings and modes/means
            (default True)
    pr: if True, it prints all the infos (default True)

    Output
    ------
    rep: pandas DataFrame with feature infos (only if report == True)
    """

    rep = pd.DataFrame(columns=['column', 'modalities', 'type', 'missing', 'mode / mean'])

    if columns is None:
        columns = data.columns
    elif type(columns) != list:
        columns = [columns]

    for col in columns:
        this_pr = pr
        distinct = len(data[col].unique())
        if distinct > min(len(data), 300) and data[col].dtype == 'O':
            tipo = 'ID'
            this_pr = False
        elif distinct == 1:
            tipo = 'unaria'
        elif distinct == 2:
            tipo = 'binaria'
        else:
            tipo = str(data[col].dtype).replace('object', 'categorica')
        if this_pr:
            print('')
            print(col, ' ---> ', tipo)
        miss = sum(data[col].isnull())
        if miss > 0 and this_pr:
            print('%d missing values\n' % miss)
        if tipo in ('categorica', 'binaria') or distinct <= 10:
            if tipo == 'categorica' and this_pr:
                print("%d modalita'" % distinct)
            if this_pr:
                print('Modes:')
            if complete or tipo not in ('categorica', 'binaria'):
                a = data[col].value_counts() / len(data)
            else:
                a = data[col].value_counts()[:2] / len(data)
            val = str(a.index[0]) + ' --- ' +  str(round(a.values[0], 3) * 100) + '%'
            m = max([len(str(i)) for i in a.index])
            if this_pr:
                for i in a.index:
                    print(i, ' ' * (m - len(str(i))), str(round(a[i] * 100, 1)) + '%')
        else:
            a = data[col].describe()[1:]
            try:
                val = a['mean']
            except:
                val = a['top']
            if this_pr:
                for i in a.index:
                    try:
                        print(i.replace('mean', 'avg'), '', round(a[i], 2))
                    except:
                        print(i.replace('mean', 'avg'), '', a[i])
                print()
        rep.loc[len(rep) + 1] = [col, int(distinct), tipo, int(miss), val]

    return rep