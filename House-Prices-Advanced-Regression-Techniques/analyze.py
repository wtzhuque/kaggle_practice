#!/usr/bin/env python3


import pandas
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression


def main():
    """
    Analyze Data
    """
    train_file = "./data/train.csv"
    train_data = pandas.read_csv(train_file)
    
    feature_columns = ['MSSubClass',
                       'LotArea',
                       'OverallQual',
                       'OverallCond',
                       'MiscVal',
                       '1stFlrSF',
                       '2ndFlrSF',
                       'GarageArea']

    features = train_data[feature_columns]
    fea_num = len(feature_columns)
    label = train_data.SalePrice

    feature_selectorA = SelectKBest(f_regression, k=5).fit(features, label)
    feature_selectorB = SelectKBest(chi2, k=5).fit(features, label)
    feature_statusA = feature_selectorA.get_support()
    feature_statusB = feature_selectorB.get_support()
    for idx in range(0, len(feature_columns)):
        if feature_statusA[idx] and feature_statusB[idx]:
            print("%s is selected" % feature_columns[idx])
        else:
            print("%s is droped" % feature_columns[idx])


if __name__ == '__main__':
    main()
