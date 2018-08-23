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
    label = train_data.SalePrice

    features_new = SelectKBest(f_regression, k=5).fit_transform(features, label)


if __name__ == '__main__':
    main()
