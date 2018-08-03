#!/bin/usr/env python


import pandas
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


def main():
    """
    Main Entry
    """
    train_file = "./data/train.csv"
    test_file = "./data/test.csv"
    train_data = pandas.read_csv(train_file)
    test_data = pandas.read_csv(test_file)

    feature_columns = ['LotArea','OverallQual','OverallCond', 'MiscVal', '1stFlrSF', '2ndFlrSF', 'GarageArea']

    features = train_data[feature_columns]
    test_features = test_data[feature_columns]

    label = train_data.SalePrice
    #test_label = test_data.SalePrice

    model = DecisionTreeRegressor()
    model.fit(features, label)

    pred_label = model.predict(test_features)
    #error = mean_absolute_error(test_label, pred_label)
    #print(error)


if __name__ == "__main__":
    main()
