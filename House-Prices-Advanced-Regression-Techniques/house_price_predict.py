#!/bin/usr/env python


import pandas
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def get_mae(max_leaf_nodes, train_features, train_labels, test_features, test_labels):
    #model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    #model = RandomForestRegressor()
    model = LinearRegression()
    model.fit(train_features, train_labels)
    pred_labels = model.predict(test_features)
    return(mean_absolute_error(test_labels, pred_labels))


def main():
    """
    Train and predict
    """
    train_file = "./data/train.csv"
    test_file = "./data/test.csv"
    train_data = pandas.read_csv(train_file)

    # Features
    feature_columns = ['LotArea','OverallQual','OverallCond', 'MiscVal', '1stFlrSF', '2ndFlrSF', 'GarageArea']

    features = train_data[feature_columns]
    label = train_data.SalePrice

    # Split samples to train set and test set
    train_feature_set, test_feature_set, train_label_set, test_label_set = train_test_split(features, label, random_state = 0)

    for max_leaf_nodes in [100, 300, 500, 1000, 3000]:
        mae = get_mae(max_leaf_nodes, train_feature_set, train_label_set, test_feature_set, test_label_set)
        print('Max Leaf Nodes: %d \t\t MAE: %f' % (max_leaf_nodes, mae))


if __name__ == "__main__":
    main()
