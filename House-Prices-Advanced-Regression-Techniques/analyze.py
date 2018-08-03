#!/usr/bin/env python3


import pandas



def main():
    """
    Analyze Data
    """
    train_file = "./data/train.csv"
    train_data = pandas.read_csv(train_file)
    print(train_data.ExterQual.describe())
    print(train_data.ExterCond.describe())



if __name__ == '__main__':
    main()
