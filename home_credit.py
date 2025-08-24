import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import numpy as np
    return np, pl


@app.cell
def _(pl):
    df = pl.read_csv('data/application_train.csv')
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(np):
    import pandas as pd
    def one_hot_encoder(df, nan_as_category = True):
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        return df, new_columns
    
    def application_train_test(num_rows = None, nan_as_category = False):
        # Read data and merge
        df = pd.read_csv('data/application_train.csv', nrows= num_rows)
        # test_df = pd.read_csv('../input/application_test.csv', nrows= num_rows)
        # print("Train samples: {}, test samples: {}".format(len(df)))
        # df = df.append(test_df).reset_index()
        # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
        df = df[df['CODE_GENDER'] != 'XNA']
    
        # Categorical features with Binary encode (0 or 1; two categories)
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[bin_feature], uniques = pd.factorize(df[bin_feature])
        # Categorical features with One-Hot encode
        df, cat_cols = one_hot_encoder(df, nan_as_category)
    
        # NaN values for DAYS_EMPLOYED: 365.243 -> nan
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
        # Some simple new features (percentages)
        df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        # del test_df
        # gc.collect()
        return df
    return (application_train_test,)


@app.cell
def _(application_train_test):
    df_train = application_train_test()
    df_train
    return (df_train,)


@app.cell
def _(df_train):
    df_train.head(1000).to_csv('data/home_credit_sample.csv')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
