import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars
    return


@app.cell
def _(X_train_cleaned, y_train):
    from sklearn.model_selection import train_test_split
    train_x, valid_x, train_y, valid_y = train_test_split(
        X_train_cleaned, y_train, test_size=0.25, random_state=42)
    return train_x, train_y, valid_x, valid_y


@app.cell
def _(accuracy_score, train_x, train_y, valid_x, valid_y):
    from pytabkit import CatBoost_TD_Classifier, CatBoost_D_Classifier, LGBM_TD_Classifier, LGBM_D_Classifier, XGB_TD_Classifier, XGB_D_Classifier
    # from pytabkit import LGBM_TD_Classifier,LGBM_D_Classifier

    for model in [CatBoost_TD_Classifier(), CatBoost_D_Classifier(), LGBM_TD_Classifier(), LGBM_D_Classifier(), XGB_TD_Classifier(), XGB_D_Classifier()]:
    # for model in [LGBM_TD_Classifier(),LGBM_D_Classifier()]:
        model.fit(train_x, train_y)
        y_pred = model.predict(valid_x)
        acc = accuracy_score(valid_y, y_pred)
        print(f"Accuracy of {model.__class__.__name__}: {acc}")
    return


@app.cell
def _(train_x, train_y, valid_x):
    import time
    from pytabkit import RealMLP_TD_Classifier
    model = RealMLP_TD_Classifier(val_metric_name='cross_entropy')  # or TabR_S_D_Classifier, CatBoost_TD_Classifier, etc.
    start = time.perf_counter()
    model.fit(train_x, train_y)
    fit_time = time.perf_counter()
    prediction = model.predict(valid_x)
    end = time.perf_counter()
    print('{:.2f}'.format((fit_time-start)/60))
    print('{:.2f}'.format((end-fit_time)/60))
    return


if __name__ == "__main__":
    app.run()
