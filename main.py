import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def CreateFeaturesTargets(filename):

    df = pd.read_csv(filename ,header=None)

    # convert from 'DataFrame' to numpy array
    data = df.values

    # Features are in columns one to end
    X = data[: ,1:]

    # Scale features
    X = StandardScaler().fit_transform(X)

    # Labels are in the column zero
    y = data[: ,0]

    # return Features and Labels
    return X, y


def RMSE(model,X,y):
    model.fit(X,y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f'RMSE: {round(rmse,2)} weeks')
    return rmse


def RMSE_CV(model, X, y):
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    rmse_cv = np.sqrt(-np.mean(scores))
    print(f'RMSE_CV: {round(rmse_cv,2)} weeks')
    return rmse_cv


def PrintDataInfo(X, y):
    print(f'Number of samples is {X.shape[0]}')
    print(f'Number of features is {X.shape[1]}')
    print(f'Number of targets is {y.shape[0]}')

def PlotGraph(X_train, y_train, X_test=None, y_test=None, X_curve=None, y_curve=None):
    if X_test is not None and y_test is not None:
        plt.plot(X_test, y_test, 'o', label='Test set')


    if X_curve is not None and y_curve is not None:
        print("plotted")
        plt.plot(X_train, y_train, 'bo', label='Training set')
        plt.plot(X_curve, y_curve, 'ro', label='Predicted set')
    else:
        plt.plot(X_train, y_train, 'o', label='Training set')

    plt.legend()
    plt.title('Train test split')
    plt.xlabel('Feature')
    plt.ylabel('Target value')

    plt.show()


if __name__ == "__main__":
    X1, y1 = CreateFeaturesTargets('datasets/1-feature.csv')
    X6, y6 = CreateFeaturesTargets('datasets/6-features.csv')
    X86, y86 = CreateFeaturesTargets('datasets/86-features.csv')

    model = LinearRegression()

    # Single feature
    print('Single feature:')
    rmse1 = RMSE(model, X1, y1)
    rmse1_cv = RMSE_CV(model, X1, y1)

    # 6 features
    print('Six features:')
    rmse6 = RMSE(model, X6, y6)
    rmse6_cv = RMSE_CV(model, X6, y6)

    # 86 features
    print('86 features:')
    rmse86 = RMSE(model, X86, y86)
    rmse86_CV = RMSE_CV(model, X86, y86)

    PlotGraph(X1, y1)
    PlotGraph(X6, y6)
    PlotGraph(X86, y86)