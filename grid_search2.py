# read data
import read_data
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


# split dataset
dpi=0
data,labels=read_data.read_all_data(dpi)

x_TrainValid,x_test,y_TrainValid,y_test=train_test_split(data,labels,random_state=42,test_size=0.1)
x_train,x_valid,y_train,y_valid=train_test_split(x_TrainValid,y_TrainValid,random_state=42,test_size=0.2)


# preprocessing
scaler=RobustScaler()
x_valid=scaler.fit_transform(x_valid)

# create model
model=XGBClassifier()


# Define the parameter grid to search
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100],
    'subsample': [0.5, 0.7, 0.9],
    'colsample_bytree': [0.5, 0.7, 0.9]
}

# Perform grid search
grid_search = GridSearchCV(estimator=model,param_grid=param_grid,cv=3,verbose=2)
grid_result = grid_search.fit(x_valid,y_valid)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Mean: %f, Std: %f with: %r" % (mean, stdev, param))

# Print the best parameters found
print("Best parameters:", grid_search.best_params_)