import numpy as np
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import read_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# read data
dpi=0
data,labels=read_data.read_all_data(dpi)


# split dataset
x_TrainValid,x_test,y_TrainValid,y_test=train_test_split(data,labels,random_state=42,test_size=0.1)
x_train,x_valid,y_train,y_valid=train_test_split(x_TrainValid,y_TrainValid,random_state=42,test_size=0.2)


# preprocessing
scaler=RobustScaler()
x_valid=scaler.fit_transform(x_valid)


# add a dimention
x_valid=np.expand_dims(x_valid,axis=1)


# Define the CNN model function
def create_cnn(num_conv_layers=1, num_dense_layers=1, num_filters=32, learning_rate=0.01, optimizer='adam'):
    model = models.Sequential()
    
    # The first layer
    model.add(layers.Conv1D(filters=64,kernel_size=3,padding='same',strides=2,activation='relu',input_shape=(x_valid.shape[1],x_valid.shape[2])))


    # Add convolutional layers
    for _ in range(num_conv_layers):
        model.add(layers.Conv1D(num_filters, 3, activation='relu',padding='same',strides=2))

    
    model.add(layers.Flatten())
    
    # Add dense layers
    for _ in range(num_dense_layers):
        model.add(layers.Dense(64, activation='relu'))
    
    model.add(layers.Dense(3, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create KerasClassifier
model = KerasClassifier(build_fn=create_cnn, verbose=0)


# Define the parameter grid to search
param_grid = {
    'num_conv_layers': [1,2,3,4,5,6,7],
    'num_dense_layers': [1,2,3,4,5,6,7],
    'num_filters': [64,128,256],
    # 'learning_rate': [0.001, 0.01, 0.1],
    'optimizer': ['adam','adagrad','adadelta','rmsprop','nadam','adamax'],
    'batch_size': [1024]
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