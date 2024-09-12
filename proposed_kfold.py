import datetime
import pickle


def reduction(data):
    from sklearn.decomposition import PCA
    # The best component number was calculated by Variance Threshold method
    best_compnent_number=51  
    pca=PCA(n_components=best_compnent_number)
    data=pca.fit_transform(data)
    return data

def train_model():
    from sklearn.metrics import (accuracy_score,confusion_matrix,
                                 f1_score,precision_score,recall_score)

    # read data
    import read_data
    dpi=0
    data,labels=read_data.read_all_data(dpi)

    # dimension reduction
    data=reduction(data)


    from keras.utils import to_categorical
    labels=to_categorical(labels)

    # split dataset
    from sklearn.model_selection import train_test_split,KFold
    x_TrainValid,x_test,y_TrainValid,y_test=train_test_split(data,labels,random_state=42,test_size=0.1)


    # normalization
    from sklearn.preprocessing import RobustScaler
    scaler=RobustScaler()
    # x_TrainValid=scaler.fit_transform(x_TrainValid)
    x_test=scaler.fit_transform(x_test)


    # prepare CNN model
    cnn_input_shape=(x_TrainValid.shape[1],1)
    model_cnn=build_CNN_model(cnn_input_shape)
    from keras.callbacks import CSVLogger

    lst_acc=[]
    lst_f1=[]
    lst_precision=[]
    lst_recall=[]
    lst_matrix=[]
    lst_times=[]
    fold_number=1
    model_name='ProposedMethod'

    
    kfold=KFold(n_splits=10,shuffle=True)
    for train,valid in kfold.split(x_TrainValid,y_TrainValid):
        
        callback=CSVLogger(f'./Proposed_logger_{fold_number}.log')

        x_train=x_TrainValid[train]
        x_valid=x_TrainValid[valid]
        y_train=y_TrainValid[train]
        y_valid=y_TrainValid[valid]

        x_train=scaler.fit_transform(x_train)
        x_valid=scaler.fit_transform(x_valid)


        # show shape of data
        print(f'\n\nTrain shape: Data:{x_train.shape}   Labels:{y_train.shape}')
        print(f'Final Validation shape: Data:{x_valid.shape}   Labels:{y_valid.shape}')

        # train CNN
        start_time=datetime.datetime.now()
        model_cnn.fit(x_train,y_train,epochs=20,batch_size=1024,validation_data=(x_valid,y_valid),callbacks=[callback])
        model_cnn.save(f'./ProposedCNN_fold{fold_number}.h5')
        feature_level2=model_cnn.predict(x_train)

        # train ML
        y_train=y_train.argmax(axis=1)
        from xgboost import XGBClassifier as classifier
        model_ml=classifier(colsample_bytree= 0.9,learning_rate= 0.2,max_depth=7,n_estimators=100,subsample=0.7)
        model_ml.fit(feature_level2,y_train)
        end_time=datetime.datetime.now()
        pickle.dump(model_ml,open(f'./ProposedXGBoost_fold{fold_number}.sav','wb'))

        
        # test and evaluation
        print(y_test.shape)
        y_test=y_test.argmax(axis=1)
        training_time=end_time-start_time
        predicts_level2=model_cnn.predict(x_test)
        predicts=model_ml.predict(predicts_level2)


        # Evaluation
        lst_acc.append(accuracy_score(y_test,predicts))
        lst_f1.append(f1_score(y_test,predicts,average='weighted'))
        lst_precision.append(precision_score(y_test,predicts,average='weighted'))
        lst_recall.append(recall_score(y_test,predicts,average='weighted'))
        lst_matrix.append(confusion_matrix(y_test,predicts))
        lst_times.append(training_time)

        print('Test ACC: ',accuracy_score(y_test,predicts))
        y_test=to_categorical(y_test)
        fold_number+=1

    import print_results
    print_results.results(lst_acc,lst_f1,lst_precision,lst_recall,lst_matrix,lst_times,model_name)


def build_CNN_model(cnn_input_shape):
    from keras.models import Sequential
    from keras.layers import Conv1D,Dense,Flatten,Dropout
    from keras.losses import categorical_crossentropy

    model=Sequential()
    model.add(Conv1D(filters=64,kernel_size=3,padding='same',strides=2,activation='relu',input_shape=cnn_input_shape))
    model.add(Conv1D(filters=256,kernel_size=3,padding='same',strides=2,activation='relu'))
    model.add(Conv1D(filters=256,kernel_size=3,padding='same',strides=2,activation='relu'))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer='adam',loss=categorical_crossentropy,metrics=['accuracy'])
    return model

train_model()





