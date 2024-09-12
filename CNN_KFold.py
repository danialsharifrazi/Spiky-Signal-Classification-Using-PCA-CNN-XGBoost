import datetime

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
    x_test=scaler.fit_transform(x_test)


    # prepare CNN model
    cnn_input_shape=(x_TrainValid.shape[1],1)
    model=build_CNN_model(cnn_input_shape)
    from keras.callbacks import CSVLogger

    lst_acc=[]
    lst_f1=[]
    lst_precision=[]
    lst_recall=[]
    lst_matrix=[]
    lst_times=[]
    fold_number=1
    model_name='CNN'

    from sklearn.model_selection import KFold,train_test_split
    kfold=KFold(n_splits=10,shuffle=True)
    for train,valid in kfold.split(x_TrainValid,y_TrainValid):
        
        callback=CSVLogger(f'./CNN/CNN_logger_{fold_number}.log')

        x_train=x_TrainValid[train]
        x_valid=x_TrainValid[valid]
        y_train=y_TrainValid[train]
        y_valid=y_TrainValid[valid]

        x_train=scaler.fit_transform(x_train)
        x_valid=scaler.fit_transform(x_valid)


        # train
        start_time=datetime.datetime.now()
        model.fit(x_train,y_train,epochs=20,batch_size=1024,validation_data=(x_valid,y_valid),callbacks=[callback])
        end_time=datetime.datetime.now()
        model.save(f'./CNN/{model_name}_fold{fold_number}.h5')

        # test and evaluation
        from sklearn.metrics import (accuracy_score,confusion_matrix,auc
            ,f1_score,precision_score,recall_score)
        
        training_time=end_time-start_time
        
        predicts=model.predict(x_test)

        actuals=y_test
        actuals=actuals.argmax(axis=1)
        predicts=predicts.argmax(axis=1)

        lst_acc.append(accuracy_score(actuals,predicts))
        lst_f1.append(f1_score(actuals,predicts,average='weighted'))
        lst_precision.append(precision_score(actuals,predicts,average='weighted'))
        lst_recall.append(recall_score(actuals,predicts,average='weighted'))
        lst_matrix.append(confusion_matrix(actuals,predicts))
        lst_times.append(training_time)

        fold_number+=1

    results(lst_acc,lst_f1,lst_precision,lst_recall,lst_matrix,lst_times,model_name)




def build_CNN_model(data_shape):
    from keras.models import Sequential
    from keras.layers import Conv1D,Dense,Flatten
    from keras.losses import categorical_crossentropy

    model=Sequential()
    model.add(Conv1D(filters=64,kernel_size=3,padding='same',strides=2,activation='relu',input_shape=data_shape))
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


def results(lst_acc,lst_f1,lst_precision,lst_recall,lst_matrix,lst_times,model_name):    
    import numpy as np
    results_path=f'./CNN/results_{model_name}.txt'
    f1=open(results_path,'a')

    f1.write('\nAverage Accuracy: '+str(np.mean(lst_acc)))
    f1.write('\nAverage Precision: '+str(np.mean(lst_precision)))
    f1.write('\nAverage Recall: '+str(np.mean(lst_recall)))
    f1.write('\nAverage F1 Score: '+str(np.mean(lst_f1)))
    f1.write('\nAverage Training Time: '+str(np.mean(lst_times)))

    f1.write('\n\n\nMetrics for all Folds: \n')

    for i in range(len(lst_acc)):
        f1.write('\n Accuracy: '+str(lst_acc[i]))
        f1.write('\n Precision: '+str(lst_precision[i]))
        f1.write('\n Recall: '+str(lst_recall[i]))
        f1.write('\n F1 Score: '+str(lst_f1[i]))
        f1.write('\nTraining Time: '+str(lst_times[i]))
        f1.write('\n\nCofusion Matrix: \n'+str(lst_matrix[i]))
        f1.write('\n\n____________________\n')
    f1.close()
    

train_model()





