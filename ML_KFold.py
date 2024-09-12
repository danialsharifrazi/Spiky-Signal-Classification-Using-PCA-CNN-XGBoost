import datetime
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import RobustScaler


def train_model(model,model_name):

    from sklearn.metrics import (accuracy_score,confusion_matrix,
                                 f1_score,precision_score,recall_score)

    # read data
    import read_data
    dpi=7
    data,labels=read_data.read_all_data(dpi)

    # split dataset
    x_TrainValid,x_test,y_TrainValid,y_test=train_test_split(data,labels,random_state=42,test_size=0.1)

    # normalization
    scaler=RobustScaler()
    x_test=scaler.fit_transform(x_test)

    # read and split data
    lst_acc=[]
    lst_f1=[]
    lst_precision=[]
    lst_recall=[]
    lst_matrix=[]
    lst_times=[]
    fold_number=1

    kfold=KFold(n_splits=10,shuffle=True)
    for train,valid in kfold.split(x_TrainValid,y_TrainValid):
        
        x_train=x_TrainValid[train]
        x_valid=x_TrainValid[valid]
        y_train=y_TrainValid[train]
        y_valid=y_TrainValid[valid]

        x_train=scaler.fit_transform(x_train)
        x_valid=scaler.fit_transform(x_valid)

        # train
        start_time=datetime.datetime.now()
        model.fit(x_train,y_train)
        end_time=datetime.datetime.now()

        import pickle
        pickle.dump(model,open(f'./results/After Reviewing/ML/{model_name}_fold{fold_number}.sav','wb'))


        # test and evaluation
        from sklearn.metrics import (accuracy_score,confusion_matrix,auc
            ,f1_score,precision_score,recall_score)
        
        training_time=end_time-start_time
        
        predicts=model.predict(x_test)

        lst_acc.append(accuracy_score(y_test,predicts))
        lst_f1.append(f1_score(y_test,predicts,average='weighted'))
        lst_precision.append(precision_score(y_test,predicts,average='weighted'))
        lst_recall.append(recall_score(y_test,predicts,average='weighted'))
        lst_matrix.append(confusion_matrix(y_test,predicts))
        lst_times.append(training_time)

        fold_number+=1
   
    results(lst_acc,lst_f1,lst_precision,lst_recall,lst_matrix,lst_times,model_name)



def build_ML_models():

    # import ML models
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import  RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression

    # build ML models
    model1=MLPClassifier(hidden_layer_sizes=10,max_iter=20)
    model2=XGBClassifier()
    model3=AdaBoostClassifier()
    model4=RandomForestClassifier(n_estimators=10)
    model5=DecisionTreeClassifier()
    model6=GaussianNB()
    model7=LogisticRegression()
    
    # acllocate names
    name1='MLPClassifier'
    name2='XGBoost'
    name3='AdaBoostClassifier'
    name4='RandomForestClassifier'
    name5='DecisionTreeClassifier'
    name6='NaiveBayesian'
    name7='LogisticRegression'

    lst_models=[model1,model2,model3,model4,model5,model6,model7]
    lst_names=[name1,name2,name3,name4,name5,name6,name7]
    return lst_models,lst_names



def results(lst_acc,lst_f1,lst_precision,lst_recall,lst_matrix,lst_times,model_name):    
    import numpy as np
    results_path=f'./results_{model_name}.txt'
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
        f1.write('\n\n______________________________\n')
    f1.close()
    

# train all models in a FOR LOOP
lst_models,lst_names=build_ML_models()
for i in range(len(lst_models)):
    train_model(lst_models[i],lst_names[i])





