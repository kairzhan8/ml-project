from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from show_confusion_matrix import show_confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from pandas.tools.plotting import scatter_matrix
import random
import statsmodels.api as sm
import time
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pylab as pl
from sklearn.neural_network import MLPClassifier
from threading import Thread
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import itertools

filename = '/Users/Kairzhan/Desktop/ml_final_project/KidCreative.csv'
features=['Buy','Income','Is_Female','Is_Married','Has_College','Is_Professional','Is_Retired','Unemployed','Residence_Length','Dual_Income','Minors','Own','House','White','English','Prev_Child_Mag','Parent']

csv=pd.read_csv(filename,sep=',')
datasets=csv.as_matrix()

dataset=[]  
target=[]
data=[]
for i in range(0,len(datasets)):
    data.append([])
    dataset.append([])
    for j in range (len(datasets[i])):
        if j==0:
            continue
        else:
            dataset[i].append(datasets[i][j])
            if j==1:
                target.append(datasets[i][j])
            else:
                data[i].append(datasets[i][j])

dataset=np.asarray(dataset)

X=np.asarray(data)
Y=np.asarray(target)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
X_train,X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.25,random_state=3)

frame =pd.DataFrame(dataset)
frame.columns=features

cols_to_norm = ['Income','Residence_Length']
frame[cols_to_norm] = frame[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))

columns=[(frame.Buy),(frame.Income),(frame.Is_Female),(frame.Is_Married),(frame.Has_College),(frame.Is_Professional),(frame.Is_Retired),(frame.Unemployed),(frame.Residence_Length),(frame.Dual_Income),(frame.Minors),(frame.Own),(frame.House),(frame.White),(frame.English),(frame.Prev_Child_Mag),(frame.Parent)]
forplot=[]
column=[]
row=[]

k=0
for i in range(0,len(columns)):
    for j in range(i+1,len(columns)-1):
        if columns[i].corr(columns[j])>0.6 or columns[i].corr(columns[j])<-0.6:
            forplot.append(columns[i].corr(columns[j]))
            column.append(features[i])
            row.append(features[j])
            k+=1
            
def get_correlations():
    for i in range(0,len(columns)):
        for j in range(i+1,len(columns)-1):
            print ('corr btw',features[i],'and',features[j],columns[i].corr(columns[j]))


def draw_high_cor():
    fig = plt.figure(figsize=(45, 15))
    plots = len(forplot)
    ax=[]
    s=0
    f=0
    for i in range(0,plots):
            ax.append(plt.subplot2grid((5,4), (s,f)))
            f+=1
            ax[i].scatter(frame[row[i]],frame[column[i]],  s=10, c=[random.random(),random.random(),random.random()], marker="o")
            ax[i].set_ylabel(column[i])
            ax[i].set_xlabel(row[i])
            if (i+1)%4==0:
                s+=1
                f=0
    plt.show()
    plt.close(fig)


def correlation_fig():
    correlations = frame.corr()
    sm.graphics.plot_corr(correlations, xnames=features,ynames=features)
    plt.show()

def scatter_matrix_fig():
    scatter_matrix(frame,alpha=0.5, figsize=(20, 20), diagonal='kde')
    plt.show()


def hist_fig():
    frame.hist()
    plt.show()

#Bayes
nb=GaussianNB()
nb.fit(X_train,y_train)
nbpred=[]

#KNN
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
knnpred=[]

#DT
model = DecisionTreeClassifier(min_samples_split=5)
model.fit(X_train, y_train)
dtpred=[]

#LR
logit = LogisticRegression()
logit.fit(X_train,y_train)
logitpred=[]

#SVM
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)
svcpred=[]

#ANN
ann = MLPClassifier()
ann.fit(X_train,y_train)
annpred=[]
data_arr=list(X_val)

for i in range(0,len(data_arr)):
    knnpred.append(knn.predict([data_arr[i]]))
    dtpred.append(model.predict([data_arr[i]]))
    nbpred.append(nb.predict([data_arr[i]]))
    logitpred.append(logit.predict([data_arr[i]]))
    svcpred.append(svc.predict([data_arr[i]]))
    annpred.append(ann.predict([data_arr[i]]))

def general_accuracy():
    print ("accuracy KNN Algorithm:",accuracy_score(y_val, knnpred))
    print ("accuracy Data Tree:",accuracy_score(y_val, dtpred))
    print ("accuracy Gaussian Normal:",accuracy_score(y_val,nbpred))
    print ("accuracy Logistic Regression:",accuracy_score(y_val, logitpred))
    print ("accuracy SVM :",accuracy_score(y_val, svcpred))
    print ("accuracy ANN :",accuracy_score(y_val, annpred))


def get_conf(predicted):
    tn, fp, fn, tp = confusion_matrix(y_val, predicted).ravel()
    print ('True positives:',tp,'\nTrue negatives:',tn,'\nFalse negatives:',fn,'\nFalse positives',fp)
    print(classification_report(np.asarray(y_val), np.asarray(predicted)))
    print ('********************')

    
    
def model_implementation():
    k_range=range(1,41)
    k_scores=[] 
    p_name=['Value of K for KNN','Value of C in Logit','Value of Max iterations for Logit','Value of Max_depth for Decition Tree','Value of alpha for ANN','Value of C for SVM']
    Max_range=pl.frange(0,200,5)
    C_range=pl.frange(0.1,1,0.1)
    n_folds=10
    C_scores=[]
    Max_scores=[]
    scores_stds=[]
    scores_std=[]

    p_i=[]
    p_j=[]
    for k in k_range:
        knn2 = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn2, X_train, y_train, cv=10)
        k_scores.append(scores.mean())
        scores_std.append(scores.std()*2)
    scores_stds.append(scores_std)
    k_scores, scores_std = np.array(k_scores), np.array(scores_std)
    p_i.append(k_scores)
    p_j.append(k_range)
    scores_std=[]
    for c in C_range:
        log = LogisticRegression(C=c)
        scores = cross_val_score(log, X_train, y_train, cv=10)
        C_scores.append(scores.mean())
        scores_std.append(scores.std()*2)
    scores_stds.append(scores_std)
    C_scores, scores_std = np.array(C_scores), np.array(scores_std)    
    p_i.append(C_scores)
    p_j.append(C_range)
    
    scores_std=[]
    for M in Max_range:
        log = LogisticRegression(max_iter=M)
        scores = cross_val_score(log, X_train, y_train, cv=10)
        Max_scores.append(scores.mean())
        scores_std.append(scores.std()*2)
    scores_stds.append(scores_std)
    Max_scores, scores_std = np.array(Max_scores), np.array(scores_std)
    p_i.append(Max_scores)
    p_j.append(Max_range)
    
    #Tree
    tree_scores=[]
    tree_range=range(3,10)
    scores_std=[]
    for M in tree_range:
        dt = DecisionTreeClassifier(max_depth=M)
        scores = cross_val_score(dt, X_train, y_train, cv=10)
        tree_scores.append(scores.mean())
        scores_std.append(scores.std()*2)
    scores_stds.append(scores_std)
    tree_scores, scores_std = np.array(tree_scores), np.array(scores_std)
    p_i.append(tree_scores)
    p_j.append(tree_range)
    

    #ANN
    ann_scores=[]
    ann_range=pl.frange(0.0001,1,0.01)
    scores_std=[]
    for M in ann_range:
        Ann = MLPClassifier(alpha=M)
        scores = cross_val_score(Ann, X_train, y_train, cv=10)
        ann_scores.append(scores.mean())
        scores_std.append(scores.std()*2)
    scores_stds.append(scores_std)
    ann_scores, scores_std = np.array(ann_scores), np.array(scores_std)
    p_i.append(ann_scores)
    p_j.append(ann_range)
    

    #CVM
    cvm_scores=[]
    cvm_range=pl.frange(0.1,10,0.1)
    scores_std=[]
    for M in cvm_range:
        Cvm = SVC(C=M)
        scores = cross_val_score(Cvm, X_train, y_train, cv=10)
        cvm_scores.append(scores.mean())
        scores_std.append(scores.std()*2)
    scores_stds.append(scores_std)
    cvm_scores, scores_std = np.array(cvm_scores), np.array(scores_std)
    p_i.append(cvm_scores)
    p_j.append(cvm_range)



    plt.figure(figsize=(45, 20))
    ax=[]
    s=0
    f=0
    for i in range(0,len(p_i)):
        ax.append(plt.subplot2grid((5,4), (s,f)))
        f+=1
        ax[i].semilogx(p_j[i], p_i[i],color='red')
        std_error = scores_stds[i] / np.sqrt(n_folds)
        ax[i].semilogx(p_j[i], p_i[i] + std_error, 'b--')
        ax[i].semilogx(p_j[i], p_i[i] - std_error, 'b--')
        ax[i].set_ylabel("Cross-validated accuracy")
        ax[i].set_xlabel(p_name[i])
        ax[i].fill_between(p_j[i], p_i[i] + std_error, p_i[i] - std_error)

       
        ax[i].axhline(np.max(p_i[i]), linestyle='--', alpha=0.2)
        ax[i].set_xlim([p_j[i][0], p_j[i][-1]])
        if (i+1)%4==0:
            s+=1
            f=0
    plt.show()


def new_models():
    global logit2
    print ("**********************************************")
    print ("Neighbors = 27 is for best model KNeighborsClassifier")
    knn2= KNeighborsClassifier(n_neighbors=27)
    knn2.fit(X_train,y_train)
    knnpred2=[]
    print ("C=0.2 is best model for Logistic Regression for ")
    logit2 = LogisticRegression(C=0.2)
    logit2.fit(X_train,y_train)
    logitpred2=[]
    
    #DT
    print ("max_depth=4 is best model for DT ")
    
    d_tree1 = DecisionTreeClassifier(max_depth=4)
    d_tree1.fit(X_train,y_train)
    dtreepred=[]
    #SVM
    print ("Best Feature Selection - SVM 1.5")

    s_v_m1 = SVC(C=1.5)
    s_v_m1.fit(X_train,y_train)
    s_v_pred=[]
            
    #ANN
    print ("Best Feature Selection - ANN 0.071")
    
    a_n_n1 = MLPClassifier(alpha=0.071)
    a_n_n1.fit(X_train,y_train)
    a_n_npred=[]
    
    for i in range(0,len(X_val)):
        knnpred2.append(knn2.predict([X_val[i]]))
        logitpred2.append(logit2.predict([X_val[i]]))
        dtreepred.append(d_tree1.predict([X_val[i]]))
        s_v_pred.append(s_v_m1.predict([X_val[i]]))
        a_n_npred.append(a_n_n1.predict([X_val[i]]))
    print ("accuracy Of New KNN:",accuracy_score(y_val, knnpred2))
    print ("accuracy Of New LogisticRegression:",accuracy_score(y_val, logitpred2))
    print ("accuracy Of New Decision Tree:",accuracy_score(y_val, dtreepred))
    print ("accuracy Of New SVM:",accuracy_score(y_val, s_v_pred))
    print ("accuracy Of New ANN:",accuracy_score(y_val, a_n_npred))
    
    
    print ("\n********************LOGISTIC*********************")
    
    print ("New Model VS OLD Model For Logit")
    
    print('Logit Variance OLD: %.2f' % logit.score(X_val, y_val))
    print('Logit Variance NEW: %.2f' % logit2.score(X_val, y_val))
     
    y_pred=logit.predict(X_val)
    get_mse_rmse_model(y_pred,'OLD','LOGIT')
    y_pred=logit2.predict(X_val)
    get_mse_rmse_model(y_pred,'NEW','LOGIT')
    
    print ("\n***************************KNN***********************")
    
    print ("New Model VS OLD Model For Knn")
    
    print('KNN Variance OLD: %.2f' % knn.score(X_val, y_val))
    print('KNN Variance NEW: %.2f' % knn2.score(X_val, y_val))
     
    y_pred=knn.predict(X_val)
    get_mse_rmse_model(y_pred,'OLD','KNN')
    
    y_pred=knn2.predict(X_val)
    get_mse_rmse_model(y_pred,'NEW','KNN')
    print ("*******************************************************")

    print ("New Model VS OLD Model For DT")
    
    print('DT Variance OLD: %.2f' % model.score(X_val, y_val))
    print('DT Variance NEW: %.2f' % d_tree1.score(X_val, y_val))
     
    y_pred=model.predict(X_val)
    get_mse_rmse_model(y_pred,'OLD','DT')
    
    y_pred=d_tree1.predict(X_val)# 
    get_mse_rmse_model(y_pred,'NEW','DT')
    print ("*******************************************************")

    print ("New Model VS OLD Model For SVM")
    
    print('SVM Variance OLD: %.2f' % svc.score(X_val, y_val))
    print('SVM Variance NEW: %.2f' % s_v_m1.score(X_val, y_val))
     
    y_pred=model.predict(X_val)
    get_mse_rmse_model(y_pred,'OLD','SVM')
    
    y_pred=d_tree1.predict(X_val)
    get_mse_rmse_model(y_pred,'NEW','SVM')
    
    print ("*******************************************************")

    print ("New Model VS OLD Model For ANN")
    
    print('ANN Variance OLD: %.2f' % ann.score(X_val, y_val))
    print('ANN Variance NEW: %.2f' % a_n_n1.score(X_val, y_val))
     
    y_pred=model.predict(X_val)
    get_mse_rmse_model(y_pred,'OLD','ANN')
    
    y_pred=d_tree1.predict(X_val)
    get_mse_rmse_model(y_pred,'NEW','ANN')

    #TEST
    print ("********************TEST best parameters**************************")
    knnpred_test=[]
    logitpred_test=[]
    svm_test=[]
    ann_test=[]
    dt_test=[]
    for i in range(0,len(X_test)):
        knnpred_test.append(knn2.predict([X_test[i]]))
        logitpred_test.append(logit2.predict([X_test[i]]))
        svm_test.append(s_v_m1.predict([X_test[i]]))
        ann_test.append(a_n_n1.predict([X_test[i]]))
        dt_test.append(d_tree1.predict([X_test[i]]))
    print ("accuracy knn TEST:",accuracy_score(y_test, knnpred_test))
    print ("accuracy logistic TEST:",accuracy_score(y_test, logitpred_test))
    print ("accuracy SVM TEST:",accuracy_score(y_test, svm_test))
    print ("accuracy DT TEST:",accuracy_score(y_test, dt_test))
    print ("accuracy ANN TEST:",accuracy_score(y_test, ann_test))
    

#Checking Accuracy and ERRORs

def Tree_class():

    model_Tree = ExtraTreesClassifier()
    model_Tree.fit(X_train,y_train)
    print (model_Tree.feature_importances_)
    
def get_mse_rmse(y_val_new,y_pred):
    print("MSE3: %.2f" % (metrics.mean_squared_error(y_val_new,y_pred)))
    print("MAE3: %.2f" % (metrics.mean_absolute_error(y_val_new,y_pred)))
    print("RMSE3: %.2f" % (np.sqrt(metrics.mean_squared_error(y_val_new,y_pred))))
    
def accuracy_metrics_for_selected_features():
    global logit2

    xx=frame[['Income','Residence_Length']]
    yy=frame['Buy']
    xx= list(np.array(xx))
    yy=list(np.array(yy))
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(xx, yy, test_size=0.2, random_state=3)#20% Test, 80%Train
    X_train_new,X_val_new, y_train_new, y_val_new = train_test_split(X_train,y_train, test_size=0.25,random_state=3)#20% Validation 60%Train
    
    print ("Best Feature Selection - Logistic Regression")
    logit3 = LogisticRegression(C=0.2)
    logit3.fit(X_train_new,y_train_new)
    logitpred3=[]
    for i in range(0,len(X_val_new)):
        logitpred3.append(logit3.predict([X_val_new[i]]))
    print ("accuracy Of New LogisticRegression:",accuracy_score(y_val_new, logitpred3))
    y_pred=logit3.predict(X_val_new)
    get_mse_rmse(y_val_new,y_pred)
    

    print ("\nBest Feature Selection - Decision Tree")
    d_tree = DecisionTreeClassifier(max_depth=4)
    d_tree.fit(X_train_new,y_train_new)
    dtreepred=[]
    for i in range(0,len(X_val_new)):
        dtreepred.append(d_tree.predict([X_val_new[i]]))
    
    print ("accuracy Of New Decision Tree:",accuracy_score(y_val_new, dtreepred))
    y_pred=d_tree.predict(X_val_new)
    get_mse_rmse(y_val_new,y_pred)

    print ("\nBest Feature Selection - KNN ")
    k_nn = KNeighborsClassifier(n_neighbors=27)
    k_nn.fit(X_train_new,y_train_new)
    k_nnpred=[]
    for i in range(0,len(X_val_new)):
        k_nnpred.append(k_nn.predict([X_val_new[i]]))
    
    print ("accuracy Of New KNN:",accuracy_score(y_val_new, k_nnpred))
    y_pred=k_nn.predict(X_val_new)
    get_mse_rmse(y_val_new,y_pred)


    print ("Best Feature Selection - SVM")

    s_v_m = SVC(C=1.5)
    s_v_m.fit(X_train_new,y_train_new)
    s_v_pred=[]
    for i in range(0,len(X_val_new)):
        s_v_pred.append(s_v_m.predict([X_val_new[i]]))
    

    print ("accuracy Of New SVM:",accuracy_score(y_val_new, s_v_pred))
    y_pred=s_v_m.predict(X_val_new)
    get_mse_rmse(y_val_new,y_pred)
    
    print ("Best Feature Selection - ANN")
    
    a_n_n = MLPClassifier(alpha=0.071)
    a_n_n.fit(X_train_new,y_train_new)
    a_n_npred=[]
    for i in range(0,len(X_val_new)):
        a_n_npred.append(a_n_n.predict([X_val_new[i]]))
    
    print ("accuracy Of New ANN:",accuracy_score(y_val_new, a_n_npred))
    y_pred=a_n_n.predict(X_val_new)
    get_mse_rmse(y_val_new,y_pred)


def feature_importance_random_forest():
    names = features[1:]
    rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    scores = []
    for i in range(X.shape[1]):
         score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                                  cv=ShuffleSplit(len(X), 3, .3))
         scores.append((round(np.mean(score), 3), names[i]))
    print (sorted(scores, reverse=True))


inp=''
while inp!='x':
    print ("1 - Correlations")
    print ("2 - Visualize correlation figure ")
    print ('3 - Visualize scatter_matrix figure')
    print ('4 - Visualize only highly correlated features')
    print ('5 - Visualize histogram figure')
    print ('6 - Print General accuracy for all appropriate algorithms')
    print ("7 - Show newly generated Model's performation and accuracy")
    
    print ("8 - Get feature Importance using ExtraTreeClassifier")
    print ("9 - New_Model from Selecting important features, and their accuracy,errors,etc")
    print ("10 - Get feature Importance using RandomForestClassifier")
    
    
    print ('x - To exit')
    
    inp=input('Enter The command: ')
    if inp=='1':
        Thread(target=get_correlations).start()   
    elif inp=='2':
        correlation_fig()
    elif inp=='3': 
        scatter_matrix_fig()
    elif inp=='4':
        draw_high_cor()
    elif inp=='5':
        hist_fig()
    elif inp=='6':
        Thread(target=general_accuracy).start()
    elif inp=='7':
        Thread(target=new_models).start()
    elif inp=='8':
        Tree_class()
    elif inp=='9':
        accuracy_metrics_for_selected_features()
    elif inp=='10':
        print ('You have to wait until it performes... about 3-5minutes...')
        feature_importance_random_forest()

    elif inp=='x':
        print ('Exiting...')
    else:
        print ('No such command')
    time.sleep(2)
