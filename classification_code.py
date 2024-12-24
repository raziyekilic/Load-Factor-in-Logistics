

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import os, os.path
from sklearn import svm

def calculate_metrics(confusion_matrix):
    # Number of classes
    num_classes = confusion_matrix.shape[0]
    
    # Initialize variables to accumulate metrics
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        tn = np.sum(confusion_matrix) - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    # Calculate macro-averaged metrics
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1_score = np.mean(f1_scores)
    
    # Calculate the standard deviation for each metric
    precision_std = np.std(precisions)
    recall_std = np.std(recalls)
    f1_std = np.std(f1_scores)

    accuracy_std = 0

    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1_score': macro_f1_score,
        'class_precisions': precisions,
        'class_recalls': recalls,
        'class_f1_scores': f1_scores,
        "accuracy_std": accuracy_std,
        "precision_std": precision_std,
        "recall_std": recall_std,
        "f1_std": f1_std
    }

def Linear_SVM(trainData, testData, trainLabels, testLabels, conf_matris_yazdir):
    print("Linear_SVM - Başlangıç")
    svmModel = LinearSVC(dual=True)
    svmModel.fit(trainData, trainLabels)
    #pickle.dump(svmModel, open('linearSvm', 'wb'))    
    #svmModel = pickle.load(open('linearSvm', 'rb'))
    predictions = svmModel.predict(testData)
    acc = svmModel.score(testData, testLabels)
    print("Linear SVM Model accuracy: {:.2f}%".format(acc * 100))
    conf_matrix = confusion_matrix(testLabels, predictions)
    if(conf_matris_yazdir==1):
        print(conf_matrix)
        
    # Çok sınıflı için decision_function kullanarak tahminler alın
    decision_values = svmModel.decision_function(testData)
    
    # Etiketleri ikili formatta dönüştür (binarize)
    testLabels_bin = label_binarize(testLabels, classes=np.unique(trainLabels))
    
    # AUC hesaplama
    auc = roc_auc_score(testLabels_bin, decision_values, multi_class='ovr')
    print("AUC (One-vs-Rest):", auc)
        
    # [tn, fp, fn, tp] = confusion_matrix(testLabels, predictions).ravel()
    full_data=np.vstack([trainData,testData])
    full_label=np.append(trainLabels,testLabels, axis=None)
    svmModel2= LinearSVC(dual=True)
    scores = cross_val_score(svmModel2, full_data, full_label, cv=cross_validation)
    metrics = calculate_metrics(conf_matrix)
    recall = cross_val_score(svmModel2, full_data, full_label, cv=cross_validation, scoring='recall')
    precision = cross_val_score(svmModel2, full_data, full_label, cv=cross_validation, scoring='precision')
    f1 = cross_val_score(svmModel2, full_data, full_label, cv=cross_validation, scoring='f1')
    scores=scores*100
    printmetrics(metrics)
    X_train = trainData
    y_train = trainLabels
    X_test = testData
    y_test = testLabels
    
    classifier = RandomForestClassifier(n_estimators=random_estimator)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    y_onehot_test.shape  # (n_samples, n_classes)

    label_binarizer.transform(["0"])

    class_of_interest = 0
    
    class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
    

    from sklearn.metrics import RocCurveDisplay
    
    display = RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_score[:, class_id],
        name=f"{class_of_interest} vs the rest",
        color="darkorange",
        plot_chance_level=True,
    )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="SVM- One-vs-Rest ROC curves:\nHigh vs (Low & Medium)",
    )
    temp=len(os.listdir("C:/Users/PC/Downloads/Revizyon1"))+1
    namex="C:/Users/PC/Downloads/Revizyon1/"+'SVM'+str(temp)+'.tiff'
    display.figure_.savefig(namex, format='tiff', dpi=300)
    print("Linear SVM with cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    auc_value = display.roc_auc
    print("auc : %0.4f" % auc_value)

    print("Linear_SVM - Son")
    return acc, conf_matrix, metrics


def Knn(trainData, testData, trainLabels, testLabels, conf_matris_yazdir,knn_sayi):
    print("Knn - Başlangıç")
    knnModel = KNeighborsClassifier(n_neighbors=knn_sayi, 	n_jobs=-1)
    knnModel.fit(trainData, trainLabels)
    predictions = knnModel.predict(testData)
    acc = knnModel.score(testData, testLabels)
    print("kNN accuracy: {:.2f}%".format(acc * 100))
    conf_matrix = confusion_matrix(testLabels, predictions)
    if(conf_matris_yazdir==1):
        print(conf_matrix)
    scores = cross_val_score(knnModel, np.vstack([trainData,testData]), np.append(trainLabels,testLabels, axis=None), cv=cross_validation)
    scores=scores*100
    metrics = calculate_metrics(conf_matrix)
    printmetrics(metrics)
    
    X_train = trainData
    y_train = trainLabels
    X_test = testData
    y_test = testLabels
    
    classifier = KNeighborsClassifier(n_neighbors=knn_sayi, 	n_jobs=-1)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    y_onehot_test.shape  # (n_samples, n_classes)

    label_binarizer.transform(["0"])

    class_of_interest = 0
    
    class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
    

    from sklearn.metrics import RocCurveDisplay
    display = RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_score[:, class_id],
        name=f"{class_of_interest} vs the rest",
        color="darkorange",
        plot_chance_level=True,
    )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="KNN (%d) - One-vs-Rest ROC curves:\nHigh vs (Low & Medium)" % knn_sayi,
    )
    temp=len(os.listdir("C:/Users/PC/Downloads/Revizyon1"))+1
    namex="C:/Users/PC/Downloads/Revizyon1/"+'kNN'+str(temp)+'.tiff'
    display.figure_.savefig(namex, format='tiff', dpi=300)
    print("kNN %d with cross validation accuracy: %0.2f (+/- %0.2f)" % (knn_sayi,scores.mean(), scores.std() * 2))
    
    auc_value = display.roc_auc
    print("auc : %0.4f" % auc_value)
    
    print("Linear_SVM - Son")
    return acc, conf_matrix, metrics

def Random_Forest(trainData, testData, trainLabels, testLabels, conf_matris_yazdir,random_estimator):
    print("Random_Forest - Başlangıç")
    randomForestModel = RandomForestClassifier(n_estimators=random_estimator)
    randomForestModel.fit(trainData, trainLabels)
    predictions = randomForestModel.predict(testData)
    acc = randomForestModel.score(testData, testLabels)
    print("randomForestModel: {:.2f}%".format(acc * 100))
    conf_matrix = confusion_matrix(testLabels, predictions)
    if(conf_matris_yazdir==1):
        print(conf_matrix)
    scores = cross_val_score(randomForestModel, np.vstack([trainData,testData]),np.append(trainLabels,testLabels, axis=None), cv=cross_validation)
    scores=scores*100
    metrics = calculate_metrics(conf_matrix)
    printmetrics(metrics)
    
    X_train = trainData
    y_train = trainLabels
    X_test = testData
    y_test = testLabels
    
    classifier = RandomForestClassifier(n_estimators=random_estimator)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    y_onehot_test.shape  # (n_samples, n_classes)

    label_binarizer.transform(["0"])

    class_of_interest = 0
    
    class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
    

    from sklearn.metrics import RocCurveDisplay
    
    display = RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_score[:, class_id],
        name=f"{class_of_interest} vs the rest",
        color="darkorange",
        plot_chance_level=True,
    )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Random Forest (%d)- One-vs-Rest ROC curves:\nHigh vs (Low & Medium)" % random_estimator,
    )

    temp=len(os.listdir("C:/Users/PC/Downloads/Revizyon1"))
    namex="C:/Users/PC/Downloads/Revizyon1/"+'RF'+str(temp+40)+'.tiff'
    display.figure_.savefig(namex, format='tiff', dpi=300)
        
    print("randomForestModel %d with cross validation accuracy: %0.2f (+/- %0.2f)" % (random_estimator,scores.mean(), scores.std() * 2))    
    
    auc_value = display.roc_auc
    print("auc : %0.4f" % auc_value)
    
    print("Random_Forest - Son")
    return acc, conf_matrix, metrics

def SVM_CrossV(data, labels, cross_validation):
    svmModel = LinearSVC(dual=True)
    scores = cross_val_score(svmModel, data, labels, cv=cross_validation)
    #print(scores*100)
    scores=scores*100
    print("SVM with cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean(), scores.std()

def printmetrics(metrics):
    print(f"Class Accuracy: {metrics['accuracy']:.4f} (+/- {metrics['accuracy_std']:.2f})")
    print(f"Class Precisions: {np.mean(metrics['class_precisions'])} (+/- {metrics['precision_std']:.2f})")
    print(f"Class Recalls: {np.mean(metrics['class_recalls'])} (+/- {metrics['recall_std']:.2f})")
    print(f"Class F1 Scores: {np.mean(metrics['class_f1_scores'])} (+/- {metrics['f1_std']:.2f})")
    
    return 


# kmeans
df = pd.read_excel('K-means_new1.xlsx')

# mean-shift
#df = pd.read_excel('meanshift-output-new3.xlsx')


# new1
X=df[["Outbound shipment total tonnage", "Outbound tonnage rate", "Outbound pallet rate", "Number of outbound pallets",	"Vehicle type",	"Vehicle departure location",	"Vehicle destination location",	"Vehicle direction",	"Number of inbound pallets",	"Inbound pallet rate",	"Inbound tonnage rate",	"Inbound shipment total tonnage",	"Total waiting time",	"Number of pending shipments",	"Number of waiting shipments",	"Day difference",	"Day difference2",	
"Waiting time performance coefficient",	"The cost coefficient",	"The maximum occupancy coefficient",	"The capacity factor",	"The shipment coefficient"]]

# new2
#df2 = df[["Outbound tonnage rate", "Outbound pallet rate", "Inbound pallet rate",	"Inbound tonnage rate", "Waiting time performance coefficient", "The cost coefficient",	"The maximum occupancy coefficient",	"The capacity factor",	"The shipment coefficient"]]

# new3
#df2 = df[["Waiting time performance coefficient", "The cost coefficient", "The maximum occupancy coefficient", "The capacity factor", "The shipment coefficient"]]


# kmeans#
y=df[["Cluster-3"]]

# mean-shift
# y=df[["Cluster-3"]]

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
scaler = StandardScaler()
le = LabelEncoder()
y = le.fit_transform(y)
X=preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40,random_state = 42)



# # feature selection code
# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Lasso
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV, KFold

# print("Shape of Train Features: {}".format(X_train.shape))
# print("Shape of Test Features: {}".format(X_test.shape))
# print("Shape of Train Target: {}".format(y_train.shape))
# print("Shape of Test Target: {}".format(y_test.shape))

# # parameters to be tested on GridSearchCV
# params = {"alpha":np.arange(0.00001, 10, 500)}

# # Number of Folds and adding the random state for replication
# kf=KFold(n_splits=5,shuffle=True, random_state=42)

# # Initializing the Model
# lasso = Lasso()

# # GridSearchCV with model, params and folds.
# lasso_cv=GridSearchCV(lasso, param_grid=params, cv=kf)
# lasso_cv.fit(X, y)
# print("Best Params {}".format(lasso_cv.best_params_))

# names=[ "Outbound shipment total tonnage", "Number of outbound pallets", "Outbound tonnage rate", "Outbound pallet rate", "Vehicle type", 
#         "Vehicle departure location", "Vehicle destination location", "Vehicle direction",
#         "Number of inbound pallets", "Inbound pallet rate", "Inbound shipment total tonnage", " Inbound tonnage rate", "Total waiting time (Hour)", 
#         "Number of waiting shipments (Hour)", " Number of pending shipments", "Day difference", "Day difference 2",
#         "Waiting time performance coefficient", "Cost coefficient", "Maximum occupancy coefficient", "Capacity factor", "Shipment Coefficient"]

# # names=[ "Outbound tonnage rate", "Outbound pallet rate", "Inbound pallet rate", " Inbound tonnage rate", "Waiting time performance coefficient", 
# #        "Cost coefficient", "Maximum occupancy coefficient", "Capacity factor", "Shipment Coefficient"]

# # names=[ "Waiting time performance coefficient", "Cost coefficient", "Maximum occupancy coefficient", "Capacity factor", "Shipment Coefficient"]


# # feature selection code
# print("Column Names: {}".format(names))
# # calling the model with the best parameter
# lasso1 = Lasso(alpha=0.00001)
# lasso1.fit(X_train, y_train)
# # Using np.abs() to make coefficients positive.  
# lasso1_coef = np.abs(lasso1.coef_)
# # plotting the Column Names and Importance of Columns. 
# fig = plt.figure(figsize=(10.20,5.80))
# plt.bar(names, lasso1_coef)
# plt.xticks(rotation=90)
# plt.grid()
# plt.title("Feature Selection Based on Lasso")
# plt.xlabel("Features")
# plt.ylabel("Importance")
# plt.ylim(0, max(lasso1_coef))
# plt.show()
# fig.savefig('myimage-meanshift.tiff', format='tiff', dpi=300)
# print(lasso1_coef)


# knn5 = KNeighborsClassifier(n_neighbors = 5)
# knn4 = KNeighborsClassifier(n_neighbors = 4)
# knn3 = KNeighborsClassifier(n_neighbors = 3)
# knn2 = KNeighborsClassifier(n_neighbors = 2)
# knn1 = KNeighborsClassifier(n_neighbors = 1)

# RF=RandomForestClassifier()

# knn5.fit(X_train, y_train)
# knn4.fit(X_train, y_train)
# knn3.fit(X_train, y_train)
# knn2.fit(X_train, y_train)
# knn1.fit(X_train, y_train)
# RF.fit(X_train, y_train)

# y_pred_6 = RF.predict(X_test)
# y_pred_5 = knn5.predict(X_test)
# y_pred_4 = knn4.predict(X_test)
# y_pred_3 = knn3.predict(X_test)
# y_pred_2 = knn2.predict(X_test)
# y_pred_1 = knn1.predict(X_test)

# from sklearn.metrics import accuracy_score
# # print("Accuracy with RF", accuracy_score(y_test, y_pred_6)*100)
# # 
# print("Accuracy with k=5", accuracy_score(y_test, y_pred_5)*100)
# print("Accuracy with k=4", accuracy_score(y_test, y_pred_4)*100)
# print("Accuracy with k=3", accuracy_score(y_test, y_pred_3)*100)
# print("Accuracy with k=2", accuracy_score(y_test, y_pred_2)*100)
# print("Accuracy with k=1", accuracy_score(y_test, y_pred_1)*100)

# predictions = RF.predict(X_test)
# cm = confusion_matrix(y_test, predictions, labels=RF.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=RF.classes_)
# plt.show()
# print(cm)


image_number=1345 #toplam görüntü sayısı 2116
class_number=1237 #1. sınıfa ait görüntü sayısı 1776
oznitelik_no=1 #0 hepsi, 1 color histogram, 2 hog, 3 lbp, 4 sift, 5 surf, 6 brief
siniflandirma=0 #0 hepsi, 1 svm, 2 knn, 3 random forest, 4 çapraz doğrulamalı svm
conf_matris_yazdir=1 #0 yazdirma, 1 yazdir
test_sizee=0.40
cross_validation=5
random_estimator=150
nu_svc_param=round((image_number-class_number)*100/image_number)/100
knn_sayi=3

for i in range(1,2,1):
    for j in range(1,2,1):
        a=6
        b=8
        if 1:
        
            acc,conf_matrix, metrics=Linear_SVM(X_train, X_test, y_train, y_test, conf_matris_yazdir)
            
            for ba in range(5,6):
                acc,conf_matrix, metrics=Knn(X_train, X_test, y_train, y_test, conf_matris_yazdir,ba)
            acc,conf_matrix, metrics=Random_Forest(X_train, X_test, y_train, y_test, conf_matris_yazdir,random_estimator)
            # score,error=SVM_CrossV(X, y, cross_validation)
        
        if 1:
            alpha2=1e-5
            from sklearn.neural_network import MLPClassifier
            gpc = MLPClassifier(solver='lbfgs', alpha=alpha2*1, hidden_layer_sizes=(a,b), random_state=42, max_iter=1000)
            gpc.fit(X_train, y_train)
            predictions = gpc.predict(X_test)
            acc = gpc.score(X_test, y_test)
            print("MLP: {:.2f}%".format(acc * 100))
            # print(y_test,predictions)
            conf_matrix = confusion_matrix(y_test, predictions)
            if(conf_matris_yazdir==1):
                print(conf_matrix)    
            
            scores = cross_val_score(gpc, np.vstack([X_train,X_test]), np.append(y_train,y_test, axis=None), cv=cross_validation)
            scores=scores*100
            
            metrics = calculate_metrics(conf_matrix)
            printmetrics(metrics)
           
            classifier = RandomForestClassifier(n_estimators=random_estimator)
            y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
            
            from sklearn.preprocessing import LabelBinarizer
            label_binarizer = LabelBinarizer().fit(y_train)
            y_onehot_test = label_binarizer.transform(y_test)
            y_onehot_test.shape  # (n_samples, n_classes)
        
            label_binarizer.transform(["0"])
        
        
            class_of_interest = 0
            
            class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
            
        
            from sklearn.metrics import RocCurveDisplay
            
            display = RocCurveDisplay.from_predictions(
                y_onehot_test[:, class_id],
                y_score[:, class_id],
                name=f"{class_of_interest} vs the rest",
                color="darkorange",
                plot_chance_level=True,
            )
            _ = display.ax_.set(
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                title="MLP - (alpha:%f,hidden layer(%d,%d) ) One-vs-Rest ROC curves:\nHigh vs ( Low & Medium)" % (alpha2*i,a,b) ,
            )
            
            temp=len(os.listdir("C:/Users/PC/Downloads/Revizyon1"))
            namex="C:/Users/PC/Downloads/Revizyon1/"+'MLP'+str(temp+1)+'.tiff'
            display.figure_.savefig(namex, format='tiff', dpi=300)
            print("MLP (%d,%d) with cross validation accuracy: %0.2f (+/- %0.2f)" % (a,b,scores.mean(), scores.std() * 2))
            
            auc_value = display.roc_auc
            print("auc : %0.4f" % auc_value)
            
            # print(acc, conf_matrix)
    
