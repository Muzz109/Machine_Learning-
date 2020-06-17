
import codecs, json
import pandas as pd
from textblob import TextBlob
import re
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
import xlsxwriter
import boto3
import json
import io
from sklearn.calibration import CalibratedClassifierCV


# In[145]:


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\S+)", " ", tweet).split())

#sentimental Analysis
def textbl(tweet):
    text = clean_tweet(tweet)
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def sent(tweet):
    text = clean_tweet(tweet)
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.5:
        return 0
    else:
        return 1

#connect to s3 Bucket and read Input data file
s3 = boto3.client('s3')
object = s3.get_object(Bucket='gp-cloud-csuf',Key='depression.json')
serializedObject = object['Body'].read()
tweets = json.loads(serializedObject)

list_tweets = [list(elem.values()) for elem in tweets]
list_columns = list(tweets[0].keys())
df = pd.DataFrame(list_tweets, columns=list_columns)
df['Depressed'] = np.array([str(sent(tweet)) for tweet in df['text']])

#connect to s3 Bucket and write sentimental analysis data
output = io.BytesIO()
writer = pd.ExcelWriter(output, engine='xlsxwriter')
df.to_excel(writer)
writer.save()
data = output.getvalue()
s3 = boto3.resource('s3')
s3.Bucket('gp-cloud-csuf').put_object(Key='data.xlsx', Body=data)
warnings.filterwarnings('ignore')

#Predictive Analysis Using Diffrent supervised learning algorithms

d = df.drop(['user', 'text', 'url', 'fullname', 'timestamp', 'id', 'html'], axis=1)

y = d.Depressed
X = d.drop('Depressed', axis=1)
tot_count = len(df.index)
print(tot_count)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape)
print(X_test.shape)

# KNN Implementation

print("****************KNN*******************")
knn = KNeighborsClassifier(n_neighbors=3)
trained_knn = knn.fit(X_train, y_train)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
print("Training Accuracy:")
print(acc_knn)
test_knn = knn.fit(X_test, y_test)
acc_test_knn = round(knn.score(X_test, y_test) * 100, 2)
print("Testing Accuracy:")
print(acc_test_knn)
y_pred = knn.predict(X_test)
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))
print("Classification Matrix:")
print(metrics.classification_report(y_test, y_pred))
cross_val = cross_val_score(KNeighborsClassifier(), X, y, scoring='accuracy', cv=10)
print("Cross Validation value:")
cross_val_knn = round(cross_val.mean() * 100, 2)
print(cross_val_knn)
probability = knn.predict_proba(X_test)
probs = probability[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test.astype(float), probs)
sp = 0.0
sr = 0.0
for i in precision:
    sp = sp + i
precision_KNN = sp / len(precision)
for i in recall:
    sr = sr + i
recall_KNN = sp / len(recall)
knnroc = roc_auc_score(y_test.astype(float), probs)
print('AUC: %.3f' % knnroc)
fpr, tpr, thresholds = roc_curve(y_test.astype(float), probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
print("ROC curve:")
img_data_knn = io.BytesIO()
plt.savefig(img_data_knn, format='png')
img_data_knn.seek(0)
s3 = boto3.resource('s3')
bucket = s3.Bucket('gp-cloud-csuf')
bucket.put_object(Body=img_data_knn, ContentType='image/png', Key='KNeighborsClassifier.png')
plt.show()
plt.close()

# Logistic Regression Implementation
print("****************Logistic regression*******************")

logistic_regression_model = LogisticRegression()
trained_logistic_regression_model = logistic_regression_model.fit(X_train, y_train)
train_LR_accuracy = round(trained_logistic_regression_model.score(X_train, y_train) * 100, 2)
print("Training Accuracy:")
print(train_LR_accuracy)
test_LR_accuracy = round(trained_logistic_regression_model.score(X_test, y_test) * 100, 2)
print("Testing Accuracy:")
print(test_LR_accuracy)
probability = logistic_regression_model.predict_proba(X_test)
predicted = logistic_regression_model.predict(X_test)
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, predicted))
print("Classification Matrix:")
print(metrics.classification_report(y_test, predicted))
cross_val = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print("Cross Validation value:")
cross_val_lr = round(cross_val.mean() * 100, 2)
print(cross_val_lr)
probability = logistic_regression_model.predict_proba(X_test)
probs = probability[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test.astype(float), probs)
sp = 0.0
sr = 0.0
for i in precision:
    sp = sp + i
precision_LR = sp / len(precision)
for i in recall:
    sr = sr + i
recall_LR = sp / len(recall)
lrroc = roc_auc_score(y_test.astype(float), probs)
print('AUC: %.3f' % lrroc)
fpr, tpr, thresholds = roc_curve(y_test.astype(float), probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
print("ROC curve:")
img_data_lr = io.BytesIO()
plt.savefig(img_data_lr, format='png')
img_data_lr.seek(0)
s3 = boto3.resource('s3')
bucket = s3.Bucket('gp-cloud-csuf')
bucket.put_object(Body=img_data_lr, ContentType='image/png', Key='LogisticRegression.png')
plt.show()
plt.close()


# Random Forest Classifier Implementation
print("****************Random Forest*******************")

random_forest_model = RandomForestClassifier(n_estimators=600)
trained_random_forest_model = random_forest_model.fit(X_train, y_train)
train_RF_accuracy = round(trained_random_forest_model.score(X_train, y_train) * 100, 2)
print("Training Accuracy:")
print(train_RF_accuracy)
test_random_forest_model = random_forest_model.fit(X_test, y_test)
test_RF_accuracy = round(test_random_forest_model.score(X_test, y_test) * 100, 2)
predicted = random_forest_model.predict(X_test)
print("Testing Accuracy:")
print(test_RF_accuracy)
probability = random_forest_model.predict_proba(X_test)
predicted = random_forest_model.predict(X_test)
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, predicted))
print("Classification Matrix:")
print(metrics.classification_report(y_test, predicted))
cross_val = cross_val_score(RandomForestClassifier(), X, y, scoring='accuracy', cv=10)
print("Cross Validation value:")
cross_val_rf = round(cross_val.mean() * 100, 2)
print(cross_val_rf)
probability = random_forest_model.predict_proba(X_test)
probs = probability[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test.astype(float), probs)
sp = 0.0
sr = 0.0
for i in precision:
    sp = sp + i
precision_RF = sp / len(precision)
for i in recall:
    sr = sr + i
recall_RF = sp / len(recall)
rfroc = roc_auc_score(y_test.astype(float), probs)
print('AUC: %.3f' % rfroc)
fpr, tpr, thresholds = roc_curve(y_test.astype(float), probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
print("ROC curve:")
img_data_rfc = io.BytesIO()
plt.savefig(img_data_rfc, format='png')
img_data_rfc.seek(0)
s3 = boto3.resource('s3')
bucket = s3.Bucket('gp-cloud-csuf')
bucket.put_object(Body=img_data_rfc, ContentType='image/png', Key='RandomForestClassifier.png')
plt.show()
plt.close()

#SVN Implementation
print("****************SVM*******************")

svm_model = svm.LinearSVC()
trained_svm_model = svm_model.fit(X_train, y_train)
train_svm_accuracy = round (trained_svm_model.score(X_train, y_train)*100, 2)
print ("Training Accuracy:")
print(train_svm_accuracy)
test_svm_model = svm_model.fit(X_test, y_test)
test_svm_accuracy = round (test_svm_model.score(X_test, y_test)*100, 2)
predicted = svm_model.predict(X_test)
print ("Testing Accuracy:")
print(test_svm_accuracy)
predicted = svm_model.predict(X_test)
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, predicted))
print("Classification Matrix:")
print(metrics.classification_report(y_test, predicted))
cross_val = cross_val_score(svm.LinearSVC(), X, y, scoring='accuracy', cv=10)
print("Cross Validation value:")
cross_val_svm = round(cross_val.mean()*100, 2)
print(cross_val_svm)
cclf = CalibratedClassifierCV(base_estimator=svm.LinearSVC(penalty='l2', dual=False), cv=10)
cclf.fit(X_train, y_train)
probability = cclf.predict_proba(X_test)
probs = probability[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test.astype(float), probs)
sp = 0.0
sr = 0.0
for i in precision:
    sp = sp + i
precision_svm = sp/len(precision)
for i in recall:
    sr = sr + i
recall_svm = sp/len(recall)
svmroc = roc_auc_score(y_test.astype(float), probs)
print('AUC: %.3f' % svmroc)
fpr, tpr, thresholds = roc_curve(y_test.astype(float), probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
print("ROC curve:")
img_data_svn = io.BytesIO()
plt.savefig(img_data_svn, format='png')
img_data_svn.seek(0)
s3 = boto3.resource('s3')
bucket = s3.Bucket('gp-cloud-csuf')
bucket.put_object(Body=img_data_svn, ContentType='image/png', Key='SVN.png')
plt.show()
plt.close()

# Accuracy of algorithms
algorithm = ['K-nearest neighbor', 'Logistic Regression', 'Random Forest', 'SVM']
trainaccuracy = [acc_knn, train_LR_accuracy, train_RF_accuracy, train_svm_accuracy]
testaccuracy = [acc_test_knn, test_LR_accuracy, test_RF_accuracy, test_svm_accuracy]
index = np.arange(len(algorithm))

fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, trainaccuracy, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Training Accuracy')

rects2 = plt.bar(index + bar_width, testaccuracy, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Testing Accuracy')

plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Accuracy of algorithms')
plt.xticks(index + bar_width, algorithm)
plt.legend()
plt.tight_layout()
img_data_aoa = io.BytesIO()
plt.savefig(img_data_aoa, format='png')
img_data_aoa.seek(0)
s3 = boto3.resource('s3')
bucket = s3.Bucket('gp-cloud-csuf')
bucket.put_object(Body=img_data_aoa, ContentType='image/png', Key='Accuracy_of_algorithms.png')
plt.show()
plt.close()

#Precision/Recall of algorithms
algorithm = ['K-nearest neighbor', 'Logistic Regression', 'Random Forest', 'SVM']
precision = [precision_KNN, precision_LR, precision_RF, precision_svm]
recall = [recall_KNN, recall_LR, recall_RF, recall_svm]
index = np.arange(len(algorithm))

fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, trainaccuracy, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Precision')

rects2 = plt.bar(index + bar_width, testaccuracy, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Recall')

plt.xlabel('Algorithms')
plt.ylabel('Units')
plt.title('Precision/Recall of algorithms')
plt.xticks(index + bar_width, algorithm)
plt.legend()
plt.tight_layout()
img_data_poa = io.BytesIO()
plt.savefig(img_data_poa, format='png')
img_data_poa.seek(0)
s3 = boto3.resource('s3')
bucket = s3.Bucket('gp-cloud-csuf')
bucket.put_object(Body=img_data_poa, ContentType='image/png', Key='Precision_of_algorithms.png')
plt.show()
plt.close()

#Cross Validation Accuracy of Algorithms
crossvalidationaccuracy = [cross_val_knn, cross_val_lr, cross_val_rf, cross_val_svm]
y_pos = np.arange(len(algorithm))

plt.barh(y_pos, crossvalidationaccuracy, align='center', alpha=0.5)
plt.yticks(y_pos, algorithm)
plt.xlabel('Accuracy')
plt.title('Cross Validation Accuracy of Algorithms')
img_data_cvaoa = io.BytesIO()
plt.savefig(img_data_cvaoa, format='png')
img_data_cvaoa.seek(0)
s3 = boto3.resource('s3')
bucket = s3.Bucket('gp-cloud-csuf')
bucket.put_object(Body=img_data_cvaoa, ContentType='image/png', Key='Cross_Validation_Accuracy_of_algorithms.png')
plt.show()
plt.close()

# efficient algorithm

def efficientAlgo():
    countKNN = 0
    countLR = 0
    countRF = 0
    countSVM = 0
    Knn_Accuracy = abs(acc_test_knn - acc_knn)
    LR_accuracy = abs(test_LR_accuracy - train_LR_accuracy)
    RF_accuracy = abs(test_RF_accuracy - train_RF_accuracy)
    SVM_accuracy = abs(test_svm_accuracy - train_svm_accuracy)
    max_Accuracy = (Knn_Accuracy if (Knn_Accuracy < LR_accuracy and Knn_Accuracy < RF_accuracy and Knn_Accuracy < SVM_accuracy)
            else (LR_accuracy if (LR_accuracy < RF_accuracy and LR_accuracy < SVM_accuracy)
            else (RF_accuracy if RF_accuracy < SVM_accuracy else SVM_accuracy)))
    max_ROC = (knnroc if (knnroc > lrroc and knnroc > rfroc and knnroc > svmroc)
            else (lrroc if (lrroc > rfroc and lrroc > svmroc)
            else (rfroc if rfroc > svmroc else svmroc)))
    max_CV = (cross_val_knn if (cross_val_knn > cross_val_lr and cross_val_knn > cross_val_rf and cross_val_knn > cross_val_svm)
            else (cross_val_lr if (cross_val_lr > cross_val_rf and cross_val_lr > cross_val_svm)
            else (cross_val_rf if cross_val_rf > cross_val_svm else cross_val_svm)))
    max_prec = (precision_KNN if (precision_KNN > precision_LR and precision_KNN > precision_RF and precision_KNN > precision_svm)
                else (precision_LR if (precision_LR > precision_RF and precision_LR > precision_svm)
                else (precision_RF if (precision_RF > precision_svm) else precision_svm)))
    max_recall = (recall_KNN if (recall_KNN > recall_LR and recall_KNN > recall_RF and recall_KNN > recall_svm)
            else (recall_LR if (recall_LR > recall_RF and recall_LR > recall_svm)
            else (recall_RF if recall_RF > recall_svm else recall_svm)))
    if max_Accuracy == Knn_Accuracy:
        countKNN += 1
    elif max_Accuracy == LR_accuracy:
        countLR += 1
    elif max_Accuracy == RF_accuracy:
        countRF += 1
    elif max_Accuracy == SVM_accuracy:
        countSVM += 1
    if max_ROC == knnroc:
        countKNN += 1
    elif max_ROC == lrroc:
        countLR += 1
    elif max_ROC == rfroc:
        countRF += 1
    elif max_ROC == svmroc:
        countSVM += 1
    if max_CV == cross_val_knn:
        countKNN += 1
    elif max_CV == cross_val_lr:
        countLR += 1
    elif max_CV == cross_val_rf:
        countRF += 1
    elif max_CV == cross_val_svm:
        countSVM += 1
    if max_prec == precision_KNN:
        countKNN += 1
    elif max_prec == precision_LR:
        countLR += 1
    elif max_prec == precision_RF:
        countRF += 1
    elif max_prec == precision_svm:
        countSVM += 1
    if max_recall == recall_KNN:
        countKNN += 1
    elif max_recall == recall_LR:
        countLR += 1
    elif max_recall == recall_RF:
        countRF += 1
    elif max_recall == recall_svm:
        countSVM += 1
    efficientAlgorithm = (countKNN if (countKNN > countLR and countKNN > countRF and countKNN > countSVM)
            else (countLR if (countLR > countRF and countLR > countSVM)
            else (countRF if countRF > countSVM else countSVM)))
    if efficientAlgorithm == countKNN:
        print("The efficient algorithm is K-Nearest Neighbors")
    elif efficientAlgorithm == countLR:
        print("The efficient algorithm is Logistic Regression")
    elif efficientAlgorithm == countRF:
        print("The efficient algorithm is Random Forest")
    else:
        print("The efficient algorithm is Support Vector Machine")


efficientAlgo()
