import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import sheryanalysis as sh
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC  
from sklearn.metrics import f1_score
import joblib
df = pd.read_csv("heart.csv")
print(df.columns)
print(df.info())
print(df.describe())
print(f"is null:\n",df.isnull().sum())
print(df.duplicated().sum())
def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde=True)
df.columns = df.columns.str.lower()
plotting('age',1)
plotting('restingbp',2)   
plotting('cholesterol',3)
plotting('maxhr',4)
plt.show()
# by figure we got to know that there are too many patients with cholestrol value of 0 which impossible 
# therefore we are going to replace the 0 with cholestrol mean 
ch_mean=df.loc[df['cholesterol']!=0,'cholesterol'].mean()
df['cholesterol']=df['cholesterol'].replace(0,ch_mean)
df['cholesterol']=df['cholesterol'].round(2)

# similarly for restingbp
ch_mean=df.loc[df['restingbp']!=0,'restingbp'].mean()
df['restingbp']=df['restingbp'].replace(0,ch_mean)
df['restingbp']=df['restingbp'].round(2)

plotting('age',1)
plotting('restingbp',2)
plotting('cholesterol',3)
plotting('maxhr',4)
plt.show()
print(sh.analyze(df))

df_encode=pd.get_dummies(df,drop_first=True)
df_encode=df_encode.astype(int)
# num_cols=['age', 'restingbp', 'cholesterol', 'maxhr', 'oldpeak']
# scaler=StandardScaler()
# df_encode[num_cols] = scaler.fit_transform(df_encode[num_cols])
# it would cause data leakegae as fit will calculate the mean and sd for alll the datasets including train and testing which woould effect the model prediction
print(df_encode.head())
print(df_encode.columns)
x=df_encode.drop('heartdisease',axis=1)
y=df_encode['heartdisease']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# model=LogisticRegression()
from sklearn.pipeline import Pipeline
# model = Pipeline([
#     ("scaler", StandardScaler()),
#     ("logistic", LogisticRegression())
# ])
# model.fit(X_train, y_train)
# y_pred=model.predict(X_test)


# for logesticregression


# accuracy = accuracy_score(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))


# Pick 5 patients from the test set
# sample = X_test.iloc[:5]

# predictions = model.predict(sample)

# print("Predictions:", predictions)

# actual = y_test.iloc[:5]

# print("Actual:     ", actual.values)
# # KNN
# mod = Pipeline([
#     ("scaler", StandardScaler()),
#     ("knn", KNeighborsClassifier())
# ])
# mod.fit(X_train, y_train)
# y_pred_knn = mod.predict(X_test)
# accuracy_knn = accuracy_score(y_test, y_pred_knn)
# print("KNN Accuracy:", accuracy_knn)
# print("\nKNN Classification Report:")
# print(classification_report(y_test, y_pred_knn))
# print("\nKNN Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_knn))

# #  naive bayes
# model_nb=GaussianNB()
# model_nb.fit(X_train, y_train)
# y_pred_nb = model_nb.predict(X_test)
# accuracy_nb = accuracy_score(y_test, y_pred_nb)
# print("Naive Bayes Accuracy:", accuracy_nb)
# print("\nNaive Bayes Classification Report:")
# print(classification_report(y_test, y_pred_nb))
# print("\nNaive Bayes Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_nb))

# # decision tree

# model_dt = DecisionTreeClassifier()
# model_dt.fit(X_train, y_train)
# y_pred_dt = model_dt.predict(X_test)
# accuracy_dt = accuracy_score(y_test, y_pred_dt)
# print("Decision Tree Accuracy:", accuracy_dt)
# print("\nDecision Tree Classification Report:")
# print(classification_report(y_test, y_pred_dt))
# print("\nDecision Tree Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_dt))

# # svm

# model_svm = SVC()
# model_svm.fit(X_train, y_train)
# y_pred_svm = model_svm.predict(X_test)
# accuracy_svm = accuracy_score(y_test, y_pred_svm)
# print("SVM Accuracy:", accuracy_svm)
# print("\nSVM Classification Report:")
# print(classification_report(y_test, y_pred_svm))
# print("\nSVM Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_svm))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models={
    "logistic_regression": LogisticRegression(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "decision_tree": DecisionTreeClassifier(),
    "svm": SVC()
}
result=[]
for name,model in models.items():
    model.fit(X_train_scaled,y_train)
    y_pred=model.predict(X_test_scaled)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    result.append({
        "model":name,
        "accuracy":round(acc, 2),
        "f1_score":round(f1, 2)
    })
print(pd.DataFrame(result).sort_values(by="accuracy",ascending=False))
joblib.dump(models["logistic_regression"], 'heart_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(x.columns.to_list(), 'features.pkl')