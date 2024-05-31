import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Iris 데이터셋 로드 및 이진 분류 설정
iris = datasets.load_iris()
X, y = iris.data, iris.target
X2, y2 = X[y != 2], y[y != 2]  # Setosa(0) vs Non-setosa(1)

# SGDClassifier로 교차 검증
clf = SGDClassifier(random_state=7)
cv = KFold(5, shuffle=True, random_state=7)
print(cross_val_score(clf, X2, y2, cv=cv))

# SGDClassifier로 전체 데이터셋에 대한 학습 및 평가
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
clf_all = SGDClassifier(random_state=7)
clf_all.fit(X_train, y_train)
print(clf_all.score(X_test, y_test))
print(cross_val_score(clf_all, X, y, cv=cv))

# 로지스틱 회귀: 공부한 시간과 합격 여부
pass_time = [3, 8, 9, 9.5, 10, 12, 14, 14.5, 15, 16, 16.5, 17, 17, 17, 17.5, 20, 20, 20]
fail_time = [1, 2, 2.1, 2.6, 2.7, 2.8, 2.9, 3, 3.2, 3.4, 3.5, 3.6, 4, 5, 5.2, 5.4, 6, 6.5, 7, 8]
X = np.hstack((pass_time, fail_time))
y = np.hstack(([1]*len(pass_time), [0]*len(fail_time)))

fig = plt.figure(figsize=(4, 2))
plt.xlim(0, 21)
plt.xlabel("Study time")
plt.scatter(X, y)
plt.ylim(-0.1, 1.1)
plt.ylabel("Pass rate")
plt.show()

model = LogisticRegression()
model.fit(X.reshape(-1, 1), y)
print(model.coef_)
print(model.intercept_)

def logreg(z):
    return 1 / (1 + np.exp(-z))

fig = plt.figure(figsize=(4, 2))
plt.xlim(0, 22)
plt.xlabel('Study time')
plt.scatter(X, y, s=50)
plt.ylim(-0.1, 1.1)
plt.ylabel('Pass rate')
XX = np.linspace(0.5, 21, 100)
yy = logreg(model.coef_ * XX + model.intercept_)[0]
plt.plot(XX, yy, c='r')

sample = np.array([7.0])
print(model.predict(sample.reshape(-1, 1)))
sample = np.array([13])
print(model.predict_proba(sample.reshape(-1, 1)))

# 유방암 데이터셋 예제
data = pd.read_csv('data/breast_cancer.csv')
print(data.shape)
print(data.head(2))
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
df = data[['diagnosis', 'radius_mean', 'texture_mean']]
print(df.head(2))

model = LogisticRegression()
y = df['diagnosis']

# radius_mean을 이용한 분류
features = ['radius_mean']
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model.fit(X_train, y_train)
print("Score: {:.2%}".format(model.score(X_test, y_test)))
print(model.coef_)
print(model.intercept_)

plt.figure(figsize=(4, 2))
plt.scatter(X_train, y_train, s=1)
plt.xlabel("radius_mean")
plt.ylabel("M(1) or B(0)")

XX = np.linspace(7.5, 40, 100)
plt.plot(XX, logreg(model.coef_ * XX + model.intercept_)[0], c='r')

# texture_mean을 이용한 분류
features = ['texture_mean']
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model.fit(X_train, y_train)
print("Score: {:.2%}".format(model.score(X_test, y_test)))
print(model.coef_)
print(model.intercept_)

plt.figure(figsize=(4, 2))
plt.scatter(X_train, y_train, s=1)
plt.xlabel("texture_mean")
plt.ylabel("M(1) or B(0)")

XX = np.linspace(7.5, 40, 100)
plt.plot(XX, logreg(model.coef_ * XX + model.intercept_)[0], c='r')

# 두 개의 특성을 사용한 분류
features = ['radius_mean', 'texture_mean']
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.coef_)
print(model.intercept_)
print("Score: {:.2%}".format(model.score(X_test, y_test)))

# 모든 특성 사용
data.drop('id', axis=1, inplace=True)
data.drop('Unnamed: 32', axis=1, inplace=True)
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model_all = LogisticRegression(max_iter=5000)
model_all.fit(X_train, y_train)
print("Score: {:.2%}".format(model_all.score(X_test, y_test)))

# KNN
for i in range(1, 21, 2):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    print("K = {}, Score: {:.2%}".format(i, knn.score(X_test, y_test)))

# Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
print("Decision Tree Score: {:.2%}".format(tree.score(X_test, y_test)))

# RandomForest
rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)
print("RandomForest Score: {:.2%}".format(rfc.score(X_test, y_test)))