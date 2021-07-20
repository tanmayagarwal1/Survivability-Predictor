# Train - Test Split 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X = df_t.drop(['Survived','PassengerId','Fare'],axis=1)
y = df_t['Survived']

# Decision Tree Classifier 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=3296)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=1,max_depth=7,min_samples_split=2).fit(X_train,y_train)
clf.score(X_test,y_test)

# K- nearest neighbours 
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)
neigh.score(X_test,y_test)

# Logistic Regression 
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression().fit(X_train,y_train)
lr.score(X_test,y_test)

# Support Vector Machines 
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
clf = make_pipeline(SVC(gamma='auto')).fit(X_train,y_train)
clf.score(X_test,y_test)

# XGBBoost 
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
model.score(X_test,y_test)#58080

# Neural Networks 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras
model = Sequential()
n_cols = X_train.shape[1]
model.add(Dense(39, activation='relu', input_shape=(n_cols,)))
model.add(Dense(27, activation='selu'))
model.add(Dense(19, activation='softplus'))
model.add(Dense(1, kernel_regularizer=keras.regularizers.l2(1e-2),activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, shuffle=False,epochs=160 )