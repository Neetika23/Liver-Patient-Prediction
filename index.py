# Necessary Imports
from sklearn.metrics import accuracy_score()
form sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
form sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier

# Load the dataset(link is in the READ_ME)

# splitting
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=SEED)

# Set seed for reproducibility
SEED=1

# Instantiate lr
lr = LogisticRegression(random_state=SEED)

# Instantiate knn
knn = KNeighborsClassifier(n_neighbors=27)

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers: 
 
    # Fit clf to the training set
    clf.fit(X_train,y_train)    
   
    # Predict y_pred
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test,y_pred) 
   
    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))
    
    # Logistic Regression : 0.747
    # K Nearest Neighbours : 0.724
    # Classification Tree : 0.730
    
    # Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier

# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)     

# Fit vc to the training set
vc.fit(X_train,y_train)   

# Evaluate the test set predictions
y_pred = vc.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test,y_pred)
print('Voting Classifier: {:.3f}'.format(accuracy))

# Voting Classifier : 0.753

#  the voting classifier achieves a test set accuracy of 75.3%. This value is greater than that achieved by LogisticRegression.
