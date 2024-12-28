import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score

def data_preprocessing(df, test_size=0.2):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Timestamp'] = df['Timestamp'].apply(lambda x: x.value)
    df['Timestamp'] = (df['Timestamp'] - df['Timestamp'].min()) / (df['Timestamp'].max() - df['Timestamp'].min())

    cat_cols = df.select_dtypes(include='object').columns.to_list()

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    y = df_scaled['Is Laundering']
    X = df_scaled.drop('Is Laundering', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(classifiers, X_train, X_test, y_train, y_test):
    for key, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        training_score = cross_val_score(classifier, X_train, y_train, cv=5)
        print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
        test_score = classifier.score(X_test, y_test)
        print("Classifiers: ", classifier.__class__.__name__, "Has a test score of", round(test_score, 2) * 100, "% accuracy score\n")
    
    # model.fit(X_train, y_train)
    # preds = clf.predict(X_test)
    
    # auprc = average_precision_score(y_test, preds)
    # print(f"AUPRC: {auprc}")

df = pd.read_csv('/Users/andrzej/python_programs/CODE/data/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans.csv')
X_train, X_test, y_train, y_test = data_preprocessing(df)


classifiers = {
        # "IsolationForest":IsolationForest(),
        "LogisiticRegression": LogisticRegression(),
        "KNearest": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }
train_model(classifiers, X_train, X_test, y_train, y_test)