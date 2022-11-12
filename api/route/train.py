import re
import time
import joblib
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


names = [
    "KNeighborsClassifier",
    "SVC",
    "Linear SVM",
    "Logistic Regression"
    "LogisticRegressionCV",
    "Bagging Classifier",
    "ExtraTrees Classifier",
    "RandomForest Classifier",
    "MultinomialNB",
    "DecisionTree Classifier",
    "AdaBoostClassifier",
    "GradientBoosting Classifier",
    "MLP Classifier()",
]

models = (
    KNeighborsClassifier(3),
    SVC(gamma='auto'),
    SVC(kernel='linear', probability=True),
    LogisticRegression(),
    LogisticRegressionCV(cv=5),
    BaggingClassifier(base_estimator=SVC(kernel='linear', probability=True), n_estimators=30, random_state=0),
    BaggingClassifier(),
    ExtraTreesClassifier(n_estimators=300),
    RandomForestClassifier(n_estimators=300),
    MultinomialNB(),
    DecisionTreeClassifier(max_depth=300),
    AdaBoostClassifier(n_estimators=300),
    GradientBoostingClassifier(n_estimators=300),
    MLPClassifier(hidden_layer_sizes=(256, 64, 16)),
)
def preprocessor(text):
    text      = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text      = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

def dataPrepare(filename):
    dt     = pd.read_csv(filename,error_bad_lines=False)
    dt     = dt.rename(columns={"0.0": "text", "0.0.1": "label", })
    dt     = dt.drop(columns={"Unnamed: 2"})
    dt     = dt.dropna(how="all").reset_index(drop=True)

    return dt

def train(X, y,X_test,y_test,  estimator,name):
    startTime   = time.time()
    tfidf       = TfidfVectorizer(strip_accents=None,lowercase=False,max_features=1000,ngram_range=(1,1))
    model       = Pipeline([('tfidf', tfidf), ('estimator', estimator)])

    model.fit(X, y)

    x_pred = model.predict(X) #test
    y_pred = model.predict(X_test) #test

    endTime     = time.time()
    f1_score    = metrics.f1_score(y_test, y_pred, average='macro')

    return {"model":model, "name": name,"f1":f1_score,"accuracy": "{} %".format(100 * accuracy_score(y_test, y_pred)),"time": (endTime - startTime)}

def runTrain(filename):

    data    = dataPrepare(filename= filename)
    X       = data['text'].apply(lambda x: preprocessor(str(x)))
    y       = data['label'].values.astype('U')

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=42)
    best_score  = 0.00
    modelList   = []

    for name, model in tqdm(zip(names, models),
                            desc="Pipeline Çalışıyor : ",
                            ascii=True,
                            unit_scale=True):

        Model       = train(X_train, y_train,X_test,y_test, model, name)


        if Model["f1"] > best_score:
            print("{0} isimli model kaydedildi. F1 Score : {1}".format(name, Model["f1"]))
            with open('api/models/best_model.pkl'.format(model), 'wb+') as f:
                joblib.dump(Model["model"], f)

            best_score = Model["f1"]

        modelList.append(
            {"name": Model["name"], "f1-score": Model["f1"], "accuracy": Model["accuracy"], "time": Model["time"]})
    return {"return" : True, "details": [modelList]}

if __name__ == "__main__" :
    runTrain("storage/dataset.csv")




