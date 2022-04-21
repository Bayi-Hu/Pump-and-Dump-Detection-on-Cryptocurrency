import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.metrics import average_precision_score,roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def load_obj(file):
    with open("./PreSelection/" + file, "rb") as f:
        return pickle.load(f)

labeled_sample = pd.read_csv("./Labeled/label.txt", sep=" ", names=["label", "channel_id", "sample_id"])
labeled_sample = labeled_sample[labeled_sample.label!="?"]
pre_select_sample = pd.read_csv("./PreSelection/preselect_sample", sep="\t", names=["channel_id", "sample_id", "message", "message_wo_stop", "date", "weekday", "on_time", "crypto_signal", "exchange_signal", "keyword_signal", "length"])
pre_select_sample = pd.merge(labeled_sample, pre_select_sample, how="inner", on=["channel_id","sample_id"])
pre_select_sample = pre_select_sample.dropna(axis=0, how="any")

class color: # Text style
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def model_train(model, X_train, y_train):
    model.fit(X_train,y_train)
    return model

def model_accuracy(model,X_test,y_test, model_name):
    print("Classifier: ", model_name)
    pred_y = model.predict(X_test)
    print(color.BOLD+'Confusion Matrix:\n'+color.END,confusion_matrix(y_test, pred_y))
    print (color.BOLD+'Report : '+color.END)
    print (classification_report(y_test, pred_y))
    acc_list.append((round(accuracy_score(y_test, pred_y),4)*100))
    pr_list.append((round(precision_score(y_test, pred_y, average='weighted'),4)*100))
    re_list.append((round(recall_score(y_test, pred_y, average='weighted'),4)*100))
    f1_list.append((round(f1_score(y_test, pred_y, average='weighted'),4)*100))


# cleaned data

tf_idf = TfidfVectorizer(ngram_range=(1,3), min_df=5, tokenizer=lambda x: x.split(), max_features=20000, use_idf = True)

X = tf_idf.fit_transform(pre_select_sample.message_wo_stop)
y = pre_select_sample.label.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr_model = LogisticRegression(random_state = 123,C =0.08)
gbdt_model = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300, min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10)
rf_model = RandomForestClassifier(n_estimators=10, criterion ='entropy', random_state = 0)
mnb_model = MultinomialNB(alpha=0.05)
model_names = ['Logistic Regression','GBDT','Random Forest','Naive Bayes']
ml_models = [lr_model,gbdt_model,rf_model,mnb_model]

trained_ml_models = []
for i in ml_models:
    tt = model_train(i,X_train, y_train)
    trained_ml_models.append(tt)

acc_list = []
pr_list = []
re_list = []
f1_list = []

for i in range(len(trained_ml_models)):
    md = trained_ml_models[i]
    name = model_names[i]
    model_accuracy(md, X_test, y_test, name)


performance_matrix = pd.DataFrame({'Accuracy':acc_list,'Precision':pr_list,
                                   'Recall':re_list,'F1 Score':f1_list},
                                  index =model_names)

print(performance_matrix)

