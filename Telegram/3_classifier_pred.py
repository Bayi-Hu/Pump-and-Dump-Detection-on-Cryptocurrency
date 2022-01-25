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

pre_select_sample = pd.read_csv("./PreSelection/preselect_sample", sep="\t", names=["channel_id", "sample_id", "message", "message_wo_stop", "date", "weekday", "on_time", "crypto_signal", "exchange_signal", "keyword_signal", "length"])
labeled_sample = pd.read_csv("./Labeled/label.txt", sep=" ", names=["label", "channel_id", "sample_id"])

labeled_sample = labeled_sample[labeled_sample.label!="?"]
labeled_sample.loc[labeled_sample.label == "2", "label"]= "0"
pre_select_sample = pre_select_sample.dropna(axis=0, how="any")

pre_select_sample = pd.merge(pre_select_sample, labeled_sample, how="left", on=["channel_id","sample_id"])
pre_select_sample.label.fillna("-1", inplace=True)

tf_idf = TfidfVectorizer(ngram_range=(1, 3), min_df=5, tokenizer=lambda x: x.split(), max_features=20000, use_idf = True)
X_all = tf_idf.fit_transform(pre_select_sample.message_wo_stop)

X_predict = X_all[np.where(pre_select_sample.label.values == "-1")[0]]
X_train = X_all[np.where(pre_select_sample.label.values != "-1")[0]]
y_train = pre_select_sample.label.values[np.where(pre_select_sample.label.values != "-1")[0]]

# use GBDT model
gbdt_model = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300, min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10)
gbdt_model.fit(X_train, y_train)
y_pred_proba = gbdt_model.predict_proba(X_predict)

y_pred = np.zeros_like(y_pred_proba[:,1])

threshold = 0.25
y_pred[np.where(y_pred_proba[:,1]>=0.25)[0]] = 1

pred_sample = pre_select_sample[pre_select_sample.label.values=="-1"]
pred_sample_final = pd.concat([pred_sample.reset_index(), pd.DataFrame(y_pred, columns=["pred_label"])], axis=1)

# 只选出pred_label == 1的sample进行保存
pred_pump_sample = pred_sample_final[pred_sample_final.pred_label == 1].reset_index()

pred_pump_sample[["channel_id","sample_id", "message", "date", "pred_label"]].to_csv("./Labeled/pred_pump_annouce.csv", index=False, sep="\t")






