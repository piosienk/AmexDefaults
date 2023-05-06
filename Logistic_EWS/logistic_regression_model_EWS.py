import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

np.random.seed(123)

# Select variables with WoE buckets
with open('./Additional_data/WoE_binning_merged_short.pickle', 'rb') as file:
    woe_dict = pickle.load(file)

woe_dict_mod = woe_dict.copy()
for key in woe_dict.keys():
    # If we have only one bucket - drop key from dictionary and from training sample
    if pd.DataFrame(woe_dict[key]).shape[0] == 1:
        del woe_dict_mod[key]

variables = list(woe_dict_mod.keys())
variables.append("target")
variables = [i for i in variables[::-1]]

# Import data
df_y = pd.read_parquet("./Additional_data/df_train_y_reduced_merged.parquet")
df_x = pd.read_parquet(data_path + "/train_data_woe_merged_short.parquet")
print(df_x.shape)

df = pd.concat([df_y,df_x],axis=1)
df = df.loc[:,variables]
df = df.fillna(0)
print(df.shape)

# Select Variables based on Random Forest model
# Split data into train and validation for Random Forest
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns="target"), df.target, test_size=0.1, random_state=123)

# random forest model to select variables
rf = RandomForestClassifier(n_estimators=100, min_samples_split=25, n_jobs=-1, random_state=123).fit(X_train,y_train)

# Extract top 20 variables and plot their importance
f_i = list(zip(variables[1:],rf.feature_importances_))
f_i.sort(key = lambda x : x[1])
plt.subplots(figsize=(10,5))
plt.barh([x[0] for x in f_i[-20:]],[x[1] for x in f_i[-20:]])
plt.show()

selected_features = [x[0] for x in f_i[-20:]]

# Save selected features
with open('./Additional_data/selected_features_short.pickle', 'wb') as file:
    pickle.dump(selected_features, file)

# Logistic Regression model
lm = LogisticRegression(penalty="l1", solver="saga").fit(X_train.loc[:,selected_features],y_train)

# Make prediction
y_predict = lm.predict(X_test.loc[:, selected_features])
y_predict_prob = lm.predict_proba(X_test.loc[:, selected_features])
print("Validation accuracy of LR + WoE: ", (y_predict==y_test).mean())

fpr, tpr, thresholds = metrics.roc_curve(y_test.to_numpy(), y_predict_prob[:,1])
print("Validation ROC AUC of LR + WoE: ",
      metrics.roc_auc_score(y_test.to_numpy(), y_predict_prob[:,1]))

# Precision-recall curve for LR
display = metrics.PrecisionRecallDisplay.from_predictions(y_test, y_predict_prob[:,1], name="LR+WoE")
_ = display.ax_.set_title("Validation 2-class Precision-Recall curve")
plt.show()

# Save model
with open('./Final_models/lm_model_short_obs.pickle', 'wb') as file:
    pickle.dump(lm, file)
