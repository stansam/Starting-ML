import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report

# *************** DATASET *******************

cols = ['fLength', 'fWidth', 'fsize', 'fConc', 'fConc1',  'fAsym', 'fM3Long', 'M3Trans', 'fAlpha', 'fDist', 'class' ]
df = pd.read_csv("magic04.data", names=cols)
df['class'] = (df['class'] == "g").astype(int)
#print(df)
head = df.head()
#print(head)
for label in cols[:-1]:
    plt.hist(df[df['class'] == 1][label], color='blue', label='gamma', alpha=0.7, density=True )
    plt.hist(df[df['class'] == 0][label], color='red', label='hydrons', alpha=0.7, density=True )
    plt.title(label)
    plt.ylabel("Probabilty")
    plt.xlabel(label)
    #plt.legend()
    #plt.show()

### TRAIN, VALIDATION, TEST DATASET***************
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (-1, 1))))
    return data, x, y

train, x_train, y_train = scale_dataset(train, oversample=True)
valid, x_valid, y_valid = scale_dataset(valid, oversample=False)
test, x_test, y_test = scale_dataset(test, oversample=False)

print(sum(y_train==1))
print(sum(y_train==0))
## print(y_train)

# ***************kNN************
knn_model = KNeighborsClassifier(n_neighbors=20)
knn_model.fit(x_train, y_train)
y_pred = knn_model.predict(x_test)
print(classification_report(y_test, y_pred))
print(y_pred)
print(y_test)
