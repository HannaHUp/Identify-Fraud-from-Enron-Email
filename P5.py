
# coding: utf-8

# In[22]:


import warnings
warnings.filterwarnings('ignore')

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
dataset = data_dict 


# In[23]:


#https://bbs.csdn.net/topics/390821957
import pandas as pd
dataset=pd.DataFrame(dataset)
dataset=pd.DataFrame.transpose(dataset)
dataset.head()


# In[5]:


dataset.shape


# In[ ]:


#该数据有146个人，21个属性。其中属性里面可能包含“NaN“，需要清理


# In[46]:


dataset.isnull().any()
#每一列都有nan


# In[53]:


dataset=pd.DataFrame.transpose(dataset)
dataset.isnull().any()
#每一个人都有nan


# In[77]:




for columns in dataset.columns:
    k=0
    for value in dataset[columns]:
        if value == 'NaN':
            k += 1
            p = 100.0*k/len(dataset[columns])
    print columns,k,p


# In[78]:


dataset=pd.DataFrame.transpose(dataset)
for columns in dataset.columns:
    k=0
    for value in dataset[columns]:
        if value == 'NaN':
            k += 1
            p = 100.0*k/len(dataset[columns])
    print columns,k,p


# In[ ]:


#最高的前五个
loan_advances                142
director_fees                129
restricted_stock_deferred    128
deferral_payments            107
deferred_income               97
#最高的前五个人
LOCKHART EUGENE E                20
GRAMM WENDY L                    18
WROBEL BRUCE                     18
WHALEY DAVID A                   18
THE TRAVEL AGENCY IN THE PARK    18


# In[80]:


dataset=pd.DataFrame.transpose(dataset)
dataset.loc["LOCKHART EUGENE E",:]
#发现除了poi，都是空值，应该注意


# In[46]:


def data_no_nan(columns):
    data = []
    for value in dataset[columns]:
        if value != 'NaN':
            data.append(value)        
    return data

for columns in dataset.columns:
    fig = plt.figure()
    x = data_no_nan(columns)
    ax = fig.add_subplot(111)
    ax.hist(x)
    plt.title(columns)
    plt.show()
    if columns!='email_address':
        newset=dataset[columns].astype(float)
        newsetmax=newset.idxmax()
        print columns,newsetmax
#https://www.jianshu.com/p/2c02a7b0b382
#https://blog.csdn.net/u014365862/article/details/51815562


# In[43]:


#删除
dataset.drop(['THE TRAVEL AGENCY IN THE PARK','TOTAL','LOCKHART EUGENE E'],inplace=True)


# In[109]:


##Feature processing
#把nan处理

def column_with_npnan(column):
    data = []
    for value in column:
        if value == 'NaN':
            value = np.nan
        data.append(value)
    return np.array(data)
dataset_np = dataset.apply(column_with_npnan)


# In[46]:


from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=1)


# In[59]:


feature_list0 = ['bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'expenses', 
                'exercised_stock_options', 'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', 
                'restricted_stock_deferred', 'salary', 'total_payments', 'total_stock_value',
                'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
                 'shared_receipt_with_poi']


# In[48]:


feature_imp = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], 
                      [0], [0], [0], [0], [0], [0], [0], [0], [0]]


# In[49]:


for i in range(len(feature_list0)):
    element = feature_list0[i]
    imp.fit([dataset_np[element]])
    feature_imp[i] = imp.transform([dataset_np[element]])
    feature_imp[i] = feature_imp[i][0]


# In[60]:


#http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_imp_scaled = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], 
                      [0], [0], [0], [0], [0], [0], [0], [0], [0]]
feature_imp_scaled= scaler.fit_transform(feature_imp)
dataset_scaled = pd.DataFrame(feature_imp_scaled)
dataset_scaled.index = feature_list0


# In[97]:


dataset_scaled = dataset_scaled.T
dataset_scaled


# In[81]:


dataset_without_p_e=dataset.drop(['poi','email_address'],axis=1)

Tdataset=dataset_without_p_e
print Tdataset.index
Tdataset_scaled=dataset_scaled
Tdataset_scaled.index= Tdataset.index
print Tdataset_scaled


# In[ ]:


#把poi和email加上df


# In[87]:


dataset_scaled=Tdataset_scaled
poi=dataset['poi']
email_address = dataset['email_address']
newdataset_scaled =pd.concat([dataset_scaled, poi, email_address], axis=1)
newdataset_scaled.head()


# In[114]:


dataset=dataset.T
my_dataset=dataset.to_dict()


# In[116]:


features_list = ['poi','salary', 'bonus', 'exercised_stock_options', 'deferred_income','from_messages',
                  'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi','expenses', 'long_term_incentive', 'restricted_stock']

data00 = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data00)

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)
print 'accuracy', score

importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(11):
    print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])


# In[117]:


from tester import test_classifier,dump_classifier_and_data,load_classifier_and_data,main


# In[118]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import neighbors

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

clf01 =  AdaBoostClassifier()
clf02 =  RandomForestClassifier(min_samples_split=50)
clf03 =  GaussianNB()

clf04 =  neighbors.KNeighborsClassifier()
clf05 =  QuadraticDiscriminantAnalysis()



# In[119]:


#features_list01 = ['poi','salary', 'bonus', 'exercised_stock_options', 'deferred_income', 'from_messages','expenses', 'long_term_incentive',
#                  'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']
features_list01 = ['poi','salary', 'bonus', 'exercised_stock_options', 'deferred_income', 'from_messages','expenses', 'long_term_incentive',
                  'from_poi_to_this_person']


# In[120]:


from sklearn.cross_validation import train_test_split
data01 = featureFormat(my_dataset, features_list01, sort_keys = True)
labels01, features01 = targetFeatureSplit(data01)
features_train01, features_test01, labels_train01, labels_test01 =     train_test_split(features01, labels01, test_size=0.3, random_state=42)
    


# In[121]:


dump_classifier_and_data(clf01, my_dataset, features_list01)
load_classifier_and_data()
print features_list01
if __name__ == '__main__':
    main()


# In[122]:


dump_classifier_and_data(clf02, my_dataset, features_list01)
load_classifier_and_data()
print features_list01
if __name__ == '__main__':
    main()


# In[123]:


dump_classifier_and_data(clf03, my_dataset, features_list01)
load_classifier_and_data()
print features_list01
if __name__ == '__main__':
    main()


# In[124]:


dump_classifier_and_data(clf04, my_dataset, features_list01)
load_classifier_and_data()
print features_list01
if __name__ == '__main__':
    main()


# In[125]:


dump_classifier_and_data(clf05, my_dataset, features_list01)
load_classifier_and_data()
print features_list01
if __name__ == '__main__':
    main()


# In[127]:


#01可留
features_list02=['poi', 'salary', 'bonus', 'exercised_stock_options', 'deferred_income', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 


# In[128]:


from sklearn.cross_validation import train_test_split
data02 = featureFormat(my_dataset, features_list02, sort_keys = True)
labels02, features02 = targetFeatureSplit(data02)
features_train02, features_test02, labels_train02, labels_test02 =     train_test_split(features02, labels02, test_size=0.3, random_state=42)


# In[129]:


dump_classifier_and_data(clf01, my_dataset, features_list02)
load_classifier_and_data()
print features_list02
if __name__ == '__main__':
    main()


# In[130]:


dump_classifier_and_data(clf02, my_dataset, features_list02)
load_classifier_and_data()
print features_list02
if __name__ == '__main__':
    main()


# In[131]:


dump_classifier_and_data(clf03, my_dataset, features_list02)
load_classifier_and_data()
print features_list02
if __name__ == '__main__':
    main()


# In[132]:


dump_classifier_and_data(clf04, my_dataset, features_list02)
load_classifier_and_data()
print features_list02
if __name__ == '__main__':
    main()


# In[133]:


dump_classifier_and_data(clf05, my_dataset, features_list02)
load_classifier_and_data()
print features_list02
if __name__ == '__main__':
    main()


# In[ ]:


#总结，04 Accuracy: 0.81043	Precision: 0.33248	Recall: 0.32450	F1: 0.32844	F2: 0.32607
	Total predictions: 14000	True positives:  649	False positives: 1303	False negatives: 1351	True negatives: 10697


# In[ ]:


#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


# In[165]:


#用缩放后的数据
#newdataset_scaled
newdataset_scaled=newdataset_scaled.T
my_dataset2=newdataset_scaled.to_dict()


# In[183]:


from sklearn.cross_validation import train_test_split
data03 = featureFormat(my_dataset2, features_list02, sort_keys = True)
labels03, features03 = targetFeatureSplit(data03)
features_train03, features_test03, labels_train03, labels_test03 =     train_test_split(features03, labels03, test_size=0.3, random_state=42)
    


# In[184]:


dump_classifier_and_data(clf02, my_dataset2, features_list02)
load_classifier_and_data()
print features_list02
if __name__ == '__main__':
    main()
    
dump_classifier_and_data(clf03, my_dataset2, features_list02)
load_classifier_and_data()
print features_list02
if __name__ == '__main__':
    main()
    
dump_classifier_and_data(clf04, my_dataset2, features_list02)
load_classifier_and_data()
print features_list02
if __name__ == '__main__':
    main()
    
dump_classifier_and_data(clf05, my_dataset2, features_list02)
load_classifier_and_data()
print features_list02
if __name__ == '__main__':
    main()


# In[ ]:


#缩放后调参


# In[185]:



data000 = featureFormat(my_dataset2, features_list02, sort_keys = True)
labels000, features000 = targetFeatureSplit(data000)
features_train000, features_test000, labels_train000, labels_test000 =     train_test_split(features000, labels000, test_size=0.3, random_state=42)

#开始调优使用GridSearchCV找到,最优参数
knn = KNeighborsClassifier()
#设置k的范围
k_range = list(range(1,31))
leaf_range = list(range(1,2))
weight_options = ['uniform','distance']
algorithm_options = ['auto','ball_tree','kd_tree','brute']
param_gridknn = dict(n_neighbors = k_range,weights = weight_options,algorithm=algorithm_options,leaf_size=leaf_range)
gridKNN = GridSearchCV(knn,param_gridknn,cv=10,scoring='accuracy',verbose=1)
gridKNN.fit(features_train000,labels_train000)
print('best score is:',str(gridKNN.best_score_))
print('best params are:',str(gridKNN.best_params_))
#https://blog.csdn.net/szj_huhu/article/details/74909773


# In[186]:


data000 = featureFormat(my_dataset2, features_list02, sort_keys = True)
labels000, features000 = targetFeatureSplit(data000)
features_train000, features_test000, labels_train000, labels_test000 =     train_test_split(features000, labels000, test_size=0.3, random_state=42)
clf07 =  KNeighborsClassifier(n_neighbors=2, weights='uniform',leaf_size=1,algorithm='auto')
dump_classifier_and_data(clf07, my_dataset2, features_list02)
load_classifier_and_data()
print features_list02
if __name__ == '__main__':
    main()


# In[177]:


#未缩放特征的调参


# In[194]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
data001 = featureFormat(my_dataset, features_list02, sort_keys = True)
labels001, features001 = targetFeatureSplit(data001)
features_train001, features_test001, labels_train001, labels_test001 =     train_test_split(features001, labels001, test_size=0.3, random_state=42)

#开始调优使用GridSearchCV找到,最优参数
knn = KNeighborsClassifier()
#设置k的范围
k_range = list(range(1,31))
leaf_range = list(range(1,2))
weight_options = ['uniform','distance']
algorithm_options = ['auto','ball_tree','kd_tree','brute']
param_gridknn = dict(n_neighbors = k_range,weights = weight_options,algorithm=algorithm_options,leaf_size=leaf_range)
gridKNN = GridSearchCV(knn,param_gridknn,cv=10,scoring='accuracy',verbose=1)
gridKNN.fit(features_train001,labels_train001)
#print('best score is:',str(gridKNN.best_score_))
print('best params are:',str(gridKNN.best_params_))


# In[193]:



clf08 =  KNeighborsClassifier(n_neighbors=3, weights='uniform',leaf_size=1,algorithm='auto')
dump_classifier_and_data(clf08, my_dataset, features_list02)
load_classifier_and_data()
print features_list02
if __name__ == '__main__':
    main()


# In[ ]:


#新特征


# In[208]:


newdataset_scaled=newdataset_scaled.T
newdataset_scaled


# In[202]:


dataset=dataset.T


# In[203]:


dataset['coefficient_bonus_salary'] = 0.0


# In[204]:


dataset


# In[209]:


for i in range(len(newdataset_scaled['salary'])):
    if newdataset_scaled['salary'][i] > 0:
        dataset['coefficient_bonus_salary'][i] =         1.0 * newdataset_scaled['bonus'][i] / newdataset_scaled['salary'][i]


# In[210]:


dataset


# In[216]:


datasetT=dataset.T
my_dataset3=datasetT.to_dict()


# In[218]:


features_list03=['poi', 'salary', 'bonus', 'exercised_stock_options', 'deferred_income', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi','coefficient_bonus_salary'] 
data04 = featureFormat(my_dataset3, features_list03, sort_keys = True)
labels04, features04 = targetFeatureSplit(data04)
features_train04, features_test04, labels_train04, labels_test03 =     train_test_split(features04, labels04, test_size=0.3, random_state=42)


# In[219]:



dump_classifier_and_data(clf01, my_dataset3, features_list03)
load_classifier_and_data()
print features_list03
if __name__ == '__main__':
    main()
    
dump_classifier_and_data(clf02, my_dataset3, features_list03)
load_classifier_and_data()
print features_list03
if __name__ == '__main__':
    main()
    
dump_classifier_and_data(clf03, my_dataset3, features_list03)
load_classifier_and_data()
print features_list03
if __name__ == '__main__':
    main()
    
dump_classifier_and_data(clf04, my_dataset3, features_list03)
load_classifier_and_data()
print features_list03
if __name__ == '__main__':
    main()
    
dump_classifier_and_data(clf05, my_dataset3, features_list03)
load_classifier_and_data()
print features_list03
if __name__ == '__main__':
    main()

