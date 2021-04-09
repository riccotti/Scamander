#Load data
import pandas as pd
import numpy as np

data_dir='./../../dataset/'
frame=pd.read_csv(data_dir+"fico.csv")
train=np.arange(frame.shape[0])%10!=0
test=np.bitwise_not(train)
target=np.asarray(frame['RiskPerformance']=='Good')
inputs=frame[frame.columns[1:]]

#Create the object and initialise it
from soc.counterfactual import linear_explanation, plot_coefficients


exp=linear_explanation()
exp.encode_pandas(inputs)
#train the logistic regression classifier
exp.train_logistic(target)
#if only a subset of the data is to be used for training use
#the following line instead
#exp.train_logistic(target[train],train)

i =409 #Explain 410th test entry
text=exp.explain_entry(inputs.iloc[i].values,10)
#The explanation is an ordered list of text strings
#print them
for t in text:
    print(t)

text=exp.explain_entry_counterfactuals(inputs.iloc[i].values,10)
#The explanation is an ordered list of text strings
#print them
for t in text:
    print(t)

#generate a full set of explanations for all data
explain_all=False
if explain_all:
    explanations=exp.explain_set(inputs.values,12)
    exp2=pd.DataFrame(explanations)
    exp2.to_csv('test.csv')

#visualise the set of linear weights ordered by their median contribution over the dataset.
# plot_coefficients(exp,med_weight=True)
