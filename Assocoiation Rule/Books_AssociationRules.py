# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:09:10 2021

@author: jaich
"""

# Association Rule for Books Datasets
# Lets impoty necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Business Objective

# We need store to regain its popularity by increasing foot fall, we will look at the sales of books type and create associations between each type.
# Once we have done that then we will advertise the best associations

data = pd.read_csv('book.csv')
data.head()

sns.heatmap(data.isnull()) # There are no missing values in the dataset

data.describe()
data.plot(kind='kde')
# Lets look for correlation in our data

sns.heatmap(data.corr(), annot=True, cmap='coolwarm')

# Lets make Association Rules
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter


frequent_itemsets = apriori(data,min_support=0.008,max_len=4,use_colnames=False)
frequent_itemsets.sort_values('support', ascending=False,inplace=True)


plt.bar(x = list(range(7,20)), height = frequent_itemsets.support[7:20], color ='rgmyk')
plt.xticks(list(range(7, 20)), frequent_itemsets.itemsets[7:20], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)


def to_list(i):
    return(sorted(list(i)))

ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)

unqiue_rules_sets = [list(m) for m in set(tuple(i)for i in rules_sets)]

index_rules=[]

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)






