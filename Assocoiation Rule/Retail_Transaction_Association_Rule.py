# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 01:23:11 2021

@author: jaich
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#data = pd.read_csv('transactions_retail1.csv')

retail = []
with open('transactions_retail1.csv') as f:
    retail = f.read()
    
    
retail = retail.split('\n')

retail_list = []
for i in retail:
    retail_list.append(i.split(","))
    
all_retail_list = [i for item in retail_list for i in item]

from collections import Counter

retail_frequencies = Counter(all_retail_list)


retail_frequencies = sorted(retail_frequencies.items(), key = lambda x:x[1])

frequencies = list(reversed([i[1] for i in retail_frequencies]))
items = list(reversed([i[0] for i in retail_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbkymc')
plt.xticks(list(range(0, 11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


retail_series = pd.DataFrame(pd.Series(retail_list))

retail_series.columns = ["transactions"]

X = retail_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')


frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

################################# Extra part ###################################
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)