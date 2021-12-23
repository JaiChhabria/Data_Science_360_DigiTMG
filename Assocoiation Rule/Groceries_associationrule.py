# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:50:59 2021

@author: jaich
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# Lets get our data
groceries = []
with open('groceries.csv') as f:
    groceries = f.read()

# Splitting the data into seperate transactions using seperator \n

groceries = groceries.split('\n')

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    
all_groceries_list = [i for item in groceries_list for i in item]


from collections import Counter

item_frequencies = Counter(all_groceries_list)

item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbykcm')
plt.xticks(list(range(0, 11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

groceries_series = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835, :]


groceries_series.columns = ["transactions"]

X = groceries_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.0050, max_len = 4, use_colnames = True)
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 30)), height = frequent_itemsets.support[0:30], color ='rgmyk')
plt.xticks(list(range(0, 30)), frequent_itemsets.itemsets[0:30], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()


rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)


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