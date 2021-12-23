# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 00:34:09 2021

@author: jaich
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Lets load our dataset

data = pd.read_csv('my_movies.csv')


# Interesting dataset, first 5 columns are list of movies and next 5 columns are dummy

# lets check for null values in the data

sns.heatmap(data.isnull())


# Since data already has dummy variables, we will drop the columns which V1... V5

columns_to_drop = ['V1','V2','V3','V4','V5']
df = data.drop(columns=columns_to_drop)

# Lets Create Association Rules

from mlxtend.frequent_patterns import apriori, association_rules

liked_movies = apriori(df,min_support=0.05, use_colnames=False)

liked_movies.sort_values('support', ascending=False, inplace=True)

plt.bar(x = list(range(0,30)), height = liked_movies.support[0:30], color = 'rgmyk')
plt.xticks(list(range(0,30)), liked_movies.itemsets[0:30], rotation = 15)
plt.ylabel('item-sets')
plt.xlabel('support')
plt.show()


plt.bar(x = list(range(0,7)), height = liked_movies.support[0:7], color = 'rgmyk')
plt.xticks(list(range(0,7)), liked_movies.itemsets[0:7], rotation = 15)
plt.ylabel('item-sets')
plt.xlabel('support')
plt.show()

rules = association_rules(liked_movies, metric = 'lift', min_threshold=2)
rules1 = association_rules(liked_movies, metric = 'confidence', min_threshold=1)


def to_list(i):
    return(sorted(list(i)))

ma_x = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
ma_x = ma_x.apply(sorted)
rules_set = list(ma_x)
unique_rules_set = [list(m) for m in set(tuple(i)for i in rules_set)]

index_rules = []

for i in unique_rules_set:
    index_rules.append(rules_set.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)




ma_x1 = rules1.antecedents.apply(to_list) + rules1.consequents.apply(to_list)
ma_x1 = ma_x1.apply(sorted)
rules_set1 = list(ma_x1)
unique_rules_set1 = [list(m) for m in set(tuple(i)for i in rules_set1)]

index_rules1 = []

for i in unique_rules_set1:
    index_rules1.append(rules_set1.index(i))

# getting rules without any redudancy 
rules_no_redudancy1 = rules1.iloc[index_rules1, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy1.sort_values('lift', ascending = False).head(10)

