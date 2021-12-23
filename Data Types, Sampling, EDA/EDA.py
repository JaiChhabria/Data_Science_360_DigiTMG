# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 17:06:09 2021

@author: jaich
"""

import pandas as pd

Q1 = pd.read_csv('Q1.csv')

Q1['Speed'].mean()
Q1['Speed'].median()
Q1['Speed'].mode()
Q1['Speed'].var()
Q1['Speed'].std()
Q1['Speed'].kurt()
Q1['Speed'].skew()


Q1['dist'].mean()
Q1['dist'].median()
Q1['dist'].mode()
Q1['dist'].var()
Q1['dist'].std()
Q1['dist'].kurt()
Q1['dist'].skew()


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=Q1)



Q2 = pd.read_csv('Q2.csv')

Q2['SP'].mean()
Q2['SP'].median()
Q2['SP'].mode()
Q2['SP'].var()
Q2['SP'].std()
Q2['SP'].kurt()
Q2['SP'].skew()


Q2['WT'].mean()
Q2['WT'].median()
Q2['WT'].mode()
Q2['WT'].var()
Q2['WT'].std()
Q2['WT'].kurt()
Q2['WT'].skew()

sns.boxplot(data=Q2)
