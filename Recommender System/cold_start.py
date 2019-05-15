#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 23:07:21 2018

@author: chunyangjia
"""

import pandas as pd
import numpy as np

names = ['uid', 'aid','weight'];
df = pd.read_csv('user_artists.dat',sep = '\s+',names =names);
df2 = pd.read_csv('artists.dat',sep='\t', names = ['aid', 'a_name','url','pictureURL']);

from collections import defaultdict 
aid_list = defaultdict(list);
aid_list = df.aid.unique().tolist();

column1 = ['aid','total_people']
column2 = ['aid','total_count']
df_cold_p = pd.DataFrame(columns = column1)
df_cold_c = pd.DataFrame(columns = column2)

for aid in aid_list:
    aid_weight = df.loc[df['aid'] == aid, ['aid','weight']];
    total_weight = aid_weight['weight'].count();
    total_count = aid_weight['weight'].sum();
    df_cold_p = df_cold_p.append({'aid':aid, 'total_people':total_weight}, ignore_index=True);
    df_cold_c = df_cold_c.append({'aid':aid, 'total_count':total_count}, ignore_index=True);

df_cold_p['total_people'] = pd.to_numeric(df_cold_p['total_people'])
df_top20_p = df_cold_p.nlargest(10, 'total_people')
print(df_top20_p)

df_cold_c['total_count'] = pd.to_numeric(df_cold_c['total_count'])
df_top20_c = df_cold_c.nlargest(10, 'total_count')
print(df_top20_c)


#pick the random NO of artists by people listened
columns = ['aid','a_name','url','pictureURL']
df_top20_names_p = pd.DataFrame(columns = columns)
aid_list = (df_top20_p.sample(n = 10)).aid.unique().tolist();
for aid in aid_list:
    aid_names = df2.loc[df2['aid']==aid, ['aid','a_name','url','pictureURL']]
    df_top20_names_p = df_top20_names_p.append(aid_names, ignore_index = True)
df_p = df_top20_names_p;
df_p = df_p.merge(df_top20_p, on = 'aid', how = 'inner')
print(df_top20_names_p)

#pick the random NO of artists by total hits
columns = ['aid','a_name','url','pictureURL']
df_top20_names_c = pd.DataFrame(columns = columns)
aid_list = (df_top20_c.sample(n = 10)).aid.unique().tolist();
for aid in aid_list:
    aid_names = df2.loc[df2['aid']==aid, ['aid','a_name','url','pictureURL']]
    df_top20_names_c = df_top20_names_c.append(aid_names, ignore_index = True)
df_c = df_top20_names_c;
df_c = df_c.merge(df_top20_c, on = 'aid', how = 'inner')
print(df_top20_names_c)

cold_names_p = df_p['a_name']
cold_names_c = df_c['a_name']
cold_names = cold_names_p.append(cold_names_c).unique().tolist()
cold_names = np.array(cold_names, dtype = pd.Series)
idx = np.random.choice(np.arange(len(cold_names)), 10, replace=False)
cold_start = cold_names[idx]
print(cold_names)


import matplotlib.pyplot as plt
#draw by people listened
x = df_p['aid']
y = df_p['total_people']

plt.style.use('ggplot')
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, y, color='green')
plt.xlabel("Top 10 Singer ID")
plt.ylabel("NO. of people listened")
plt.title("Hottest 10 singers by NO. of people listened")
plt.xticks(x_pos, x)
plt.show()

#draw by total hits
x = df_c['aid']
y = df_c['total_count']

plt.style.use('ggplot')
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, y, color='green')
plt.xlabel("Top 10 Singer ID")
plt.ylabel("Total listening counts")
plt.title("Hottest 10 singers by total listening count")
plt.xticks(x_pos, x)
plt.show()

#find out user 7 listening history
uid = 7
df_7 = df.loc[ df['uid'] == uid, ['aid','uid','weight']]
df_7_like = df_7.nlargest(13,'weight');
aid_7 = df_7_like.aid.unique().tolist()

columns = ['aid','a_name','url']
df_7_names = pd.DataFrame(columns = columns)

#df_2_names are the artists user 7 liked.
for aid in aid_7:
    aid_weight = df2.loc[df2['aid'] == aid,['aid','a_name','url','pictureURL']]
    df_7_names = df_7_names.append(aid_weight)

df_7_aid = df_7_names.a_name.unique().tolist()
print(df_7_aid)

























