#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[43]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as ss
from scipy import stats
import scipy
from sklearn import metrics
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import cohen_kappa_score
import mne


# In[28]:


np.set_printoptions(precision=2)
time=(((np.arange(1, 1320.5, 1, dtype=int)-1)/1200)*1000)-100


# # Loading .mat data

# In[29]:


animate_data_dic = loadmat('animate_data.mat')
animate_data=animate_data_dic['animate_data']

inanimate_data_dic = loadmat('inanimate_data.mat')
inanimate_data=inanimate_data_dic['inanimate_data']

face_data_dic = loadmat('face_data.mat')
face_data=face_data_dic['face_data']

body_data_dic = loadmat('body_data.mat')
body_data=body_data_dic['body_data']

male_data_dic = loadmat('male_data.mat')
male_data=male_data_dic['male_data']

female_data_dic = loadmat('female_data.mat')
female_data=female_data_dic['female_data']


# In[30]:


channel_num=len(animate_data[1][1][:])
data_num=len(animate_data[1][:][:])
animate_num=len(animate_data[:])
inanimate_num=len(inanimate_data[:])
face_num=len(face_data[:])
body_num=len(body_data[:])
male_num=len(male_data[:])
female_num=len(female_data[:])


# In[31]:


female_num


# # All channels

# ## ERP

# In[32]:


animate_mean_trial=np.mean(animate_data,axis=0)
animate_ERP=np.mean(animate_mean_trial,axis=1)

inanimate_mean_trial=np.mean(inanimate_data,axis=0)
inanimate_ERP=np.mean(inanimate_mean_trial,axis=1)

face_mean_trial=np.mean(face_data,axis=0)
face_ERP=np.mean(face_mean_trial,axis=1)

body_mean_trial=np.mean(body_data,axis=0)
body_ERP=np.mean(body_mean_trial,axis=1)

male_mean_trial=np.mean(male_data,axis=0)
male_ERP=np.mean(male_mean_trial,axis=1)

female_mean_trial=np.mean(female_data,axis=0)
female_ERP=np.mean(female_mean_trial,axis=1)


# In[33]:


fig, (ax1,ax2,ax3) = plt.subplots(3, 1)
fig.set_figheight(20)
fig.set_figwidth(12)

ax1.plot(time, animate_ERP, color="c", label = "animate")
ax1.plot(time, inanimate_ERP, color="orange", label = "inanimate")
ax1.plot(time,np.zeros(data_num),color="black")
ax1.set(xlabel='time(ms)',ylabel='ERP')
ax1.set_title("ERP of all channels animate Vs inanimate", size=15)
ax1.legend()


ax2.plot(time, face_ERP, color="red", label = "face")
ax2.plot(time, body_ERP, color="blue", label = "body")
ax2.plot(time,np.zeros(data_num),color="black")
ax2.set(xlabel='time(ms)',ylabel='ERP')
ax2.set_title("ERP of all channels face Vs body", size=15)
ax2.legend()

ax3.plot(time, male_ERP, color="pink", label = "male")
ax3.plot(time, female_ERP, color="purple", label = "female")
ax3.plot(time,np.zeros(data_num),color="black")
ax3.set(xlabel='time(ms)',ylabel='ERP')
ax3.set_title("ERP of all channels male Vs female", size=15)
ax3.legend()


# ## smoothing data

# In[34]:


animate_data_smooth=np.empty([animate_num, data_num,channel_num])
inanimate_data_smooth=np.empty([inanimate_num, data_num,channel_num])
face_data_smooth=np.empty([face_num, data_num,channel_num])
body_data_smooth=np.empty([body_num, data_num,channel_num])
male_data_smooth=np.empty([male_num, data_num,channel_num])
female_data_smooth=np.empty([female_num, data_num,channel_num])


# In[35]:


for i in range(animate_num):
    for j in range(data_num-60):
        animate_data_smooth[i][j]=np.mean(animate_data[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        animate_data_smooth[i][k]=np.mean(animate_data[i][k:], axis=0)
    
for i in range(inanimate_num):
    for j in range(data_num-60):
        inanimate_data_smooth[i][j]=np.mean(inanimate_data[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        inanimate_data_smooth[i][k]=np.mean(inanimate_data[i][k:], axis=0)
    
for i in range(face_num):
    for j in range(data_num-60):
        face_data_smooth[i][j]=np.mean(face_data[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        face_data_smooth[i][k]=np.mean(face_data[i][k:], axis=0)
    
for i in range(body_num):
    for j in range(data_num-60):
        body_data_smooth[i][j]=np.mean(body_data[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        body_data_smooth[i][k]=np.mean(body_data[i][k:], axis=0)
    
for i in range(male_num):
    for j in range(data_num-60):
        male_data_smooth[i][j]=np.mean(male_data[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        male_data_smooth[i][k]=np.mean(male_data[i][k:], axis=0)
    
for i in range(female_num):
    for j in range(data_num-60):
        female_data_smooth[i][j]=np.mean(female_data[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        female_data_smooth[i][k]=np.mean(female_data[i][k:], axis=0)


# ## Creating data frames

# ### animat =1 inanimate=2  face=3 body=4 male=5 female=6

# In[37]:


animate_df_smooth = pd.DataFrame(columns=['data', 'label'])
inanimate_df_smooth = pd.DataFrame(columns=['data', 'label'])

face_df_smooth = pd.DataFrame(columns=['data', 'label'])
body_df_smooth = pd.DataFrame(columns=['data', 'label'])

male_df_smooth = pd.DataFrame(columns=['data', 'label'])
female_df_smooth = pd.DataFrame(columns=['data', 'label'])


# In[38]:


for i in range(animate_num):
    new_row={'data':animate_data_smooth[i][:][:], 'label':1}
    animate_df_smooth = animate_df_smooth.append(new_row, ignore_index=True)
    
for i in range(inanimate_num):
    new_row={'data':inanimate_data_smooth[i][:][:], 'label':2}
    inanimate_df_smooth =inanimate_df_smooth.append(new_row, ignore_index=True)
    
for i in range(face_num):
    new_row={'data':face_data_smooth[i][:][:], 'label':3}
    face_df_smooth = face_df_smooth.append(new_row, ignore_index=True)
    
for i in range(body_num):
    new_row={'data':body_data_smooth[i][:][:], 'label':4}
    body_df_smooth = body_df_smooth.append(new_row, ignore_index=True)
    
for i in range(male_num):
    new_row={'data':male_data_smooth[i][:][:], 'label':5}
    male_df_smooth = male_df_smooth.append(new_row, ignore_index=True)
    
for i in range(female_num):
    new_row={'data':female_data_smooth[i][:][:], 'label':6}
    female_df_smooth= female_df_smooth.append(new_row, ignore_index=True)


# In[42]:


female_df_smooth.data[0]


# ## Averaging each 4 trials

# In[40]:


def averaging_trials():
    ave_num=4
    animate_shuffle_df=animate_df_smooth.sample(frac = 1)
    inanimate_shuffle_df=inanimate_df_smooth.sample(frac = 1)
    face_shuffle_df=face_df_smooth.sample(frac = 1)
    body_shuffle_df=body_df_smooth.sample(frac = 1)
    male_shuffle_df=male_df_smooth.sample(frac = 1)
    female_shuffle_df=female_df_smooth.sample(frac = 1)
    
    animate_inanimate_df_average = pd.DataFrame(columns=['data', 'label'])
    face_body_df_average = pd.DataFrame(columns=['data', 'label'])
    male_female_df_average = pd.DataFrame(columns=['data', 'label'])
    
    for i in range(int(animate_num/ave_num)):
        new_row={'data':animate_shuffle_df.data[i*ave_num:i*ave_num+ave_num].mean(), 'label':1}
        animate_inanimate_df_average = animate_inanimate_df_average.append(new_row, ignore_index=True)
    if animate_num%ave_num !=0:
        tmp_animate=int(animate_num/ave_num)
        new_row={'data':animate_shuffle_df.data[tmp_animate*ave_num:].mean(), 'label':1}
        animate_inanimate_df_average = animate_inanimate_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(inanimate_num/ave_num)):
        new_row={'data':inanimate_shuffle_df.data[i*ave_num:i*ave_num+ave_num].mean(), 'label':2}
        animate_inanimate_df_average = animate_inanimate_df_average.append(new_row, ignore_index=True)
    if inanimate_num%ave_num !=0:
        tmp_inanimate=int(inanimate_num/ave_num)
        new_row={'data':inanimate_shuffle_df.data[tmp_inanimate*ave_num:].mean(), 'label':2}
        animate_inanimate_df_average = animate_inanimate_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(face_num/ave_num)):
        new_row={'data':face_shuffle_df.data[i*ave_num:i*ave_num+ave_num].mean(), 'label':3}
        face_body_df_average = face_body_df_average.append(new_row, ignore_index=True)
    if face_num%ave_num !=0:
        tmp_face=int(face_num/ave_num)
        new_row={'data':face_shuffle_df.data[tmp_face*ave_num:].mean(), 'label':3}
        face_body_df_average = face_body_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(body_num/ave_num)):
        new_row={'data':body_shuffle_df.data[i*ave_num:i*ave_num+ave_num].mean(), 'label':4}
        face_body_df_average = face_body_df_average.append(new_row, ignore_index=True)
    if body_num%ave_num !=0:
        tmp_body=int(body_num/ave_num)
        new_row={'data':body_shuffle_df.data[tmp_body*ave_num:].mean(), 'label':4}
        face_body_df_average = face_body_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(male_num/ave_num)):
        new_row={'data':male_shuffle_df.data[i*ave_num:i*ave_num+ave_num].mean(), 'label':5}
        male_female_df_average = male_female_df_average.append(new_row, ignore_index=True)
    if male_num%ave_num !=0:
        tmp_male=int(male_num/ave_num)
        new_row={'data':male_shuffle_df.data[tmp_male*ave_num:].mean(), 'label':5}
        male_female_df_average = male_female_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(female_num/ave_num)):
        new_row={'data':female_shuffle_df.data[i*ave_num:i*ave_num+ave_num].mean(), 'label':6}
        male_female_df_average = male_female_df_average.append(new_row, ignore_index=True)
    if female_num%ave_num !=0:
        tmp_female=int(female_num/ave_num)
        new_row={'data':female_shuffle_df.data[tmp_female*ave_num:].mean(), 'label':6}
        male_female_df_average = male_female_df_average.append(new_row, ignore_index=True)
    
    
    return animate_inanimate_df_average,face_body_df_average,male_female_df_average


# ## Z-score Normalization

# In[ ]:


def z_score(data):
    data_zscored_df=pd.DataFrame(columns=['data', 'label'])
    for i in range(len(data)):
        data_tmp=
        for j in range(data_num)
    return 


# ## Training the classifier

# In[39]:


acc_animate_inanimate_all=[]
kappa_animate_inanimate_all=[]
acc_face_body_all=[]
kappa_face_body_all=[]
acc_male_female_all=[]
kappa_male_female_all=[]

for j in range(100):
    print(j)
    # averaging each 4 trials
    animate_inanimate_df_average,face_body_df_average,male_female_df_average=averaging_trials()
    
    # z_score normalization
    
    # train test split
    train_animate_inanimate, test_animate_inanimate = train_test_split(animate_inanimate_df_average, test_size=0.25, shuffle=True)
    train_face_body, test_face_body = train_test_split(face_body_df_average, test_size=0.25, shuffle=True)
    train_male_female, test_male_female = train_test_split(male_female_df_average, test_size=0.25, shuffle=True)
    
    # animate vs inanimate
    acc_animate_inanimate=[]
    kappa_animate_inanimate=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_animate_inanimate.iterrows():
            train.append(row.data[i])
        for index, row in test_animate_inanimate.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_animate_inanimate.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_animate_inanimate.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_animate_inanimate.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_animate_inanimate.append(kappa_tmp)
    acc_animate_inanimate_all.append(acc_animate_inanimate)
    kappa_animate_inanimate_all.append(kappa_animate_inanimate)
    
    # face vs body
    acc_face_body=[]
    kappa_face_body=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_face_body.iterrows():
            train.append(row.data[i])
        for index, row in test_face_body.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_face_body.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_face_body.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_face_body.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_face_body.append(kappa_tmp)
    acc_face_body_all.append(acc_face_body)
    kappa_face_body_all.append(kappa_face_body)
    
    # male vs female
    acc_male_female=[]
    kappa_male_female=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_male_female.iterrows():
            train.append(row.data[i])
        for index, row in test_male_female.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_male_female.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_male_female.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_male_female.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_male_female.append(kappa_tmp)
    acc_male_female_all.append(acc_male_female)
    kappa_male_female_all.append(kappa_male_female)
        


# In[57]:


acc_animate_inanimate_mean_avg4=np.mean(np.array(acc_animate_inanimate_all_avg4), axis=0)
kappa_animate_inanimate_mean_avg4= np.mean(np.array(kappa_animate_inanimate_all_avg4), axis=0)
acc_face_body_mean_avg4= np.mean(np.array(acc_face_body_all_avg4), axis=0)
kappa_face_body_mean_avg4= np.mean(np.array(kappa_face_body_all_avg4), axis=0) 
acc_male_female_mean_avg4= np.mean(np.array(acc_male_female_all_avg4), axis=0) 
kappa_male_female_mean_avg4= np.nanmean(np.array(kappa_male_female_all_avg4), axis=0) 


# In[58]:


# saving
with open('acc_animate_inanimate_mean_sub6_avg4.npy', 'wb') as f:
    np.save(f, acc_animate_inanimate_mean_avg4)
    
with open('kappa_animate_inanimate_mean_sub6_avg4.npy', 'wb') as f:
    np.save(f, kappa_animate_inanimate_mean_avg4)
    
with open('acc_face_body_mean_sub6_avg4.npy', 'wb') as f:
    np.save(f, acc_face_body_mean_avg4)
    
with open('kappa_face_body_mean_sub6_avg4.npy', 'wb') as f:
    np.save(f, kappa_face_body_mean_avg4)
    
with open('acc_male_female_mean_sub6_avg4.npy', 'wb') as f:
    np.save(f, acc_male_female_mean_avg4)
    
with open('kappa_male_female_mean_sub6_avg4.npy', 'wb') as f:
    np.save(f, kappa_male_female_mean_avg4)


# In[7]:


# uploading
with open('acc_animate_inanimate_mean_sub6_avg4.npy', 'rb') as f:
    acc_animate_inanimate_mean_avg4 = np.load(f)
    
with open('kappa_animate_inanimate_mean_sub6_avg4.npy', 'rb') as f:
    kappa_animate_inanimate_mean_avg4 = np.load(f)
    
    
with open('acc_face_body_mean_sub6_avg4.npy', 'rb') as f:
    acc_face_body_mean_avg4 = np.load(f)
    
with open('kappa_face_body_mean_sub6_avg4.npy', 'rb') as f:
    kappa_face_body_mean_avg4 = np.load(f)
    
with open('acc_male_female_mean_sub6_avg4.npy', 'rb') as f:
    acc_male_female_mean_avg4 = np.load(f)
    
with open('kappa_male_female_mean_sub6_avg4.npy', 'rb') as f:
    kappa_male_female_mean_avg4 = np.load(f)


# ## Visualizing before smoothing with gaussian kernel

# In[8]:


plt.figure(figsize=(12,5))
plt.plot(time, kappa_animate_inanimate_mean_avg4, color="c", label = "animate_inanimate")
plt.plot(time, kappa_face_body_mean_avg4, color="red", label = "face_body")
plt.plot(time, kappa_male_female_mean_avg4, color="green", label = "male_female")

plt.title("Kappa in time")
plt.xlabel("time(ms)")
plt.ylabel("Kappa")
plt.legend()
plt.show()


# In[9]:


plt.figure(figsize=(12,6))
plt.plot(time, acc_animate_inanimate_mean_avg4, color="c", label = "animate_inanimate")
plt.plot(time, acc_face_body_mean_avg4, color="red", label = "face_body")
plt.plot(time, acc_male_female_mean_avg4, color="green", label = "male_female")

plt.title("Accuracy in time")
plt.xlabel("time(ms)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ## Smoothing with gaussian kernel

# In[10]:


x = np.arange(0, data_num)
sigma = 5
smoothed_acc_animate_inanimate_avg4 = np.zeros(acc_animate_inanimate_mean_avg4.shape)
smoothed_kappa_animate_inanimate_avg4 = np.zeros(kappa_animate_inanimate_mean_avg4.shape)
smoothed_acc_face_body_avg4 = np.zeros(acc_face_body_mean_avg4.shape)
smoothed_kappa_face_body_avg4 = np.zeros(kappa_face_body_mean_avg4.shape)
smoothed_acc_male_female_avg4 = np.zeros(acc_male_female_mean_avg4.shape)
smoothed_kappa_male_female_avg4 = np.zeros(kappa_male_female_mean_avg4.shape)

for x_position in x:
    kernel = np.exp(-((x - x_position) ** 2) / (2 * sigma**2))
    kernel = kernel / sum(kernel)
    smoothed_acc_animate_inanimate_avg4[x_position] = sum(acc_animate_inanimate_mean_avg4 * kernel)
    smoothed_kappa_animate_inanimate_avg4[x_position] = sum(kappa_animate_inanimate_mean_avg4* kernel)
    smoothed_acc_face_body_avg4[x_position] = sum(acc_face_body_mean_avg4 * kernel)
    smoothed_kappa_face_body_avg4[x_position] = sum(kappa_face_body_mean_avg4 * kernel)
    smoothed_acc_male_female_avg4[x_position] = sum(acc_male_female_mean_avg4 * kernel)
    smoothed_kappa_male_female_avg4[x_position] = sum(kappa_male_female_mean_avg4 * kernel)
    
    


# ## Visualize data Vs smoothed

# In[13]:


fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(12,15))

axes[0][0].plot(time, acc_animate_inanimate_mean_avg4, color="blue", label = "animate_inanimate")
axes[0][0].plot(time, smoothed_acc_animate_inanimate_avg4, color="red", label = " smooth animate_inanimate")
axes[0][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[0][0].set_title("Accuracy animate_inanimate Vs smoothed", size=10)
axes[0][0].legend()

axes[0][1].plot(time, kappa_animate_inanimate_mean_avg4, color="blue", label = "animate_inanimate")
axes[0][1].plot(time, smoothed_kappa_animate_inanimate_avg4, color="red", label = " smooth animate_inanimate")
axes[0][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[0][1].set_title("Kappa animate_inanimate Vs smoothed", size=10)
axes[0][1].legend()

axes[1][0].plot(time, acc_face_body_mean_avg4, color="blue", label = "face_body")
axes[1][0].plot(time, smoothed_acc_face_body_avg4, color="red", label = " smooth face_body")
axes[1][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[1][0].set_title("Accuracy face_body Vs smoothed", size=10)
axes[1][0].legend()

axes[1][1].plot(time, kappa_face_body_mean_avg4, color="blue", label = "face_body")
axes[1][1].plot(time, smoothed_kappa_face_body_avg4, color="red", label = " smooth face_body")
axes[1][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[1][1].set_title("Kappa face_body Vs smoothed", size=10)
axes[1][1].legend()

axes[2][0].plot(time, acc_male_female_mean_avg4, color="blue", label = "male_female")
axes[2][0].plot(time, smoothed_acc_male_female_avg4, color="red", label = " smooth male_female")
axes[2][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[2][0].set_title("Accuracy male_female Vs smoothed", size=10)
axes[2][0].legend()

axes[2][1].plot(time, kappa_male_female_mean_avg4, color="blue", label = "male_female")
axes[2][1].plot(time, smoothed_kappa_male_female_avg4, color="red", label = " smooth male_female")
axes[2][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[2][1].set_title("Kappa male_female Vs smoothed", size=10)
axes[2][1].legend()



# ## Visualizing smoothed data

# In[14]:


plt.figure(figsize=(12,5))
plt.plot(time, smoothed_kappa_animate_inanimate_avg4, color="c", label = "animate_inanimate")
plt.plot(time, smoothed_kappa_face_body_avg4, color="red", label = "face_body")
plt.plot(time, smoothed_kappa_male_female_avg4, color="green", label = "male_female")

plt.title("Smoothed Kappa in time")
plt.xlabel("time(ms)")
plt.ylabel("Kappa")
plt.legend()
plt.show()


# In[15]:


plt.figure(figsize=(12,5))
plt.plot(time, smoothed_acc_animate_inanimate_avg4, color="c", label = "animate_inanimate")
plt.plot(time, smoothed_acc_face_body_avg4, color="red", label = "face_body")
plt.plot(time, smoothed_acc_male_female_avg4, color="green", label = "male_female")

plt.title("Smoothed Accuracy in time averaging each 4 trials")
plt.xlabel("time(ms)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ## Onset Time

# In[ ]:


def find_onset_time(data):
    max_index=np.argmax(data)
    max_value=data.max()
    th=max_value*0.1
    for i in range(max_index):
        is_onset=true
        for j in range(i,i+13):
            if 
    


# In[26]:


# animate inanimate
animate_inanimate_all_tmp=smoothed_kappa_animate_inanimate_avg4-smoothed_kappa_animate_inanimate_avg4.min()
max_index_animate_inanimate_all=np.argmax(animate_inanimate_all_tmp)
max_value_animate_inanimate_all=animate_inanimate_all_tmp.max()
onset_index_animate_inanimate_all=(np.abs(animate_inanimate_all_tmp[:max_index_animate_inanimate_all] - max_value_animate_inanimate_all*0.1)).argmin()
onset_time_animate_inanimate_all=time[onset_index_animate_inanimate_all]
# face body
face_body_all_tmp=smoothed_kappa_face_body_avg4-smoothed_kappa_face_body_avg4.min()
max_index_animate_face_body_all=np.argmax(face_body_all_tmp)
max_value_face_body_all=face_body_all_tmp.max()
onset_index_face_body_all=(np.abs(face_body_all_tmp[:max_index_face_body_all] - max_value_face_body_all*0.1)).argmin()
onset_time_face_body_all=time[onset_index_face_body_all]

# male female
print(onset_time_animate_inanimate_all)
print(onset_time_face_body_all)


# In[22]:


max_value_animate_inanimate_all*0.1


# # Frontal channels

# ## seperating frontal data

# In[66]:


fc=np.array([1,2,3,4,5,6,7,10,11,12,13,14,15,16,61,62,63,64,65,66,67,68,69,70,71,72])-1


# In[67]:


animate_data_fc=np.zeros((animate_num,data_num,len(fc)))
inanimate_data_fc=np.zeros((inanimate_num,data_num,len(fc)))
face_data_fc=np.zeros((face_num,data_num,len(fc)))
body_data_fc=np.zeros((body_num,data_num,len(fc)))
male_data_fc=np.zeros((male_num,data_num,len(fc)))
female_data_fc=np.zeros((female_num,data_num,len(fc)))

for j in range(data_num):
    for i in range(animate_num):
        animate_data_fc[i][j][:]=animate_data[i][j][fc]
    for i in range(inanimate_num):
        inanimate_data_fc[i][j][:]=inanimate_data[i][j][fc]
    for i in range(face_num):
        face_data_fc[i][j][:]=face_data[i][j][fc]
    for i in range(body_num):
        body_data_fc[i][j][:]=body_data[i][j][fc]
    for i in range(male_num):
        male_data_fc[i][j][:]=male_data[i][j][fc]
    for i in range(female_num):
        female_data_fc[i][j][:]=female_data[i][j][fc]



# ## ERP

# In[68]:


animate_fc_mean_trial=np.mean(animate_data_fc,axis=0)
animate_fc_ERP=np.mean(animate_fc_mean_trial,axis=1)

inanimate_fc_mean_trial=np.mean(inanimate_data_fc,axis=0)
inanimate_fc_ERP=np.mean(inanimate_fc_mean_trial,axis=1)

face_fc_mean_trial=np.mean(face_data_fc,axis=0)
face_fc_ERP=np.mean(face_fc_mean_trial,axis=1)

body_fc_mean_trial=np.mean(body_data_fc,axis=0)
body_fc_ERP=np.mean(body_fc_mean_trial,axis=1)

male_fc_mean_trial=np.mean(male_data_fc,axis=0)
male_fc_ERP=np.mean(male_fc_mean_trial,axis=1)

female_fc_mean_trial=np.mean(female_data_fc,axis=0)
female_fc_ERP=np.mean(female_fc_mean_trial,axis=1)


# In[69]:


fig, (ax1,ax2,ax3) = plt.subplots(3, 1)
fig.set_figheight(20)
fig.set_figwidth(12)

ax1.plot(time, animate_fc_ERP, color="c", label = "animate")
ax1.plot(time, inanimate_fc_ERP, color="orange", label = "inanimate")
ax1.plot(time,np.zeros(data_num),color="black")
ax1.set(xlabel='time(ms)',ylabel='ERP')
ax1.set_title("ERP of frontal channels animate Vs inanimate", size=15)
ax1.legend()


ax2.plot(time, face_fc_ERP, color="red", label = "face")
ax2.plot(time, body_fc_ERP, color="blue", label = "body")
ax2.plot(time,np.zeros(data_num),color="black")
ax2.set(xlabel='time(ms)',ylabel='ERP')
ax2.set_title("ERP of frontal channels face Vs body", size=15)
ax2.legend()

ax3.plot(time, male_fc_ERP, color="pink", label = "male")
ax3.plot(time, female_fc_ERP, color="purple", label = "female")
ax3.plot(time,np.zeros(data_num),color="black")
ax3.set(xlabel='time(ms)',ylabel='ERP')
ax3.set_title("ERP of frontal channels male Vs female", size=15)
ax3.legend()


# ## smoothing data

# In[70]:


animate_data_fc_smooth=np.empty([animate_num, data_num,len(fc)])
inanimate_data_fc_smooth=np.empty([inanimate_num, data_num,len(fc)])
face_data_fc_smooth=np.empty([face_num, data_num,len(fc)])
body_data_fc_smooth=np.empty([body_num, data_num,len(fc)])
male_data_fc_smooth=np.empty([male_num, data_num,len(fc)])
female_data_fc_smooth=np.empty([female_num, data_num,len(fc)])


# In[71]:


for i in range(animate_num):
    for j in range(data_num-60):
        animate_data_fc_smooth[i][j]=np.mean(animate_data_fc[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        animate_data_fc_smooth[i][k]=np.mean(animate_data_fc[i][k:], axis=0)
    
for i in range(inanimate_num):
    for j in range(data_num-60):
        inanimate_data_fc_smooth[i][j]=np.mean(inanimate_data_fc[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        inanimate_data_fc_smooth[i][k]=np.mean(inanimate_data_fc[i][k:], axis=0)
    
for i in range(face_num):
    for j in range(data_num-60):
        face_data_fc_smooth[i][j]=np.mean(face_data_fc[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        face_data_fc_smooth[i][k]=np.mean(face_data_fc[i][k:], axis=0)
    
for i in range(body_num):
    for j in range(data_num-60):
        body_data_fc_smooth[i][j]=np.mean(body_data_fc[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        body_data_fc_smooth[i][k]=np.mean(body_data_fc[i][k:], axis=0)
    
for i in range(male_num):
    for j in range(data_num-60):
        male_data_fc_smooth[i][j]=np.mean(male_data_fc[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        male_data_fc_smooth[i][k]=np.mean(male_data_fc[i][k:], axis=0)
    
for i in range(female_num):
    for j in range(data_num-60):
        female_data_fc_smooth[i][j]=np.mean(female_data_fc[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        female_data_fc_smooth[i][k]=np.mean(female_data_fc[i][k:], axis=0)


# ## Creating data frames

# ### animat =1 inanimate=2  face=3 body=4 male=5 female=6

# In[72]:


animate_fc_df_smooth = pd.DataFrame(columns=['data', 'label'])
inanimate_fc_df_smooth = pd.DataFrame(columns=['data', 'label'])

face_fc_df_smooth = pd.DataFrame(columns=['data', 'label'])
body_fc_df_smooth = pd.DataFrame(columns=['data', 'label'])

male_fc_df_smooth = pd.DataFrame(columns=['data', 'label'])
female_fc_df_smooth = pd.DataFrame(columns=['data', 'label'])


# In[73]:


for i in range(animate_num):
    new_row={'data':animate_data_fc_smooth[i][:][:], 'label':1}
    animate_fc_df_smooth = animate_fc_df_smooth.append(new_row, ignore_index=True)
    
for i in range(inanimate_num):
    new_row={'data':inanimate_data_fc_smooth[i][:][:], 'label':2}
    inanimate_fc_df_smooth =inanimate_fc_df_smooth.append(new_row, ignore_index=True)
    
for i in range(face_num):
    new_row={'data':face_data_fc_smooth[i][:][:], 'label':3}
    face_fc_df_smooth = face_fc_df_smooth.append(new_row, ignore_index=True)
    
for i in range(body_num):
    new_row={'data':body_data_fc_smooth[i][:][:], 'label':4}
    body_fc_df_smooth = body_fc_df_smooth.append(new_row, ignore_index=True)
    
for i in range(male_num):
    new_row={'data':male_data_fc_smooth[i][:][:], 'label':5}
    male_fc_df_smooth = male_fc_df_smooth.append(new_row, ignore_index=True)
    
for i in range(female_num):
    new_row={'data':female_data_fc_smooth[i][:][:], 'label':6}
    female_fc_df_smooth= female_fc_df_smooth.append(new_row, ignore_index=True)


# In[75]:


female_fc_df_smooth


# ## Averaging each 4 trials randomly

# In[76]:


def averaging_trials_fc():
    ave_num=4
    animate_fc_shuffle_df=animate_fc_df_smooth.sample(frac = 1)
    inanimate_fc_shuffle_df=inanimate_fc_df_smooth.sample(frac = 1)
    face_fc_shuffle_df=face_fc_df_smooth.sample(frac = 1)
    body_fc_shuffle_df=body_fc_df_smooth.sample(frac = 1)
    male_fc_shuffle_df=male_fc_df_smooth.sample(frac = 1)
    female_fc_shuffle_df=female_fc_df_smooth.sample(frac = 1)
    
    animate_inanimate_fc_df_average = pd.DataFrame(columns=['data', 'label'])
    face_body_fc_df_average = pd.DataFrame(columns=['data', 'label'])
    male_female_fc_df_average = pd.DataFrame(columns=['data', 'label'])
    
    for i in range(int(animate_num/ave_num)):
        new_row={'data':animate_fc_shuffle_df.data[i*4:i*4+4].mean(), 'label':1}
        animate_inanimate_fc_df_average = animate_inanimate_fc_df_average.append(new_row, ignore_index=True)
    if animate_num%ave_num !=0:
        tmp_animate=int(animate_num/ave_num)
        new_row={'data':animate_fc_shuffle_df.data[tmp_animate*4:].mean(), 'label':1}
        animate_inanimate_fc_df_average = animate_inanimate_fc_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(inanimate_num/ave_num)):
        new_row={'data':inanimate_fc_shuffle_df.data[i*4:i*4+4].mean(), 'label':2}
        animate_inanimate_fc_df_average = animate_inanimate_fc_df_average.append(new_row, ignore_index=True)
    if inanimate_num%ave_num !=0:
        tmp_inanimate=int(inanimate_num/ave_num)
        new_row={'data':inanimate_fc_shuffle_df.data[tmp_inanimate*4:].mean(), 'label':2}
        animate_inanimate_fc_df_average = animate_inanimate_fc_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(face_num/ave_num)):
        new_row={'data':face_fc_shuffle_df.data[i*4:i*4+4].mean(), 'label':3}
        face_body_fc_df_average = face_body_fc_df_average.append(new_row, ignore_index=True)
    if face_num%ave_num !=0:
        tmp_face=int(face_num/ave_num)
        new_row={'data':face_fc_shuffle_df.data[tmp_face*4:].mean(), 'label':3}
        face_body_fc_df_average = face_body_fc_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(body_num/ave_num)):
        new_row={'data':body_fc_shuffle_df.data[i*4:i*4+4].mean(), 'label':4}
        face_body_fc_df_average = face_body_fc_df_average.append(new_row, ignore_index=True)
    if body_num%ave_num !=0:
        tmp_body=int(body_num/ave_num)
        new_row={'data':body_fc_shuffle_df.data[tmp_body*4:].mean(), 'label':4}
        face_body_fc_df_average = face_body_fc_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(male_num/ave_num)):
        new_row={'data':male_fc_shuffle_df.data[i*4:i*4+4].mean(), 'label':5}
        male_female_fc_df_average = male_female_fc_df_average.append(new_row, ignore_index=True)
    if male_num%ave_num !=0:
        tmp_male=int(male_num/ave_num)
        new_row={'data':male_fc_shuffle_df.data[tmp_male*4:].mean(), 'label':5}
        male_female_fc_df_average = male_female_fc_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(female_num/ave_num)):
        new_row={'data':female_fc_shuffle_df.data[i*4:i*4+4].mean(), 'label':6}
        male_female_fc_df_average = male_female_fc_df_average.append(new_row, ignore_index=True)
    if female_num%ave_num !=0:
        tmp_female=int(female_num/ave_num)
        new_row={'data':female_fc_shuffle_df.data[tmp_female*4:].mean(), 'label':6}
        male_female_fc_df_average = male_female_fc_df_average.append(new_row, ignore_index=True)
    
    
    return animate_inanimate_fc_df_average,face_body_fc_df_average,male_female_fc_df_average


# ## Training the classifier

# In[77]:


acc_animate_inanimate_fc_all=[]
kappa_animate_inanimate_fc_all=[]
acc_face_body_fc_all=[]
kappa_face_body_fc_all=[]
acc_male_female_fc_all=[]
kappa_male_female_fc_all=[]

for j in range(100):
    print(j)
    # averaging each 4 trials
    animate_inanimate_fc_df_average,face_body_fc_df_average,male_female_fc_df_average=averaging_trials_fc()
    
    # train test split
    train_animate_inanimate_fc, test_animate_inanimate_fc = train_test_split(animate_inanimate_fc_df_average, test_size=0.25, shuffle=True)
    train_face_body_fc, test_face_body_fc = train_test_split(face_body_fc_df_average, test_size=0.25, shuffle=True)
    train_male_female_fc, test_male_female_fc = train_test_split(male_female_fc_df_average, test_size=0.25, shuffle=True)
    
    # animate vs inanimate
    acc_animate_inanimate_fc=[]
    kappa_animate_inanimate_fc=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_animate_inanimate_fc.iterrows():
            train.append(row.data[i])
        for index, row in test_animate_inanimate_fc.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_animate_inanimate_fc.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_animate_inanimate_fc.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_animate_inanimate_fc.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_animate_inanimate_fc.append(kappa_tmp)
    acc_animate_inanimate_fc_all.append(acc_animate_inanimate_fc)
    kappa_animate_inanimate_fc_all.append(kappa_animate_inanimate_fc)
    
    # face vs body
    acc_face_body_fc=[]
    kappa_face_body_fc=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_face_body_fc.iterrows():
            train.append(row.data[i])
        for index, row in test_face_body_fc.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_face_body_fc.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_face_body_fc.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_face_body_fc.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_face_body_fc.append(kappa_tmp)
    acc_face_body_fc_all.append(acc_face_body_fc)
    kappa_face_body_fc_all.append(kappa_face_body_fc)
    
    # male vs female
    acc_male_female_fc=[]
    kappa_male_female_fc=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_male_female_fc.iterrows():
            train.append(row.data[i])
        for index, row in test_male_female_fc.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_male_female_fc.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_male_female_fc.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_male_female_fc.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_male_female_fc.append(kappa_tmp)
    acc_male_female_fc_all.append(acc_male_female_fc)
    kappa_male_female_fc_all.append(kappa_male_female_fc)
        


# In[78]:


acc_animate_inanimate_fc_mean=np.nanmean(np.array(acc_animate_inanimate_fc_all), axis=0)
kappa_animate_inanimate_fc_mean= np.nanmean(np.array(kappa_animate_inanimate_fc_all), axis=0)
acc_face_body_fc_mean= np.nanmean(np.array(acc_face_body_fc_all), axis=0)
kappa_face_body_fc_mean= np.nanmean(np.array(kappa_face_body_fc_all), axis=0) 
acc_male_female_fc_mean= np.nanmean(np.array(acc_male_female_fc_all), axis=0) 
kappa_male_female_fc_mean= np.nanmean(np.array(kappa_male_female_fc_all), axis=0) 


# In[79]:


# saving
with open('acc_animate_inanimate_fc_mean_sub6.npy', 'wb') as f:
    np.save(f, acc_animate_inanimate_fc_mean)
    
with open('kappa_animate_inanimate_fc_mean_sub6.npy', 'wb') as f:
    np.save(f, kappa_animate_inanimate_fc_mean)
    
with open('acc_face_body_fc_mean_sub6.npy', 'wb') as f:
    np.save(f, acc_face_body_fc_mean)
    
with open('kappa_face_body_fc_mean_sub6.npy', 'wb') as f:
    np.save(f, kappa_face_body_fc_mean)
    
with open('acc_male_female_fc_mean_sub6.npy', 'wb') as f:
    np.save(f, acc_male_female_fc_mean)
    
with open('kappa_male_female_fc_mean_sub6.npy', 'wb') as f:
    np.save(f, kappa_male_female_fc_mean)


# In[80]:


# uploading
with open('acc_animate_inanimate_fc_mean_sub6.npy', 'rb') as f:
    acc_animate_inanimate_fc_mean = np.load(f)
    
with open('kappa_animate_inanimate_fc_mean_sub6.npy', 'rb') as f:
    kappa_animate_inanimate_fc_mean = np.load(f)
    
    
with open('acc_face_body_fc_mean_sub6.npy', 'rb') as f:
    acc_face_body_fc_mean = np.load(f)
    
with open('kappa_face_body_fc_mean_sub6.npy', 'rb') as f:
    kappa_face_body_fc_mean = np.load(f)
    
with open('acc_male_female_fc_mean_sub6.npy', 'rb') as f:
    acc_male_female_fc_mean = np.load(f)
    
with open('kappa_male_female_fc_mean_sub6.npy', 'rb') as f:
    kappa_male_female_fc_mean = np.load(f)


# ## visualization before smoothing with gaussian kernel

# In[81]:


plt.figure(figsize=(12,5))
plt.plot(time, kappa_animate_inanimate_fc_mean, color="c", label = "animate_inanimate")
plt.plot(time, kappa_face_body_fc_mean, color="red", label = "face_body")
plt.plot(time, kappa_male_female_fc_mean, color="green", label = "male_female")

plt.title("Kappa in time for frontal channels")
plt.xlabel("time(ms)")
plt.ylabel("Kappa")
plt.legend()
plt.show()


# In[82]:


plt.figure(figsize=(12,6))
plt.plot(time, acc_animate_inanimate_fc_mean, color="c", label = "animate_inanimate")
plt.plot(time, acc_face_body_fc_mean, color="red", label = "face_body")
plt.plot(time, acc_male_female_fc_mean, color="green", label = "male_female")

plt.title("Accuracy in time for frontal channels")
plt.xlabel("time(ms)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ## smoothing with gaussian kernel

# In[83]:


x = np.arange(0, data_num)
sigma = 5
smoothed_acc_animate_inanimate_fc = np.zeros(acc_animate_inanimate_fc_mean.shape)
smoothed_kappa_animate_inanimate_fc = np.zeros(kappa_animate_inanimate_fc_mean.shape)
smoothed_acc_face_body_fc = np.zeros(acc_face_body_fc_mean.shape)
smoothed_kappa_face_body_fc = np.zeros(kappa_face_body_fc_mean.shape)
smoothed_acc_male_female_fc = np.zeros(acc_male_female_fc_mean.shape)
smoothed_kappa_male_female_fc = np.zeros(kappa_male_female_fc_mean.shape)

for x_position in x:
    kernel = np.exp(-((x - x_position) ** 2) / (2 * sigma**2))
    kernel = kernel / sum(kernel)
    smoothed_acc_animate_inanimate_fc[x_position] = sum(acc_animate_inanimate_fc_mean * kernel)
    smoothed_kappa_animate_inanimate_fc[x_position] = sum(kappa_animate_inanimate_fc_mean * kernel)
    smoothed_acc_face_body_fc[x_position] = sum(acc_face_body_fc_mean * kernel)
    smoothed_kappa_face_body_fc[x_position] = sum(kappa_face_body_fc_mean * kernel)
    smoothed_acc_male_female_fc[x_position] = sum(acc_male_female_fc_mean * kernel)
    smoothed_kappa_male_female_fc[x_position] = sum(kappa_male_female_fc_mean * kernel)


# ## visualising data Vs smoothed

# In[84]:


fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(12,15))

axes[0][0].plot(time, acc_animate_inanimate_fc_mean, color="blue", label = "animate_inanimate")
axes[0][0].plot(time, smoothed_acc_animate_inanimate_fc, color="red", label = " smooth animate_inanimate")
axes[0][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[0][0].set_title("Accuracy animate_inanimate Vs smoothed in frontal channels", size=10)
axes[0][0].legend()

axes[0][1].plot(time, kappa_animate_inanimate_fc_mean, color="blue", label = "animate_inanimate")
axes[0][1].plot(time, smoothed_kappa_animate_inanimate_fc, color="red", label = " smooth animate_inanimate")
axes[0][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[0][1].set_title("Kappa animate_inanimate Vs smoothed in frontal channels", size=10)
axes[0][1].legend()

axes[1][0].plot(time, acc_face_body_fc_mean, color="blue", label = "face_body")
axes[1][0].plot(time, smoothed_acc_face_body_fc, color="red", label = " smooth face_body")
axes[1][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[1][0].set_title("Accuracy face_body Vs smoothed in frontal channels", size=10)
axes[1][0].legend()

axes[1][1].plot(time, kappa_face_body_fc_mean, color="blue", label = "face_body")
axes[1][1].plot(time, smoothed_kappa_face_body_fc, color="red", label = " smooth face_body")
axes[1][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[1][1].set_title("Kappa face_body Vs smoothed in frontal channels", size=10)
axes[1][1].legend()

axes[2][0].plot(time, acc_male_female_fc_mean, color="blue", label = "male_female")
axes[2][0].plot(time, smoothed_acc_male_female_fc, color="red", label = " smooth male_female")
axes[2][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[2][0].set_title("Accuracy male_female Vs smoothed in frontal channels", size=10)
axes[2][0].legend()

axes[2][1].plot(time, kappa_male_female_fc_mean, color="blue", label = "male_female")
axes[2][1].plot(time, smoothed_kappa_male_female_fc, color="red", label = " smooth male_female")
axes[2][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[2][1].set_title("Kappa male_female Vs smoothed in frontal channels", size=10)
axes[2][1].legend()


# ## visualizing smoothed data

# In[85]:


plt.figure(figsize=(12,5))
plt.plot(time, smoothed_kappa_animate_inanimate_fc, color="c", label = "animate_inanimate")
plt.plot(time, smoothed_kappa_face_body_fc, color="red", label = "face_body")
plt.plot(time, smoothed_kappa_male_female_fc, color="green", label = "male_female")

plt.title("Smoothed Kappa in time in frontal channels")
plt.xlabel("time(ms)")
plt.ylabel("Kappa")
plt.legend()
plt.show()


# In[86]:


plt.figure(figsize=(12,5))
plt.plot(time, smoothed_acc_animate_inanimate_fc, color="c", label = "animate_inanimate")
plt.plot(time, smoothed_acc_face_body_fc, color="red", label = "face_body")
plt.plot(time, smoothed_acc_male_female_fc, color="green", label = "male_female")

plt.title("Smoothed Accuracy in time in frontal channels")
plt.xlabel("time(ms)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# # Temporal channels

# ## seperating temporal data

# In[87]:


tc=np.array([8,9,18,19,20,27,28,36,37,44,45,73,74,75,83,84,91,92,99,100,17,24,25,26,34,35,43,51,52,80,81,82,89,90,97,98,105,106])-1


# In[88]:


animate_data_tc=np.zeros((animate_num,data_num,len(tc)))
inanimate_data_tc=np.zeros((inanimate_num,data_num,len(tc)))
face_data_tc=np.zeros((face_num,data_num,len(tc)))
body_data_tc=np.zeros((body_num,data_num,len(tc)))
male_data_tc=np.zeros((male_num,data_num,len(tc)))
female_data_tc=np.zeros((female_num,data_num,len(tc)))

for j in range(data_num):
    for i in range(animate_num):
        animate_data_tc[i][j][:]=animate_data[i][j][tc]
    for i in range(inanimate_num):
        inanimate_data_tc[i][j][:]=inanimate_data[i][j][tc]
    for i in range(face_num):
        face_data_tc[i][j][:]=face_data[i][j][tc]
    for i in range(body_num):
        body_data_tc[i][j][:]=body_data[i][j][tc]
    for i in range(male_num):
        male_data_tc[i][j][:]=male_data[i][j][tc]
    for i in range(female_num):
        female_data_tc[i][j][:]=female_data[i][j][tc]



# ## ERP

# In[89]:


animate_tc_mean_trial=np.mean(animate_data_tc,axis=0)
animate_tc_ERP=np.mean(animate_tc_mean_trial,axis=1)

inanimate_tc_mean_trial=np.mean(inanimate_data_tc,axis=0)
inanimate_tc_ERP=np.mean(inanimate_tc_mean_trial,axis=1)

face_tc_mean_trial=np.mean(face_data_tc,axis=0)
face_tc_ERP=np.mean(face_tc_mean_trial,axis=1)

body_tc_mean_trial=np.mean(body_data_tc,axis=0)
body_tc_ERP=np.mean(body_tc_mean_trial,axis=1)

male_tc_mean_trial=np.mean(male_data_tc,axis=0)
male_tc_ERP=np.mean(male_tc_mean_trial,axis=1)

female_tc_mean_trial=np.mean(female_data_tc,axis=0)
female_tc_ERP=np.mean(female_tc_mean_trial,axis=1)


# In[90]:


fig, (ax1,ax2,ax3) = plt.subplots(3, 1)
fig.set_figheight(20)
fig.set_figwidth(12)

ax1.plot(time, animate_tc_ERP, color="c", label = "animate")
ax1.plot(time, inanimate_tc_ERP, color="orange", label = "inanimate")
ax1.plot(time,np.zeros(data_num),color="black")
ax1.set(xlabel='time(ms)',ylabel='ERP')
ax1.set_title("ERP of temporal channels animate Vs inanimate", size=15)
ax1.legend()


ax2.plot(time, face_tc_ERP, color="red", label = "face")
ax2.plot(time, body_tc_ERP, color="blue", label = "body")
ax2.plot(time,np.zeros(data_num),color="black")
ax2.set(xlabel='time(ms)',ylabel='ERP')
ax2.set_title("ERP of temporal channels face Vs body", size=15)
ax2.legend()

ax3.plot(time, male_tc_ERP, color="pink", label = "male")
ax3.plot(time, female_tc_ERP, color="purple", label = "female")
ax3.plot(time,np.zeros(data_num),color="black")
ax3.set(xlabel='time(ms)',ylabel='ERP')
ax3.set_title("ERP of temporal channels male Vs female", size=15)
ax3.legend()


# ## smoothing data

# In[91]:


animate_data_tc_smooth=np.empty([animate_num, data_num,len(tc)])
inanimate_data_tc_smooth=np.empty([inanimate_num, data_num,len(tc)])
face_data_tc_smooth=np.empty([face_num, data_num,len(tc)])
body_data_tc_smooth=np.empty([body_num, data_num,len(tc)])
male_data_tc_smooth=np.empty([male_num, data_num,len(tc)])
female_data_tc_smooth=np.empty([female_num, data_num,len(tc)])


# In[92]:


for i in range(animate_num):
    for j in range(data_num-60):
        animate_data_tc_smooth[i][j]=np.mean(animate_data_tc[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        animate_data_tc_smooth[i][k]=np.mean(animate_data_tc[i][k:], axis=0)
    
for i in range(inanimate_num):
    for j in range(data_num-60):
        inanimate_data_tc_smooth[i][j]=np.mean(inanimate_data_tc[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        inanimate_data_tc_smooth[i][k]=np.mean(inanimate_data_tc[i][k:], axis=0)
    
for i in range(face_num):
    for j in range(data_num-60):
        face_data_tc_smooth[i][j]=np.mean(face_data_tc[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        face_data_tc_smooth[i][k]=np.mean(face_data_tc[i][k:], axis=0)
    
for i in range(body_num):
    for j in range(data_num-60):
        body_data_tc_smooth[i][j]=np.mean(body_data_tc[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        body_data_tc_smooth[i][k]=np.mean(body_data_tc[i][k:], axis=0)
    
for i in range(male_num):
    for j in range(data_num-60):
        male_data_tc_smooth[i][j]=np.mean(male_data_tc[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        male_data_tc_smooth[i][k]=np.mean(male_data_tc[i][k:], axis=0)
    
for i in range(female_num):
    for j in range(data_num-60):
        female_data_tc_smooth[i][j]=np.mean(female_data_tc[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        female_data_tc_smooth[i][k]=np.mean(female_data_tc[i][k:], axis=0)


# ## Creating data frames

# ### animat =1 inanimate=2  face=3 body=4 male=5 female=6

# In[93]:


animate_tc_df_smooth = pd.DataFrame(columns=['data', 'label'])
inanimate_tc_df_smooth = pd.DataFrame(columns=['data', 'label'])

face_tc_df_smooth = pd.DataFrame(columns=['data', 'label'])
body_tc_df_smooth = pd.DataFrame(columns=['data', 'label'])

male_tc_df_smooth = pd.DataFrame(columns=['data', 'label'])
female_tc_df_smooth = pd.DataFrame(columns=['data', 'label'])


# In[94]:


for i in range(animate_num):
    new_row={'data':animate_data_tc_smooth[i][:][:], 'label':1}
    animate_tc_df_smooth = animate_tc_df_smooth.append(new_row, ignore_index=True)
    
for i in range(inanimate_num):
    new_row={'data':inanimate_data_tc_smooth[i][:][:], 'label':2}
    inanimate_tc_df_smooth =inanimate_tc_df_smooth.append(new_row, ignore_index=True)
    
for i in range(face_num):
    new_row={'data':face_data_tc_smooth[i][:][:], 'label':3}
    face_tc_df_smooth = face_tc_df_smooth.append(new_row, ignore_index=True)
    
for i in range(body_num):
    new_row={'data':body_data_tc_smooth[i][:][:], 'label':4}
    body_tc_df_smooth = body_tc_df_smooth.append(new_row, ignore_index=True)
    
for i in range(male_num):
    new_row={'data':male_data_tc_smooth[i][:][:], 'label':5}
    male_tc_df_smooth = male_tc_df_smooth.append(new_row, ignore_index=True)
    
for i in range(female_num):
    new_row={'data':female_data_tc_smooth[i][:][:], 'label':6}
    female_tc_df_smooth= female_tc_df_smooth.append(new_row, ignore_index=True)


# In[106]:


female_tc_df_smooth


# ## Averaging each 4 trials randomly

# In[96]:


def averaging_trials_tc():
    ave_num=4
    animate_tc_shuffle_df=animate_tc_df_smooth.sample(frac = 1)
    inanimate_tc_shuffle_df=inanimate_tc_df_smooth.sample(frac = 1)
    face_tc_shuffle_df=face_tc_df_smooth.sample(frac = 1)
    body_tc_shuffle_df=body_tc_df_smooth.sample(frac = 1)
    male_tc_shuffle_df=male_tc_df_smooth.sample(frac = 1)
    female_tc_shuffle_df=female_tc_df_smooth.sample(frac = 1)
    
    animate_inanimate_tc_df_average = pd.DataFrame(columns=['data', 'label'])
    face_body_tc_df_average = pd.DataFrame(columns=['data', 'label'])
    male_female_tc_df_average = pd.DataFrame(columns=['data', 'label'])
    
    for i in range(int(animate_num/ave_num)):
        new_row={'data':animate_tc_shuffle_df.data[i*4:i*4+4].mean(), 'label':1}
        animate_inanimate_tc_df_average = animate_inanimate_tc_df_average.append(new_row, ignore_index=True)
    if animate_num%ave_num !=0:
        tmp_animate=int(animate_num/ave_num)
        new_row={'data':animate_tc_shuffle_df.data[tmp_animate*4:].mean(), 'label':1}
        animate_inanimate_tc_df_average = animate_inanimate_tc_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(inanimate_num/ave_num)):
        new_row={'data':inanimate_tc_shuffle_df.data[i*4:i*4+4].mean(), 'label':2}
        animate_inanimate_tc_df_average = animate_inanimate_tc_df_average.append(new_row, ignore_index=True)
    if inanimate_num%ave_num !=0:
        tmp_inanimate=int(inanimate_num/ave_num)
        new_row={'data':inanimate_tc_shuffle_df.data[tmp_inanimate*4:].mean(), 'label':2}
        animate_inanimate_tc_df_average = animate_inanimate_tc_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(face_num/ave_num)):
        new_row={'data':face_tc_shuffle_df.data[i*4:i*4+4].mean(), 'label':3}
        face_body_tc_df_average = face_body_tc_df_average.append(new_row, ignore_index=True)
    if face_num%ave_num !=0:
        tmp_face=int(face_num/ave_num)
        new_row={'data':face_tc_shuffle_df.data[tmp_face*4:].mean(), 'label':3}
        face_body_tc_df_average = face_body_tc_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(body_num/ave_num)):
        new_row={'data':body_tc_shuffle_df.data[i*4:i*4+4].mean(), 'label':4}
        face_body_tc_df_average = face_body_tc_df_average.append(new_row, ignore_index=True)
    if body_num%ave_num !=0:
        tmp_body=int(body_num/ave_num)
        new_row={'data':body_tc_shuffle_df.data[tmp_body*4:].mean(), 'label':4}
        face_body_tc_df_average = face_body_tc_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(male_num/ave_num)):
        new_row={'data':male_tc_shuffle_df.data[i*4:i*4+4].mean(), 'label':5}
        male_female_tc_df_average = male_female_tc_df_average.append(new_row, ignore_index=True)
    if male_num%ave_num !=0:
        tmp_male=int(male_num/ave_num)
        new_row={'data':male_tc_shuffle_df.data[tmp_male*4:].mean(), 'label':5}
        male_female_tc_df_average = male_female_tc_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(female_num/ave_num)):
        new_row={'data':female_tc_shuffle_df.data[i*4:i*4+4].mean(), 'label':6}
        male_female_tc_df_average = male_female_tc_df_average.append(new_row, ignore_index=True)
    if female_num%ave_num !=0:
        tmp_female=int(female_num/ave_num)
        new_row={'data':female_tc_shuffle_df.data[tmp_female*4:].mean(), 'label':6}
        male_female_tc_df_average = male_female_tc_df_average.append(new_row, ignore_index=True)
    
    
    return animate_inanimate_tc_df_average,face_body_tc_df_average,male_female_tc_df_average


# ## Training the classifier

# In[97]:


acc_animate_inanimate_tc_all=[]
kappa_animate_inanimate_tc_all=[]
acc_face_body_tc_all=[]
kappa_face_body_tc_all=[]
acc_male_female_tc_all=[]
kappa_male_female_tc_all=[]

for j in range(100):
    print(j)
    # averaging each 4 trials
    animate_inanimate_tc_df_average,face_body_tc_df_average,male_female_tc_df_average=averaging_trials_tc()
    
    # train test split
    train_animate_inanimate_tc, test_animate_inanimate_tc = train_test_split(animate_inanimate_tc_df_average, test_size=0.25, shuffle=True)
    train_face_body_tc, test_face_body_tc = train_test_split(face_body_tc_df_average, test_size=0.25, shuffle=True)
    train_male_female_tc, test_male_female_tc = train_test_split(male_female_tc_df_average, test_size=0.25, shuffle=True)
    
    # animate vs inanimate
    acc_animate_inanimate_tc=[]
    kappa_animate_inanimate_tc=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_animate_inanimate_tc.iterrows():
            train.append(row.data[i])
        for index, row in test_animate_inanimate_tc.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_animate_inanimate_tc.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_animate_inanimate_tc.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_animate_inanimate_tc.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_animate_inanimate_tc.append(kappa_tmp)
    acc_animate_inanimate_tc_all.append(acc_animate_inanimate_tc)
    kappa_animate_inanimate_tc_all.append(kappa_animate_inanimate_tc)
    
    # face vs body
    acc_face_body_tc=[]
    kappa_face_body_tc=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_face_body_tc.iterrows():
            train.append(row.data[i])
        for index, row in test_face_body_tc.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_face_body_tc.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_face_body_tc.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_face_body_tc.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_face_body_tc.append(kappa_tmp)
    acc_face_body_tc_all.append(acc_face_body_tc)
    kappa_face_body_tc_all.append(kappa_face_body_tc)
    
    # male vs female
    acc_male_female_tc=[]
    kappa_male_female_tc=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_male_female_tc.iterrows():
            train.append(row.data[i])
        for index, row in test_male_female_tc.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_male_female_tc.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_male_female_tc.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_male_female_tc.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_male_female_tc.append(kappa_tmp)
    acc_male_female_tc_all.append(acc_male_female_tc)
    kappa_male_female_tc_all.append(kappa_male_female_tc)
        


# In[107]:


acc_animate_inanimate_tc_mean=np.nanmean(np.array(acc_animate_inanimate_tc_all), axis=0)
kappa_animate_inanimate_tc_mean= np.nanmean(np.array(kappa_animate_inanimate_tc_all), axis=0)
acc_face_body_tc_mean= np.nanmean(np.array(acc_face_body_tc_all), axis=0)
kappa_face_body_tc_mean= np.nanmean(np.array(kappa_face_body_tc_all), axis=0) 
acc_male_female_tc_mean= np.nanmean(np.array(acc_male_female_tc_all), axis=0) 
kappa_male_female_tc_mean= np.nanmean(np.array(kappa_male_female_tc_all), axis=0) 


# In[108]:


# saving
with open('acc_animate_inanimate_tc_mean_sub6.npy', 'wb') as f:
    np.save(f, acc_animate_inanimate_tc_mean)
    
with open('kappa_animate_inanimate_tc_mean_sub6.npy', 'wb') as f:
    np.save(f, kappa_animate_inanimate_tc_mean)
    
with open('acc_face_body_tc_mean_sub6.npy', 'wb') as f:
    np.save(f, acc_face_body_tc_mean)
    
with open('kappa_face_body_tc_mean_sub6.npy', 'wb') as f:
    np.save(f, kappa_face_body_tc_mean)
    
with open('acc_male_female_tc_mean_sub6.npy', 'wb') as f:
    np.save(f, acc_male_female_tc_mean)
    
with open('kappa_male_female_tc_mean_sub6.npy', 'wb') as f:
    np.save(f, kappa_male_female_tc_mean)


# In[109]:


# uploading
with open('acc_animate_inanimate_tc_mean_sub6.npy', 'rb') as f:
    acc_animate_inanimate_tc_mean = np.load(f)
    
with open('kappa_animate_inanimate_tc_mean_sub6.npy', 'rb') as f:
    kappa_animate_inanimate_tc_mean = np.load(f)
    
    
with open('acc_face_body_tc_mean_sub6.npy', 'rb') as f:
    acc_face_body_tc_mean = np.load(f)
    
with open('kappa_face_body_tc_mean_sub6.npy', 'rb') as f:
    kappa_face_body_tc_mean = np.load(f)
    
with open('acc_male_female_tc_mean_sub6.npy', 'rb') as f:
    acc_male_female_tc_mean = np.load(f)
    
with open('kappa_male_female_tc_mean_sub6.npy', 'rb') as f:
    kappa_male_female_tc_mean = np.load(f)


# ## Visualization before smoothing with gaussian kernel

# In[110]:


plt.figure(figsize=(12,5))
plt.plot(time, kappa_animate_inanimate_tc_mean, color="c", label = "animate_inanimate")
plt.plot(time, kappa_face_body_tc_mean, color="red", label = "face_body")
plt.plot(time, kappa_male_female_tc_mean, color="green", label = "male_female")

plt.title("Kappa in time for temporal channels")
plt.xlabel("time(ms)")
plt.ylabel("Kappa")
plt.legend()
plt.show()


# In[111]:


plt.figure(figsize=(12,6))
plt.plot(time, acc_animate_inanimate_tc_mean, color="c", label = "animate_inanimate")
plt.plot(time, acc_face_body_tc_mean, color="red", label = "face_body")
plt.plot(time, acc_male_female_tc_mean, color="green", label = "male_female")

plt.title("Accuracy in time for temporal channels")
plt.xlabel("time(ms)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ## smoothing with gaussian kernel

# In[112]:


x = np.arange(0, data_num)
sigma = 5
smoothed_acc_animate_inanimate_tc = np.zeros(acc_animate_inanimate_tc_mean.shape)
smoothed_kappa_animate_inanimate_tc = np.zeros(kappa_animate_inanimate_tc_mean.shape)
smoothed_acc_face_body_tc = np.zeros(acc_face_body_tc_mean.shape)
smoothed_kappa_face_body_tc = np.zeros(kappa_face_body_tc_mean.shape)
smoothed_acc_male_female_tc = np.zeros(acc_male_female_tc_mean.shape)
smoothed_kappa_male_female_tc = np.zeros(kappa_male_female_tc_mean.shape)

for x_position in x:
    kernel = np.exp(-((x - x_position) ** 2) / (2 * sigma**2))
    kernel = kernel / sum(kernel)
    smoothed_acc_animate_inanimate_tc[x_position] = sum(acc_animate_inanimate_tc_mean * kernel)
    smoothed_kappa_animate_inanimate_tc[x_position] = sum(kappa_animate_inanimate_tc_mean * kernel)
    smoothed_acc_face_body_tc[x_position] = sum(acc_face_body_tc_mean * kernel)
    smoothed_kappa_face_body_tc[x_position] = sum(kappa_face_body_tc_mean * kernel)
    smoothed_acc_male_female_tc[x_position] = sum(acc_male_female_tc_mean * kernel)
    smoothed_kappa_male_female_tc[x_position] = sum(kappa_male_female_tc_mean * kernel)


# ## Visualising data Vs smoothed

# In[113]:


fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(12,15))

axes[0][0].plot(time, acc_animate_inanimate_tc_mean, color="blue", label = "animate_inanimate")
axes[0][0].plot(time, smoothed_acc_animate_inanimate_tc, color="red", label = " smooth animate_inanimate")
axes[0][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[0][0].set_title("Accuracy animate_inanimate Vs smoothed in temporal channels", size=10)
axes[0][0].legend()

axes[0][1].plot(time, kappa_animate_inanimate_tc_mean, color="blue", label = "animate_inanimate")
axes[0][1].plot(time, smoothed_kappa_animate_inanimate_tc, color="red", label = " smooth animate_inanimate")
axes[0][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[0][1].set_title("Kappa animate_inanimate Vs smoothed in temporal channels", size=10)
axes[0][1].legend()

axes[1][0].plot(time, acc_face_body_tc_mean, color="blue", label = "face_body")
axes[1][0].plot(time, smoothed_acc_face_body_tc, color="red", label = " smooth face_body")
axes[1][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[1][0].set_title("Accuracy face_body Vs smoothed in temporal channels", size=10)
axes[1][0].legend()

axes[1][1].plot(time, kappa_face_body_tc_mean, color="blue", label = "face_body")
axes[1][1].plot(time, smoothed_kappa_face_body_tc, color="red", label = " smooth face_body")
axes[1][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[1][1].set_title("Kappa face_body Vs smoothed in temporal channels", size=10)
axes[1][1].legend()

axes[2][0].plot(time, acc_male_female_tc_mean, color="blue", label = "male_female")
axes[2][0].plot(time, smoothed_acc_male_female_tc, color="red", label = " smooth male_female")
axes[2][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[2][0].set_title("Accuracy male_female Vs smoothed in temporal channels", size=10)
axes[2][0].legend()

axes[2][1].plot(time, kappa_male_female_tc_mean, color="blue", label = "male_female")
axes[2][1].plot(time, smoothed_kappa_male_female_tc, color="red", label = " smooth male_female")
axes[2][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[2][1].set_title("Kappa male_female Vs smoothed in temporal channels", size=10)
axes[2][1].legend()


# ## Visualising smoothed data

# In[114]:


plt.figure(figsize=(12,5))
plt.plot(time, smoothed_kappa_animate_inanimate_tc, color="c", label = "animate_inanimate")
plt.plot(time, smoothed_kappa_face_body_tc, color="red", label = "face_body")
plt.plot(time, smoothed_kappa_male_female_tc, color="green", label = "male_female")

plt.title("Smoothed Kappa in time in temporal channels")
plt.xlabel("time(ms)")
plt.ylabel("Kappa")
plt.legend()
plt.show()


# In[115]:


plt.figure(figsize=(12,5))
plt.plot(time, smoothed_acc_animate_inanimate_tc, color="c", label = "animate_inanimate")
plt.plot(time, smoothed_acc_face_body_tc, color="red", label = "face_body")
plt.plot(time, smoothed_acc_male_female_tc, color="green", label = "male_female")

plt.title("Smoothed Accuracy in time in temporal channels")
plt.xlabel("time(ms)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# # Left temporal channels

# ## Seperating left temporal data

# In[116]:


tcl=np.array([8,9,18,19,20,27,28,36,37,44,45,73,74,75,83,84,91,92,99,100])-1


# In[117]:


animate_data_tcl=np.zeros((animate_num,data_num,len(tcl)))
inanimate_data_tcl=np.zeros((inanimate_num,data_num,len(tcl)))
face_data_tcl=np.zeros((face_num,data_num,len(tcl)))
body_data_tcl=np.zeros((body_num,data_num,len(tcl)))
male_data_tcl=np.zeros((male_num,data_num,len(tcl)))
female_data_tcl=np.zeros((female_num,data_num,len(tcl)))

for j in range(data_num):
    for i in range(animate_num):
        animate_data_tcl[i][j][:]=animate_data[i][j][tcl]
    for i in range(inanimate_num):
        inanimate_data_tcl[i][j][:]=inanimate_data[i][j][tcl]
    for i in range(face_num):
        face_data_tcl[i][j][:]=face_data[i][j][tcl]
    for i in range(body_num):
        body_data_tcl[i][j][:]=body_data[i][j][tcl]
    for i in range(male_num):
        male_data_tcl[i][j][:]=male_data[i][j][tcl]
    for i in range(female_num):
        female_data_tcl[i][j][:]=female_data[i][j][tcl]



# ## ERP

# In[118]:


animate_tcl_mean_trial=np.mean(animate_data_tcl,axis=0)
animate_tcl_ERP=np.mean(animate_tcl_mean_trial,axis=1)

inanimate_tcl_mean_trial=np.mean(inanimate_data_tcl,axis=0)
inanimate_tcl_ERP=np.mean(inanimate_tcl_mean_trial,axis=1)

face_tcl_mean_trial=np.mean(face_data_tcl,axis=0)
face_tcl_ERP=np.mean(face_tcl_mean_trial,axis=1)

body_tcl_mean_trial=np.mean(body_data_tcl,axis=0)
body_tcl_ERP=np.mean(body_tcl_mean_trial,axis=1)

male_tcl_mean_trial=np.mean(male_data_tcl,axis=0)
male_tcl_ERP=np.mean(male_tcl_mean_trial,axis=1)

female_tcl_mean_trial=np.mean(female_data_tcl,axis=0)
female_tcl_ERP=np.mean(female_tcl_mean_trial,axis=1)


# In[119]:


fig, (ax1,ax2,ax3) = plt.subplots(3, 1)
fig.set_figheight(20)
fig.set_figwidth(12)

ax1.plot(time, animate_tcl_ERP, color="c", label = "animate")
ax1.plot(time, inanimate_tcl_ERP, color="orange", label = "inanimate")
ax1.plot(time,np.zeros(data_num),color="black")
ax1.set(xlabel='time(ms)',ylabel='ERP')
ax1.set_title("ERP of left temporal channels animate Vs inanimate", size=15)
ax1.legend()


ax2.plot(time, face_tcl_ERP, color="red", label = "face")
ax2.plot(time, body_tcl_ERP, color="blue", label = "body")
ax2.plot(time,np.zeros(data_num),color="black")
ax2.set(xlabel='time(ms)',ylabel='ERP')
ax2.set_title("ERP of left temporal channels face Vs body", size=15)
ax2.legend()

ax3.plot(time, male_tcl_ERP, color="pink", label = "male")
ax3.plot(time, female_tcl_ERP, color="purple", label = "female")
ax3.plot(time,np.zeros(data_num),color="black")
ax3.set(xlabel='time(ms)',ylabel='ERP')
ax3.set_title("ERP of left temporal channels male Vs female", size=15)
ax3.legend()


# ## Smoothing data

# In[120]:


animate_data_tcl_smooth=np.empty([animate_num, data_num,len(tcl)])
inanimate_data_tcl_smooth=np.empty([inanimate_num, data_num,len(tcl)])
face_data_tcl_smooth=np.empty([face_num, data_num,len(tcl)])
body_data_tcl_smooth=np.empty([body_num, data_num,len(tcl)])
male_data_tcl_smooth=np.empty([male_num, data_num,len(tcl)])
female_data_tcl_smooth=np.empty([female_num, data_num,len(tcl)])


# In[121]:


for i in range(animate_num):
    for j in range(data_num-60):
        animate_data_tcl_smooth[i][j]=np.mean(animate_data_tcl[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        animate_data_tcl_smooth[i][k]=np.mean(animate_data_tcl[i][k:], axis=0)
    
for i in range(inanimate_num):
    for j in range(data_num-60):
        inanimate_data_tcl_smooth[i][j]=np.mean(inanimate_data_tcl[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        inanimate_data_tcl_smooth[i][k]=np.mean(inanimate_data_tcl[i][k:], axis=0)
    
for i in range(face_num):
    for j in range(data_num-60):
        face_data_tcl_smooth[i][j]=np.mean(face_data_tcl[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        face_data_tcl_smooth[i][k]=np.mean(face_data_tcl[i][k:], axis=0)
    
for i in range(body_num):
    for j in range(data_num-60):
        body_data_tcl_smooth[i][j]=np.mean(body_data_tcl[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        body_data_tcl_smooth[i][k]=np.mean(body_data_tcl[i][k:], axis=0)
    
for i in range(male_num):
    for j in range(data_num-60):
        male_data_tcl_smooth[i][j]=np.mean(male_data_tcl[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        male_data_tcl_smooth[i][k]=np.mean(male_data_tcl[i][k:], axis=0)
    
for i in range(female_num):
    for j in range(data_num-60):
        female_data_tcl_smooth[i][j]=np.mean(female_data_tcl[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        female_data_tcl_smooth[i][k]=np.mean(female_data_tcl[i][k:], axis=0)


# ## Creating data frames

# ### animat =1 inanimate=2  face=3 body=4 male=5 female=6

# In[122]:


animate_tcl_df_smooth = pd.DataFrame(columns=['data', 'label'])
inanimate_tcl_df_smooth = pd.DataFrame(columns=['data', 'label'])

face_tcl_df_smooth = pd.DataFrame(columns=['data', 'label'])
body_tcl_df_smooth = pd.DataFrame(columns=['data', 'label'])

male_tcl_df_smooth = pd.DataFrame(columns=['data', 'label'])
female_tcl_df_smooth = pd.DataFrame(columns=['data', 'label'])


# In[123]:


for i in range(animate_num):
    new_row={'data':animate_data_tcl_smooth[i][:][:], 'label':1}
    animate_tcl_df_smooth = animate_tcl_df_smooth.append(new_row, ignore_index=True)
    
for i in range(inanimate_num):
    new_row={'data':inanimate_data_tcl_smooth[i][:][:], 'label':2}
    inanimate_tcl_df_smooth =inanimate_tcl_df_smooth.append(new_row, ignore_index=True)
    
for i in range(face_num):
    new_row={'data':face_data_tcl_smooth[i][:][:], 'label':3}
    face_tcl_df_smooth = face_tcl_df_smooth.append(new_row, ignore_index=True)
    
for i in range(body_num):
    new_row={'data':body_data_tcl_smooth[i][:][:], 'label':4}
    body_tcl_df_smooth = body_tcl_df_smooth.append(new_row, ignore_index=True)
    
for i in range(male_num):
    new_row={'data':male_data_tcl_smooth[i][:][:], 'label':5}
    male_tcl_df_smooth = male_tcl_df_smooth.append(new_row, ignore_index=True)
    
for i in range(female_num):
    new_row={'data':female_data_tcl_smooth[i][:][:], 'label':6}
    female_tcl_df_smooth= female_tcl_df_smooth.append(new_row, ignore_index=True)


# In[124]:


female_tcl_df_smooth.data[0].shape


# ## Averaging each 4 trials randomly

# In[125]:


def averaging_trials_tcl():
    ave_num=4
    animate_tcl_shuffle_df=animate_tcl_df_smooth.sample(frac = 1)
    inanimate_tcl_shuffle_df=inanimate_tcl_df_smooth.sample(frac = 1)
    face_tcl_shuffle_df=face_tcl_df_smooth.sample(frac = 1)
    body_tcl_shuffle_df=body_tcl_df_smooth.sample(frac = 1)
    male_tcl_shuffle_df=male_tcl_df_smooth.sample(frac = 1)
    female_tcl_shuffle_df=female_tcl_df_smooth.sample(frac = 1)
    
    animate_inanimate_tcl_df_average = pd.DataFrame(columns=['data', 'label'])
    face_body_tcl_df_average = pd.DataFrame(columns=['data', 'label'])
    male_female_tcl_df_average = pd.DataFrame(columns=['data', 'label'])
    
    for i in range(int(animate_num/ave_num)):
        new_row={'data':animate_tcl_shuffle_df.data[i*4:i*4+4].mean(), 'label':1}
        animate_inanimate_tcl_df_average = animate_inanimate_tcl_df_average.append(new_row, ignore_index=True)
    if animate_num%ave_num !=0:
        tmp_animate=int(animate_num/ave_num)
        new_row={'data':animate_tcl_shuffle_df.data[tmp_animate*4:].mean(), 'label':1}
        animate_inanimate_tcl_df_average = animate_inanimate_tcl_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(inanimate_num/ave_num)):
        new_row={'data':inanimate_tcl_shuffle_df.data[i*4:i*4+4].mean(), 'label':2}
        animate_inanimate_tcl_df_average = animate_inanimate_tcl_df_average.append(new_row, ignore_index=True)
    if inanimate_num%ave_num !=0:
        tmp_inanimate=int(inanimate_num/ave_num)
        new_row={'data':inanimate_tcl_shuffle_df.data[tmp_inanimate*4:].mean(), 'label':2}
        animate_inanimate_tcl_df_average = animate_inanimate_tcl_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(face_num/ave_num)):
        new_row={'data':face_tcl_shuffle_df.data[i*4:i*4+4].mean(), 'label':3}
        face_body_tcl_df_average = face_body_tcl_df_average.append(new_row, ignore_index=True)
    if face_num%ave_num !=0:
        tmp_face=int(face_num/ave_num)
        new_row={'data':face_tcl_shuffle_df.data[tmp_face*4:].mean(), 'label':3}
        face_body_tcl_df_average = face_body_tcl_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(body_num/ave_num)):
        new_row={'data':body_tcl_shuffle_df.data[i*4:i*4+4].mean(), 'label':4}
        face_body_tcl_df_average = face_body_tcl_df_average.append(new_row, ignore_index=True)
    if body_num%ave_num !=0:
        tmp_body=int(body_num/ave_num)
        new_row={'data':body_tcl_shuffle_df.data[tmp_body*4:].mean(), 'label':4}
        face_body_tcl_df_average = face_body_tcl_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(male_num/ave_num)):
        new_row={'data':male_tcl_shuffle_df.data[i*4:i*4+4].mean(), 'label':5}
        male_female_tcl_df_average = male_female_tcl_df_average.append(new_row, ignore_index=True)
    if male_num%ave_num !=0:
        tmp_male=int(male_num/ave_num)
        new_row={'data':male_tcl_shuffle_df.data[tmp_male*4:].mean(), 'label':5}
        male_female_tcl_df_average = male_female_tcl_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(female_num/ave_num)):
        new_row={'data':female_tcl_shuffle_df.data[i*4:i*4+4].mean(), 'label':6}
        male_female_tcl_df_average = male_female_tcl_df_average.append(new_row, ignore_index=True)
    if female_num%ave_num !=0:
        tmp_female=int(female_num/ave_num)
        new_row={'data':female_tcl_shuffle_df.data[tmp_female*4:].mean(), 'label':6}
        male_female_tcl_df_average = male_female_tcl_df_average.append(new_row, ignore_index=True)
    
    
    return animate_inanimate_tcl_df_average,face_body_tcl_df_average,male_female_tcl_df_average


# ## Training the classifier

# In[126]:


acc_animate_inanimate_tcl_all=[]
kappa_animate_inanimate_tcl_all=[]
acc_face_body_tcl_all=[]
kappa_face_body_tcl_all=[]
acc_male_female_tcl_all=[]
kappa_male_female_tcl_all=[]

for j in range(100):
    print(j)
    # averaging each 4 trials
    animate_inanimate_tcl_df_average,face_body_tcl_df_average,male_female_tcl_df_average=averaging_trials_tcl()
    
    # train test split
    train_animate_inanimate_tcl, test_animate_inanimate_tcl = train_test_split(animate_inanimate_tcl_df_average, test_size=0.25, shuffle=True)
    train_face_body_tcl, test_face_body_tcl = train_test_split(face_body_tcl_df_average, test_size=0.25, shuffle=True)
    train_male_female_tcl, test_male_female_tcl = train_test_split(male_female_tcl_df_average, test_size=0.25, shuffle=True)
    
    # animate vs inanimate
    acc_animate_inanimate_tcl=[]
    kappa_animate_inanimate_tcl=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_animate_inanimate_tcl.iterrows():
            train.append(row.data[i])
        for index, row in test_animate_inanimate_tcl.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_animate_inanimate_tcl.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_animate_inanimate_tcl.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_animate_inanimate_tcl.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_animate_inanimate_tcl.append(kappa_tmp)
    acc_animate_inanimate_tcl_all.append(acc_animate_inanimate_tcl)
    kappa_animate_inanimate_tcl_all.append(kappa_animate_inanimate_tcl)
    
    # face vs body
    acc_face_body_tcl=[]
    kappa_face_body_tcl=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_face_body_tcl.iterrows():
            train.append(row.data[i])
        for index, row in test_face_body_tcl.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_face_body_tcl.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_face_body_tcl.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_face_body_tcl.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_face_body_tcl.append(kappa_tmp)
    acc_face_body_tcl_all.append(acc_face_body_tcl)
    kappa_face_body_tcl_all.append(kappa_face_body_tcl)
    
    # male vs female
    acc_male_female_tcl=[]
    kappa_male_female_tcl=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_male_female_tcl.iterrows():
            train.append(row.data[i])
        for index, row in test_male_female_tcl.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_male_female_tcl.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_male_female_tcl.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_male_female_tcl.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_male_female_tcl.append(kappa_tmp)
    acc_male_female_tcl_all.append(acc_male_female_tcl)
    kappa_male_female_tcl_all.append(kappa_male_female_tcl)
        


# In[127]:


acc_animate_inanimate_tcl_mean=np.nanmean(np.array(acc_animate_inanimate_tcl_all), axis=0)
kappa_animate_inanimate_tcl_mean= np.nanmean(np.array(kappa_animate_inanimate_tcl_all), axis=0)
acc_face_body_tcl_mean= np.nanmean(np.array(acc_face_body_tcl_all), axis=0)
kappa_face_body_tcl_mean= np.nanmean(np.array(kappa_face_body_tcl_all), axis=0) 
acc_male_female_tcl_mean= np.nanmean(np.array(acc_male_female_tcl_all), axis=0) 
kappa_male_female_tcl_mean= np.nanmean(np.array(kappa_male_female_tcl_all), axis=0) 


# In[128]:


# saving
with open('acc_animate_inanimate_tcl_mean_sub6.npy', 'wb') as f:
    np.save(f, acc_animate_inanimate_tcl_mean)
    
with open('kappa_animate_inanimate_tcl_mean_sub6.npy', 'wb') as f:
    np.save(f, kappa_animate_inanimate_tcl_mean)
    
with open('acc_face_body_tcl_mean_sub6.npy', 'wb') as f:
    np.save(f, acc_face_body_tcl_mean)
    
with open('kappa_face_body_tcl_mean_sub6.npy', 'wb') as f:
    np.save(f, kappa_face_body_tcl_mean)
    
with open('acc_male_female_tcl_mean_sub6.npy', 'wb') as f:
    np.save(f, acc_male_female_tcl_mean)
    
with open('kappa_male_female_tcl_mean_sub6.npy', 'wb') as f:
    np.save(f, kappa_male_female_tcl_mean)


# In[129]:


# uploading
with open('acc_animate_inanimate_tcl_mean_sub6.npy', 'rb') as f:
    acc_animate_inanimate_tcl_mean = np.load(f)
    
with open('kappa_animate_inanimate_tcl_mean_sub6.npy', 'rb') as f:
    kappa_animate_inanimate_tcl_mean = np.load(f)
    
    
with open('acc_face_body_tcl_mean_sub6.npy', 'rb') as f:
    acc_face_body_tcl_mean = np.load(f)
    
with open('kappa_face_body_tcl_mean_sub6.npy', 'rb') as f:
    kappa_face_body_tcl_mean = np.load(f)
    
with open('acc_male_female_tcl_mean_sub6.npy', 'rb') as f:
    acc_male_female_tcl_mean = np.load(f)
    
with open('kappa_male_female_tcl_mean_sub6.npy', 'rb') as f:
    kappa_male_female_tcl_mean = np.load(f)


# ## Visualization before smoothing with gaussian kernel

# In[130]:


plt.figure(figsize=(12,5))
plt.plot(time, kappa_animate_inanimate_tcl_mean, color="c", label = "animate_inanimate")
plt.plot(time, kappa_face_body_tcl_mean, color="red", label = "face_body")
plt.plot(time, kappa_male_female_tcl_mean, color="green", label = "male_female")

plt.title("Kappa in time for left temporal channels")
plt.xlabel("time(ms)")
plt.ylabel("Kappa")
plt.legend()
plt.show()


# In[131]:


plt.figure(figsize=(12,6))
plt.plot(time, acc_animate_inanimate_tcl_mean, color="c", label = "animate_inanimate")
plt.plot(time, acc_face_body_tcl_mean, color="red", label = "face_body")
plt.plot(time, acc_male_female_tcl_mean, color="green", label = "male_female")

plt.title("Accuracy in time for left temporal channels")
plt.xlabel("time(ms)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ## smoothing with gaussian kernel

# In[132]:


x = np.arange(0, data_num)
sigma = 5
smoothed_acc_animate_inanimate_tcl = np.zeros(acc_animate_inanimate_tcl_mean.shape)
smoothed_kappa_animate_inanimate_tcl = np.zeros(kappa_animate_inanimate_tcl_mean.shape)
smoothed_acc_face_body_tcl = np.zeros(acc_face_body_tcl_mean.shape)
smoothed_kappa_face_body_tcl = np.zeros(kappa_face_body_tcl_mean.shape)
smoothed_acc_male_female_tcl = np.zeros(acc_male_female_tcl_mean.shape)
smoothed_kappa_male_female_tcl = np.zeros(kappa_male_female_tcl_mean.shape)

for x_position in x:
    kernel = np.exp(-((x - x_position) ** 2) / (2 * sigma**2))
    kernel = kernel / sum(kernel)
    smoothed_acc_animate_inanimate_tcl[x_position] = sum(acc_animate_inanimate_tcl_mean * kernel)
    smoothed_kappa_animate_inanimate_tcl[x_position] = sum(kappa_animate_inanimate_tcl_mean * kernel)
    smoothed_acc_face_body_tcl[x_position] = sum(acc_face_body_tcl_mean * kernel)
    smoothed_kappa_face_body_tcl[x_position] = sum(kappa_face_body_tcl_mean * kernel)
    smoothed_acc_male_female_tcl[x_position] = sum(acc_male_female_tcl_mean * kernel)
    smoothed_kappa_male_female_tcl[x_position] = sum(kappa_male_female_tcl_mean * kernel)


# ## Visualizing data Vs smoothed

# In[133]:


fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(12,15))

axes[0][0].plot(time, acc_animate_inanimate_tcl_mean, color="blue", label = "animate_inanimate")
axes[0][0].plot(time, smoothed_acc_animate_inanimate_tcl, color="red", label = " smooth animate_inanimate")
axes[0][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[0][0].set_title("Accuracy animate_inanimate Vs smoothed in left temporal channels", size=10)
axes[0][0].legend()

axes[0][1].plot(time, kappa_animate_inanimate_tcl_mean, color="blue", label = "animate_inanimate")
axes[0][1].plot(time, smoothed_kappa_animate_inanimate_tcl, color="red", label = " smooth animate_inanimate")
axes[0][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[0][1].set_title("Kappa animate_inanimate Vs smoothed in left temporal channels", size=10)
axes[0][1].legend()

axes[1][0].plot(time, acc_face_body_tcl_mean, color="blue", label = "face_body")
axes[1][0].plot(time, smoothed_acc_face_body_tcl, color="red", label = " smooth face_body")
axes[1][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[1][0].set_title("Accuracy face_body Vs smoothed in left temporal channels", size=10)
axes[1][0].legend()

axes[1][1].plot(time, kappa_face_body_tcl_mean, color="blue", label = "face_body")
axes[1][1].plot(time, smoothed_kappa_face_body_tcl, color="red", label = " smooth face_body")
axes[1][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[1][1].set_title("Kappa face_body Vs smoothed in left temporal channels", size=10)
axes[1][1].legend()

axes[2][0].plot(time, acc_male_female_tcl_mean, color="blue", label = "male_female")
axes[2][0].plot(time, smoothed_acc_male_female_tcl, color="red", label = " smooth male_female")
axes[2][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[2][0].set_title("Accuracy male_female Vs smoothed in left temporal channels", size=10)
axes[2][0].legend()

axes[2][1].plot(time, kappa_male_female_tcl_mean, color="blue", label = "male_female")
axes[2][1].plot(time, smoothed_kappa_male_female_tcl, color="red", label = " smooth male_female")
axes[2][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[2][1].set_title("Kappa male_female Vs smoothed in left temporal channels", size=10)
axes[2][1].legend()


# ## Visualizing smoothed data

# In[134]:


plt.figure(figsize=(12,5))
plt.plot(time, smoothed_kappa_animate_inanimate_tcl, color="c", label = "animate_inanimate")
plt.plot(time, smoothed_kappa_face_body_tcl, color="red", label = "face_body")
plt.plot(time, smoothed_kappa_male_female_tcl, color="green", label = "male_female")

plt.title("Smoothed Kappa in time in left temporal channels")
plt.xlabel("time(ms)")
plt.ylabel("Kappa")
plt.legend()
plt.show()


# In[135]:


plt.figure(figsize=(12,5))
plt.plot(time, smoothed_acc_animate_inanimate_tcl, color="c", label = "animate_inanimate")
plt.plot(time, smoothed_acc_face_body_tcl, color="red", label = "face_body")
plt.plot(time, smoothed_acc_male_female_tcl, color="green", label = "male_female")

plt.title("Smoothed Accuracy in time in left temporal channels")
plt.xlabel("time(ms)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# # Right Temporal channels

# ## Seporating right temporal data

# In[136]:


tcr=np.array([17,24,25,26,34,35,43,51,52,80,81,82,89,90,97,98,105,106])-1


# In[137]:


animate_data_tcr=np.zeros((animate_num,data_num,len(tcr)))
inanimate_data_tcr=np.zeros((inanimate_num,data_num,len(tcr)))
face_data_tcr=np.zeros((face_num,data_num,len(tcr)))
body_data_tcr=np.zeros((body_num,data_num,len(tcr)))
male_data_tcr=np.zeros((male_num,data_num,len(tcr)))
female_data_tcr=np.zeros((female_num,data_num,len(tcr)))

for j in range(data_num):
    for i in range(animate_num):
        animate_data_tcr[i][j][:]=animate_data[i][j][tcr]
    for i in range(inanimate_num):
        inanimate_data_tcr[i][j][:]=inanimate_data[i][j][tcr]
    for i in range(face_num):
        face_data_tcr[i][j][:]=face_data[i][j][tcr]
    for i in range(body_num):
        body_data_tcr[i][j][:]=body_data[i][j][tcr]
    for i in range(male_num):
        male_data_tcr[i][j][:]=male_data[i][j][tcr]
    for i in range(female_num):
        female_data_tcr[i][j][:]=female_data[i][j][tcr]



# ## ERP

# In[138]:


animate_tcr_mean_trial=np.mean(animate_data_tcr,axis=0)
animate_tcr_ERP=np.mean(animate_tcr_mean_trial,axis=1)

inanimate_tcr_mean_trial=np.mean(inanimate_data_tcr,axis=0)
inanimate_tcr_ERP=np.mean(inanimate_tcr_mean_trial,axis=1)

face_tcr_mean_trial=np.mean(face_data_tcr,axis=0)
face_tcr_ERP=np.mean(face_tcr_mean_trial,axis=1)

body_tcr_mean_trial=np.mean(body_data_tcr,axis=0)
body_tcr_ERP=np.mean(body_tcr_mean_trial,axis=1)

male_tcr_mean_trial=np.mean(male_data_tcr,axis=0)
male_tcr_ERP=np.mean(male_tcr_mean_trial,axis=1)

female_tcr_mean_trial=np.mean(female_data_tcr,axis=0)
female_tcr_ERP=np.mean(female_tcr_mean_trial,axis=1)


# In[139]:


fig, (ax1,ax2,ax3) = plt.subplots(3, 1)
fig.set_figheight(20)
fig.set_figwidth(12)

ax1.plot(time, animate_tcr_ERP, color="c", label = "animate")
ax1.plot(time, inanimate_tcr_ERP, color="orange", label = "inanimate")
ax1.plot(time,np.zeros(data_num),color="black")
ax1.set(xlabel='time(ms)',ylabel='ERP')
ax1.set_title("ERP of right temporal channels animate Vs inanimate", size=15)
ax1.legend()


ax2.plot(time, face_tcr_ERP, color="red", label = "face")
ax2.plot(time, body_tcr_ERP, color="blue", label = "body")
ax2.plot(time,np.zeros(data_num),color="black")
ax2.set(xlabel='time(ms)',ylabel='ERP')
ax2.set_title("ERP of right temporal channels face Vs body", size=15)
ax2.legend()

ax3.plot(time, male_tcr_ERP, color="pink", label = "male")
ax3.plot(time, female_tcr_ERP, color="purple", label = "female")
ax3.plot(time,np.zeros(data_num),color="black")
ax3.set(xlabel='time(ms)',ylabel='ERP')
ax3.set_title("ERP of right temporal channels male Vs female", size=15)
ax3.legend()


# ## Smoothing data

# In[140]:


animate_data_tcr_smooth=np.empty([animate_num, data_num,len(tcr)])
inanimate_data_tcr_smooth=np.empty([inanimate_num, data_num,len(tcr)])
face_data_tcr_smooth=np.empty([face_num, data_num,len(tcr)])
body_data_tcr_smooth=np.empty([body_num, data_num,len(tcr)])
male_data_tcr_smooth=np.empty([male_num, data_num,len(tcr)])
female_data_tcr_smooth=np.empty([female_num, data_num,len(tcr)])


# In[141]:


for i in range(animate_num):
    for j in range(data_num-60):
        animate_data_tcr_smooth[i][j]=np.mean(animate_data_tcr[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        animate_data_tcr_smooth[i][k]=np.mean(animate_data_tcr[i][k:], axis=0)
    
for i in range(inanimate_num):
    for j in range(data_num-60):
        inanimate_data_tcr_smooth[i][j]=np.mean(inanimate_data_tcr[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        inanimate_data_tcr_smooth[i][k]=np.mean(inanimate_data_tcr[i][k:], axis=0)
    
for i in range(face_num):
    for j in range(data_num-60):
        face_data_tcr_smooth[i][j]=np.mean(face_data_tcr[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        face_data_tcr_smooth[i][k]=np.mean(face_data_tcr[i][k:], axis=0)
    
for i in range(body_num):
    for j in range(data_num-60):
        body_data_tcr_smooth[i][j]=np.mean(body_data_tcr[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        body_data_tcr_smooth[i][k]=np.mean(body_data_tcr[i][k:], axis=0)
    
for i in range(male_num):
    for j in range(data_num-60):
        male_data_tcr_smooth[i][j]=np.mean(male_data_tcr[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        male_data_tcr_smooth[i][k]=np.mean(male_data_tcr[i][k:], axis=0)
    
for i in range(female_num):
    for j in range(data_num-60):
        female_data_tcr_smooth[i][j]=np.mean(female_data_tcr[i][j:j+60], axis=0)
    for j in range(60):
        k=j+data_num-60
        female_data_tcr_smooth[i][k]=np.mean(female_data_tcr[i][k:], axis=0)


# ## Creating data frames

# ### animat =1 inanimate=2  face=3 body=4 male=5 female=6

# In[142]:


animate_tcr_df_smooth = pd.DataFrame(columns=['data', 'label'])
inanimate_tcr_df_smooth = pd.DataFrame(columns=['data', 'label'])

face_tcr_df_smooth = pd.DataFrame(columns=['data', 'label'])
body_tcr_df_smooth = pd.DataFrame(columns=['data', 'label'])

male_tcr_df_smooth = pd.DataFrame(columns=['data', 'label'])
female_tcr_df_smooth = pd.DataFrame(columns=['data', 'label'])


# In[143]:


for i in range(animate_num):
    new_row={'data':animate_data_tcr_smooth[i][:][:], 'label':1}
    animate_tcr_df_smooth = animate_tcr_df_smooth.append(new_row, ignore_index=True)
    
for i in range(inanimate_num):
    new_row={'data':inanimate_data_tcr_smooth[i][:][:], 'label':2}
    inanimate_tcr_df_smooth =inanimate_tcr_df_smooth.append(new_row, ignore_index=True)
    
for i in range(face_num):
    new_row={'data':face_data_tcr_smooth[i][:][:], 'label':3}
    face_tcr_df_smooth = face_tcr_df_smooth.append(new_row, ignore_index=True)
    
for i in range(body_num):
    new_row={'data':body_data_tcr_smooth[i][:][:], 'label':4}
    body_tcr_df_smooth = body_tcr_df_smooth.append(new_row, ignore_index=True)
    
for i in range(male_num):
    new_row={'data':male_data_tcr_smooth[i][:][:], 'label':5}
    male_tcr_df_smooth = male_tcr_df_smooth.append(new_row, ignore_index=True)
    
for i in range(female_num):
    new_row={'data':female_data_tcr_smooth[i][:][:], 'label':6}
    female_tcr_df_smooth= female_tcr_df_smooth.append(new_row, ignore_index=True)


# In[144]:


female_tcr_df_smooth.data[0].shape


# ## Averaging each 4 trials randomly

# In[145]:


def averaging_trials_tcr():
    ave_num=4
    animate_tcr_shuffle_df=animate_tcr_df_smooth.sample(frac = 1)
    inanimate_tcr_shuffle_df=inanimate_tcr_df_smooth.sample(frac = 1)
    face_tcr_shuffle_df=face_tcr_df_smooth.sample(frac = 1)
    body_tcr_shuffle_df=body_tcr_df_smooth.sample(frac = 1)
    male_tcr_shuffle_df=male_tcr_df_smooth.sample(frac = 1)
    female_tcr_shuffle_df=female_tcr_df_smooth.sample(frac = 1)
    
    animate_inanimate_tcr_df_average = pd.DataFrame(columns=['data', 'label'])
    face_body_tcr_df_average = pd.DataFrame(columns=['data', 'label'])
    male_female_tcr_df_average = pd.DataFrame(columns=['data', 'label'])
    
    for i in range(int(animate_num/ave_num)):
        new_row={'data':animate_tcr_shuffle_df.data[i*4:i*4+4].mean(), 'label':1}
        animate_inanimate_tcr_df_average = animate_inanimate_tcr_df_average.append(new_row, ignore_index=True)
    if animate_num%ave_num !=0:
        tmp_animate=int(animate_num/ave_num)
        new_row={'data':animate_tcr_shuffle_df.data[tmp_animate*4:].mean(), 'label':1}
        animate_inanimate_tcr_df_average = animate_inanimate_tcr_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(inanimate_num/ave_num)):
        new_row={'data':inanimate_tcr_shuffle_df.data[i*4:i*4+4].mean(), 'label':2}
        animate_inanimate_tcr_df_average = animate_inanimate_tcr_df_average.append(new_row, ignore_index=True)
    if inanimate_num%ave_num !=0:
        tmp_inanimate=int(inanimate_num/ave_num)
        new_row={'data':inanimate_tcr_shuffle_df.data[tmp_inanimate*4:].mean(), 'label':2}
        animate_inanimate_tcr_df_average = animate_inanimate_tcr_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(face_num/ave_num)):
        new_row={'data':face_tcr_shuffle_df.data[i*4:i*4+4].mean(), 'label':3}
        face_body_tcr_df_average = face_body_tcr_df_average.append(new_row, ignore_index=True)
    if face_num%ave_num !=0:
        tmp_face=int(face_num/ave_num)
        new_row={'data':face_tcr_shuffle_df.data[tmp_face*4:].mean(), 'label':3}
        face_body_tcr_df_average = face_body_tcr_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(body_num/ave_num)):
        new_row={'data':body_tcr_shuffle_df.data[i*4:i*4+4].mean(), 'label':4}
        face_body_tcr_df_average = face_body_tcr_df_average.append(new_row, ignore_index=True)
    if body_num%ave_num !=0:
        tmp_body=int(body_num/ave_num)
        new_row={'data':body_tcr_shuffle_df.data[tmp_body*4:].mean(), 'label':4}
        face_body_tcr_df_average = face_body_tcr_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(male_num/ave_num)):
        new_row={'data':male_tcr_shuffle_df.data[i*4:i*4+4].mean(), 'label':5}
        male_female_tcr_df_average = male_female_tcr_df_average.append(new_row, ignore_index=True)
    if male_num%ave_num !=0:
        tmp_male=int(male_num/ave_num)
        new_row={'data':male_tcr_shuffle_df.data[tmp_male*4:].mean(), 'label':5}
        male_female_tcr_df_average = male_female_tcr_df_average.append(new_row, ignore_index=True)
    
    for i in range(int(female_num/ave_num)):
        new_row={'data':female_tcr_shuffle_df.data[i*4:i*4+4].mean(), 'label':6}
        male_female_tcr_df_average = male_female_tcr_df_average.append(new_row, ignore_index=True)
    if female_num%ave_num !=0:
        tmp_female=int(female_num/ave_num)
        new_row={'data':female_tcr_shuffle_df.data[tmp_female*4:].mean(), 'label':6}
        male_female_tcr_df_average = male_female_tcr_df_average.append(new_row, ignore_index=True)
    
    
    return animate_inanimate_tcr_df_average,face_body_tcr_df_average,male_female_tcr_df_average


# ## Training the classifier

# In[146]:


acc_animate_inanimate_tcr_all=[]
kappa_animate_inanimate_tcr_all=[]
acc_face_body_tcr_all=[]
kappa_face_body_tcr_all=[]
acc_male_female_tcr_all=[]
kappa_male_female_tcr_all=[]

for j in range(100):
    print(j)
    # averaging each 4 trials
    animate_inanimate_tcr_df_average,face_body_tcr_df_average,male_female_tcr_df_average=averaging_trials_tcr()
    
    # train test split
    train_animate_inanimate_tcr, test_animate_inanimate_tcr = train_test_split(animate_inanimate_tcr_df_average, test_size=0.25, shuffle=True)
    train_face_body_tcr, test_face_body_tcr = train_test_split(face_body_tcr_df_average, test_size=0.25, shuffle=True)
    train_male_female_tcr, test_male_female_tcr = train_test_split(male_female_tcr_df_average, test_size=0.25, shuffle=True)
    
    # animate vs inanimate
    acc_animate_inanimate_tcr=[]
    kappa_animate_inanimate_tcr=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_animate_inanimate_tcr.iterrows():
            train.append(row.data[i])
        for index, row in test_animate_inanimate_tcr.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_animate_inanimate_tcr.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_animate_inanimate_tcr.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_animate_inanimate_tcr.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_animate_inanimate_tcr.append(kappa_tmp)
    acc_animate_inanimate_tcr_all.append(acc_animate_inanimate_tcr)
    kappa_animate_inanimate_tcr_all.append(kappa_animate_inanimate_tcr)
    
    # face vs body
    acc_face_body_tcr=[]
    kappa_face_body_tcr=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_face_body_tcr.iterrows():
            train.append(row.data[i])
        for index, row in test_face_body_tcr.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_face_body_tcr.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_face_body_tcr.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_face_body_tcr.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_face_body_tcr.append(kappa_tmp)
    acc_face_body_tcr_all.append(acc_face_body_tcr)
    kappa_face_body_tcr_all.append(kappa_face_body_tcr)
    
    # male vs female
    acc_male_female_tcr=[]
    kappa_male_female_tcr=[]
    for i in range(data_num):
        train=[]
        test=[]
        for index, row in train_male_female_tcr.iterrows():
            train.append(row.data[i])
        for index, row in test_male_female_tcr.iterrows():
            test.append(row.data[i])
        clf_tmp=LinearDiscriminantAnalysis()
        train_label=np.array(train_male_female_tcr.label.to_numpy().tolist())
        clf_tmp.fit(train,train_label)
        test_label=np.array(test_male_female_tcr.label.to_numpy().tolist())
        acc_tmp=clf_tmp.score(test,test_label)
        acc_male_female_tcr.append(acc_tmp)
        pred=clf_tmp.predict(test)
        kappa_tmp=cohen_kappa_score(test_label, pred)
        kappa_male_female_tcr.append(kappa_tmp)
    acc_male_female_tcr_all.append(acc_male_female_tcr)
    kappa_male_female_tcr_all.append(kappa_male_female_tcr)
        


# In[147]:


acc_animate_inanimate_tcr_mean=np.nanmean(np.array(acc_animate_inanimate_tcr_all), axis=0)
kappa_animate_inanimate_tcr_mean= np.nanmean(np.array(kappa_animate_inanimate_tcr_all), axis=0)
acc_face_body_tcr_mean= np.nanmean(np.array(acc_face_body_tcr_all), axis=0)
kappa_face_body_tcr_mean= np.nanmean(np.array(kappa_face_body_tcr_all), axis=0) 
acc_male_female_tcr_mean= np.nanmean(np.array(acc_male_female_tcr_all), axis=0) 
kappa_male_female_tcr_mean= np.nanmean(np.array(kappa_male_female_tcr_all), axis=0) 


# In[148]:


# saving
with open('acc_animate_inanimate_tcr_mean_sub6.npy', 'wb') as f:
    np.save(f, acc_animate_inanimate_tcr_mean)
    
with open('kappa_animate_inanimate_tcr_mean_sub6.npy', 'wb') as f:
    np.save(f, kappa_animate_inanimate_tcr_mean)
    
with open('acc_face_body_tcr_mean_sub6.npy', 'wb') as f:
    np.save(f, acc_face_body_tcr_mean)
    
with open('kappa_face_body_tcr_mean_sub6.npy', 'wb') as f:
    np.save(f, kappa_face_body_tcr_mean)
    
with open('acc_male_female_tcr_mean_sub6.npy', 'wb') as f:
    np.save(f, acc_male_female_tcr_mean)
    
with open('kappa_male_female_tcr_mean_sub6.npy', 'wb') as f:
    np.save(f, kappa_male_female_tcr_mean)


# In[149]:


# uploading
with open('acc_animate_inanimate_tcr_mean_sub6.npy', 'rb') as f:
    acc_animate_inanimate_tcr_mean = np.load(f)
    
with open('kappa_animate_inanimate_tcr_mean_sub6.npy', 'rb') as f:
    kappa_animate_inanimate_tcl_mean = np.load(f)
    
    
with open('acc_face_body_tcr_mean_sub6.npy', 'rb') as f:
    acc_face_body_tcr_mean = np.load(f)
    
with open('kappa_face_body_tcr_mean_sub6.npy', 'rb') as f:
    kappa_face_body_tcr_mean = np.load(f)
    
with open('acc_male_female_tcr_mean_sub6.npy', 'rb') as f:
    acc_male_female_tcr_mean = np.load(f)
    
with open('kappa_male_female_tcr_mean_sub6.npy', 'rb') as f:
    kappa_male_female_tcr_mean = np.load(f)


# ## Visualization before smoothing with gaussian kernel

# In[150]:


plt.figure(figsize=(12,5))
plt.plot(time, kappa_animate_inanimate_tcr_mean, color="c", label = "animate_inanimate")
plt.plot(time, kappa_face_body_tcr_mean, color="red", label = "face_body")
plt.plot(time, kappa_male_female_tcr_mean, color="green", label = "male_female")

plt.title("Kappa in time for right temporal channels")
plt.xlabel("time(ms)")
plt.ylabel("Kappa")
plt.legend()
plt.show()


# In[151]:


plt.figure(figsize=(12,6))
plt.plot(time, acc_animate_inanimate_tcr_mean, color="c", label = "animate_inanimate")
plt.plot(time, acc_face_body_tcr_mean, color="red", label = "face_body")
plt.plot(time, acc_male_female_tcr_mean, color="green", label = "male_female")

plt.title("Accuracy in time for right temporal channels")
plt.xlabel("time(ms)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ## Smoothing with gaussian kernel

# In[152]:


x = np.arange(0, data_num)
sigma = 5
smoothed_acc_animate_inanimate_tcr = np.zeros(acc_animate_inanimate_tcr_mean.shape)
smoothed_kappa_animate_inanimate_tcr = np.zeros(kappa_animate_inanimate_tcr_mean.shape)
smoothed_acc_face_body_tcr = np.zeros(acc_face_body_tcr_mean.shape)
smoothed_kappa_face_body_tcr = np.zeros(kappa_face_body_tcr_mean.shape)
smoothed_acc_male_female_tcr = np.zeros(acc_male_female_tcr_mean.shape)
smoothed_kappa_male_female_tcr = np.zeros(kappa_male_female_tcr_mean.shape)

for x_position in x:
    kernel = np.exp(-((x - x_position) ** 2) / (2 * sigma**2))
    kernel = kernel / sum(kernel)
    smoothed_acc_animate_inanimate_tcr[x_position] = sum(acc_animate_inanimate_tcr_mean * kernel)
    smoothed_kappa_animate_inanimate_tcr[x_position] = sum(kappa_animate_inanimate_tcr_mean * kernel)
    smoothed_acc_face_body_tcr[x_position] = sum(acc_face_body_tcr_mean * kernel)
    smoothed_kappa_face_body_tcr[x_position] = sum(kappa_face_body_tcr_mean * kernel)
    smoothed_acc_male_female_tcr[x_position] = sum(acc_male_female_tcr_mean * kernel)
    smoothed_kappa_male_female_tcr[x_position] = sum(kappa_male_female_tcr_mean * kernel)


# ## Visualizing data Vs smoothed

# In[153]:


fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(12,15))

axes[0][0].plot(time, acc_animate_inanimate_tcr_mean, color="blue", label = "animate_inanimate")
axes[0][0].plot(time, smoothed_acc_animate_inanimate_tcr, color="red", label = " smooth animate_inanimate")
axes[0][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[0][0].set_title("Accuracy animate_inanimate Vs smoothed in right temporal channels", size=10)
axes[0][0].legend()

axes[0][1].plot(time, kappa_animate_inanimate_tcr_mean, color="blue", label = "animate_inanimate")
axes[0][1].plot(time, smoothed_kappa_animate_inanimate_tcr, color="red", label = " smooth animate_inanimate")
axes[0][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[0][1].set_title("Kappa animate_inanimate Vs smoothed in right temporal channels", size=10)
axes[0][1].legend()

axes[1][0].plot(time, acc_face_body_tcr_mean, color="blue", label = "face_body")
axes[1][0].plot(time, smoothed_acc_face_body_tcr, color="red", label = " smooth face_body")
axes[1][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[1][0].set_title("Accuracy face_body Vs smoothed in right temporal channels", size=10)
axes[1][0].legend()

axes[1][1].plot(time, kappa_face_body_tcr_mean, color="blue", label = "face_body")
axes[1][1].plot(time, smoothed_kappa_face_body_tcr, color="red", label = " smooth face_body")
axes[1][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[1][1].set_title("Kappa face_body Vs smoothed in right temporal channels", size=10)
axes[1][1].legend()

axes[2][0].plot(time, acc_male_female_tcr_mean, color="blue", label = "male_female")
axes[2][0].plot(time, smoothed_acc_male_female_tcr, color="red", label = " smooth male_female")
axes[2][0].set(xlabel='time(ms)',ylabel='Accuracy')
axes[2][0].set_title("Accuracy male_female Vs smoothed in right temporal channels", size=10)
axes[2][0].legend()

axes[2][1].plot(time, kappa_male_female_tcr_mean, color="blue", label = "male_female")
axes[2][1].plot(time, smoothed_kappa_male_female_tcr, color="red", label = " smooth male_female")
axes[2][1].set(xlabel='time(ms)',ylabel='Kappa')
axes[2][1].set_title("Kappa male_female Vs smoothed in right temporal channels", size=10)
axes[2][1].legend()


# ## Visualizing smoothed data

# In[154]:


plt.figure(figsize=(12,5))
plt.plot(time, smoothed_kappa_animate_inanimate_tcr, color="c", label = "animate_inanimate")
plt.plot(time, smoothed_kappa_face_body_tcr, color="red", label = "face_body")
plt.plot(time, smoothed_kappa_male_female_tcr, color="green", label = "male_female")

plt.title("Smoothed Kappa in time in right temporal channels")
plt.xlabel("time(ms)")
plt.ylabel("Kappa")
plt.legend()
plt.show()


# In[155]:


plt.figure(figsize=(12,5))
plt.plot(time, smoothed_acc_animate_inanimate_tcr, color="c", label = "animate_inanimate")
plt.plot(time, smoothed_acc_face_body_tcr, color="red", label = "face_body")
plt.plot(time, smoothed_acc_male_female_tcr, color="green", label = "male_female")

plt.title("Smoothed Accuracy in time in right temporal channels")
plt.xlabel("time(ms)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




