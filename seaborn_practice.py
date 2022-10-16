#!/usr/bin/env python
# coding: utf-8

# # Scatter plot--> scatter plot shows thr relation ship between two continous features

# In[2]:


#scatter plot line up a set of two continous features and plots them out as coordinates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("dm_office_sales.csv")


# In[5]:


df.head()


# In[16]:


plt.figure(figsize = (12,4),dpi = 200)
sns.scatterplot(x = "salary", y = "sales",data = df,hue = "level of education",size = "salary",style = "work experience")
#plt.savefig("myplot.jpg")


# # Distplot--> dist plots display a single continous feature and help visualize properties such as deviation and average values.

# In[17]:


#There are 3 main distplot types
#Rugplot
#Histogram
#KDE plot


# In[18]:


#Rugplot --> add a tick for every single value,for y-axis there is no meaning at all
#if we want to count how many ticks there are per various x ranges , we can crewate a histogram.
#KDE--> it is a way of estimating a continous probability curve for a finite data sample


# In[23]:


plt.figure(figsize = (5,8),dpi = 100 )
sns.rugplot(x = 'salary',data = df,height = 0.5) #height is 50% of y-axis


# In[31]:


#do not use distplot, it will depreciated
sns.set(style = 'darkgrid') #or whitegrid oe white or dark
sns.displot(data = df, x = "salary",bins = 20,color = 'red',edgecolor = 'blue',linewidth = 4)


# In[36]:


sns.histplot(data = df,x = "salary",kde = True)


# In[37]:


sns.kdeplot(data = df, x = 'salary')


# In[38]:


np.random.seed(42)
sample_ages = np.random.randint(0,100,200)
  


# In[39]:


sample_ages


# In[41]:


sample_ages = pd.DataFrame(sample_ages,columns = ["age"])


# In[43]:


sample_ages


# In[44]:


sns.rugplot(data = sample_ages,x = "age")


# In[47]:


sns.displot(data = sample_ages,x = 'age',bins = 30,kde = True)


# In[52]:


sns.kdeplot(data = sample_ages,x = 'age',bw_adjust = 0.1,shade = True)


# # Categorical plots --> it will display a statistical metrics per a category. for example  mean value per category or a count of the number of rows per category

# In[53]:


#Ther are two main types of plots 
#countplot-->counts number of rows per category
#barplot--> general form of displaying any chosen metric per category


# In[54]:


df.head()


# In[55]:


df['division'].value_counts()


# In[59]:


plt.figure(figsize = (10,4),dpi = 200)
sns.countplot(data = df,x = 'level of education',hue = 'division')


# In[64]:


sns.barplot(data = df,x = 'level of education',y = 'salary',estimator = np.mean,ci = 'sd',hue = 'division')  #sd-->standard deviation
plt.legend(bbox_to_anchor = (1.05,1))


# In[66]:


#distribution with in categories
#boxplot
#violinplot
#swarmplot
#boxenplot


# In[69]:


#boxplot --> it displays the distribution of a continous variable, it does through the sue of quartiles, quartiles seperates out the data into 4 equal number of data points

#violinplot--> it plays a similar role as the boxplot, it displays the probability density across the data using a kde

#swarmplot--> it simply shows all the data points in the distribution,foe very large datasets , it wont show all the points, but will display the general distribution of them.

#boxenplot--> it is designed as an expansion upon the normal boxplot
#boxenplot showing letter-value-quantiles to display against a standard boxplot


# In[71]:


df = pd.read_csv("StudentsPerformance.csv")
df.head()


# In[76]:


plt.figure(figsize= (10,4),dpi = 200)
sns.boxplot(data = df,y = 'reading score',x = 'test preparation course')


# In[80]:


plt.figure(figsize= (10,4),dpi = 200)
sns.boxplot(data = df,y = 'reading score',x = 'parental level of education',hue = 'test preparation course')
plt.legend(bbox_to_anchor=(1.05,0.5))


# In[83]:


plt.figure(figsize= (10,4),dpi = 200)
sns.violinplot(data = df,y = 'reading score',x = 'parental level of education',hue = 'test preparation course')
plt.legend(bbox_to_anchor=(1.05,0.5))


# In[84]:


plt.figure(figsize= (10,4),dpi = 200)
sns.violinplot(data = df,y = 'reading score',x = 'parental level of education',hue = 'test preparation course',split = True)
plt.legend(bbox_to_anchor=(1.05,0.5))


# In[95]:


#swarmplot
plt.figure(figsize = (8,4),dpi = 200)
sns.swarmplot(data = df, x = 'math score',y = 'gender',size = 2,hue = 'test preparation course',dodge = True)
plt.legend(bbox_to_anchor=(1.35,0.5))


# In[97]:


sns.boxenplot(x = 'math score', y ='test preparation course',data = df, hue = 'gender')


# # Comparison plots --> these are essentially 2 dimensional versions of the plots we have learned about so far.

# In[98]:


#there are 2 main plots
#jointplot
#pairplot


# In[99]:


df.head()


# In[102]:


sns.jointplot(data= df,x = 'math score', y = 'reading score',kind = 'hex') #or kind = 'scatter'


# In[107]:


sns.jointplot(data= df,x = 'math score', y = 'reading score',kind = 'scatter',hue = 'gender') #or kind = 'scatter'


# In[111]:


sns.pairplot(data = df,hue = 'gender',diag_kind = 'hist')


# # Grid plots --> seaborn grids calls use Matplotlib subplots() to automatically create a grid based off a categorical column

# In[112]:


#instead of passing in a spevific number of cols or rows for the subplots, we can simply pass in the name of the column and seaborn will automatically map the subplots grid.


# In[113]:


df.head()


# In[115]:


#catplot
sns.catplot(x = 'gender', y = 'math score',kind = 'box',data = df,row = 'lunch') #or col = 'lunch'


# In[122]:


g = sns.PairGrid(df,hue = 'gender')
g = g.map_lower(sns.scatterplot)
g = g.map_diag(sns.histplot)
g = g.map_upper(sns.kdeplot)
g = g.add_legend()


# # Matrix plot --> visual equivalent of displaying a pivot table,it displays all the data passed in, visualiszing all the numeric values in a DataFrame.

# In[123]:


#There are 2 types of matrix plots
#heatmap
#clustermap


# In[138]:


df = pd.read_csv("country_table.csv")
df.head()


# In[139]:


df = df.set_index('Countries')


# In[140]:


df


# In[148]:


plt.figure(dpi = 200)
sns.heatmap(df.drop('Life expectancy',axis = 1),annot = True,cmap = 'viridis')


# In[149]:


plt.figure(dpi = 200)
sns.clustermap(df.drop('Life expectancy',axis = 1),annot = True,cmap = 'viridis')

