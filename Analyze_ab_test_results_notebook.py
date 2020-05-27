#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, I will be working to understand the results of an A/B test run by an e-commerce website.  My goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv('ab_data.csv')


# b. Use the cell below to find the number of rows in the dataset.

# In[3]:


df.shape[0]


# In[4]:


df.info()


# In[5]:


# Sample Records
df.head(10)


# c. The number of unique users in the dataset.

# In[6]:


df['user_id'].nunique()


# d. The proportion of users converted.

# In[7]:


df['converted'].mean()


# e. The number of times the `new_page` and `treatment` don't match.

# ### new_page

# In[8]:


grp1=df.query("group == 'treatment' and landing_page == 'new_page'")
print('Total rows of Group_One : ',len(grp1))


# ### Old_Page

# In[9]:


grp2=df.query("group == 'control' and landing_page == 'old_page'")
print('Total rows of Group_Two : ',len(grp2))


# ### Total Rows - addition of Group_One & Group_Two

# In[10]:


print("New_Page vs Old_Page Dosn't Satisfied :: ",df.shape[0]-(len(grp1)+len(grp2)))


# f. Do any of the rows have missing values?

# In[11]:


df.isna().sum()


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[72]:


df2=df.copy()
df2.drop(df.query("(group == 'treatment' and landing_page == 'old_page') or (group == 'control' and landing_page == 'new_page')").index, inplace=True)


# ### Check all of the correct rows were removed

# In[73]:


df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[74]:


df2['user_id'].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[75]:


index_of = df2[df2.duplicated(['user_id'], keep=False)]['user_id'].unique()
print("User_Id Repeated in this dataset : ",index_of[0])


# c. What is the row information for the repeat **user_id**? 

# In[76]:


df2[df2.duplicated(['user_id'], keep=False)]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[77]:


df2.drop_duplicates(['user_id'],inplace=True)


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[78]:


df2['converted'].mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[79]:


df2[df2['group'] == 'control']['converted'].mean()


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[80]:


df2[df2['group'] == 'treatment']['converted'].mean()


# d. What is the probability that an individual received the new page?

# In[81]:


npr = len(df2.query("landing_page == 'new_page'"))
#find total no of rows
total_rows = df2.shape[0]
print("Total Rows : ",total_rows)
print('New page probability : ',npr/total_rows)


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# Probability of
#  1) individual converting
#  2) control group, 
#  3) treatment group
#  4) individual receiving a new page
# 
# **Based on above insufficient evidence to conclude that new treatment page leads to more conversions**

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# I will run the test utill<br>
# if **Null Hypothesis (new_page - old_page) = 0** <br>
# else **Alternative Hypothesis (new_page - old_page) > 0**  <br>

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[82]:


p_new = df2['converted'].mean()
p_new


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[83]:


p_old = df2['converted'].mean()
p_old


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[84]:


n_new = len(df2.query("landing_page == 'new_page'"))           
n_new


# d. What is $n_{old}$, the number of individuals in the control group?

# In[85]:


n_old = len(df2.query("landing_page == 'old_page'"))           
n_old


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[86]:


new_page_converted = np.random.binomial(n_new,p_new)
new_page_converted


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[87]:


old_page_converted = np.random.binomial(n_old,p_old)
old_page_converted


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[88]:


new_page_converted/n_new - old_page_converted/n_old


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[89]:


p_diffs = []
for _ in range(10000):
    new_page_converted = np.random.binomial(n_new,p_new)
    old_page_converted = np.random.binomial(n_old, p_old)
    p_diff = new_page_converted/n_new - old_page_converted/n_old
    p_diffs.append(p_diff)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[90]:


plt.xlabel('p_diff value')
plt.ylabel('Frequency')
plt.title('Plot of Simulated p_diffs');
plt.hist(p_diffs)


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[91]:


current_diff = df[df['landing_page'] == 'new_page']['converted'].mean() -  df[df['landing_page'] == 'old_page']['converted'].mean()
current_diff


# In[92]:


p_diffs = np.array(p_diffs)
(p_diffs > current_diff).mean()


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **Based on the p_value we reject the null hypothesis because the p_value is Significantly higher**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[93]:


conv_old = sum(df2.query("landing_page == 'old_page'")['converted'])
conv_new = sum(df2.query("landing_page == 'new_page'")['converted'])
n_old = len(df2.query("landing_page == 'old_page'"))
n_new = len(df2.query("landing_page == 'new_page'"))


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[94]:


z_score, p_value = sm.stats.proportions_ztest([conv_old, conv_new], [n_old, n_new], alternative='smaller')
print('z_score :: ',z_score)
print('p_value :: ',p_value)


# In[95]:


from scipy.stats import norm

print('Z_Score' , norm.cdf(z_score))


# In[96]:


# Based on 95% confidence level, we calculate
norm.ppf(1-(0.05))


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **z-score is less than the critical value which means we can't reject the null hypothesis it suggests there is no significant difference between old page and new page conversions**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Logistic Regression.**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[97]:


df2['intercept'] = 1

# df2 = pd.get_dummies(df2, prefix=['page'], columns=['group'],drop_first=True)

df2[['control', 'ab_page']]=pd.get_dummies(df2['group'])
df2.drop(labels=['control'], axis=1, inplace=True)
df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[98]:


logit = sm.Logit(df2['converted'],df2[['intercept' ,'ab_page']])
results = logit.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[99]:


stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# **The p-value is (0.190 > 0.05) indicates strong evidence that we can reject alternative hypothesis.<br>
#    where 0.190 is Obtained in this Part <br>
#    and 0.05 is Obtained in Part 2**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Based on provided data we performed logisitc regression and got some intel but if we possess some other information such as viewer wait time in the page , page performance data... think's like this features can help to provide more accurate prediction <br>**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[37]:


countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_new.head()


# In[40]:


df_new[['CA','UK','US']]=pd.get_dummies(df_new['country'])
df_new.head()


# In[42]:


mod = sm.Logit(df_new['converted'], df_new[['intercept', 'CA', 'UK']])
results = mod.fit()
results.summary()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[43]:


mod = sm.Logit(df_new['converted'], df_new[['intercept', 'CA', 'UK','ab_page']])
results = mod.fit()
results.summary()


# <a id='conclusions'></a>
# ## Conclusions
# 
# **Based on Part 2 and Part 3 we can conclude that we reject alternative hypothesis because the conversion rate shows no spike in new_page hence we accept Null Hypothesis <br>**
# 
# Suggestion
# 
# Based on this data we can do some feature engineering such as creating an new column's which state time & date from the timestamp and provide some intel saying in which time most user's access this page and based on which date such as week_end are week_days the user traffic hit the page
# 
# 

# In[ ]:





# In[ ]:




