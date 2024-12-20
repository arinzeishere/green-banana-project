#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy.stats import chi2, chi2_contingency
from statsmodels.api import GLM
from statsmodels.genmod import families
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc

from pycaret.classification import *

mydata = pd.read_csv("file path")




# In[3]:


num_features = df.shape[1]
print("Number of features:", num_features)
num_observations = df.shape[0]
print("Number of observations:", num_observations)

missing_percentage = df.isnull().mean() * 100
print("Percentage of missing data for each column:")
print(missing_percentage)




# In[5]:


mydata
num_observations = mydata.shape[0]
print("Number of observations:", num_observations)


# In[6]:


# check data types
mydata.dtypes


# In[6]:


# Define colors for 'Churned' variable
colors = {0: 'blue', 1: 'red'}

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(mydata['Net-spend'], mydata['Tenure'], c=mydata['Churned'].map(colors), alpha=0.5)

# Add labels and title
plt.xlabel('Net-spend')
plt.ylabel('Tenure')
plt.title('Scatter plot of Net-spend vs. Tenure colored by Churned')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Churned: 1'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Churned: 0')],
           title='Churned', loc='center left', bbox_to_anchor=(1, 0.5))

# Show plot
plt.show()


# In[7]:


# Calculate average Net-Spend for each Churned category
avg_netspend = mydata.groupby('Churned')['Net-spend'].mean()

# Create bar plot
plt.figure(figsize=(8, 6))
avg_netspend.plot(kind='bar', color=['blue', 'red'])
plt.xlabel('Churned')
plt.ylabel('Average Net-Spend')
plt.title('Average Net-Spend by Churned')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
plt.show()


# In[8]:


from pycaret.classification import *
clf1= setup(mydata, target='Churned', ignore_features = ['customerID'])


# In[7]:


mydata = mydata.rename(columns={'Number-orders': 'Numberorders', 'Number-subscriptions': 'Numbersubscriptions', 'Net-spend': 'Netspend'})

# Define the formula for the logistic regression model
formula_final = ("Churned ~ Numbersubscriptions + Numberorders  + Netspend + Tenure")

# Fit the logistic regression model
logit_model = GLM.from_formula(formula=formula_final, family=families.Binomial(), data=mydata).fit()

# Print summary of the model
print(logit_model.summary())


# In[8]:


#PREDICTIONS BASED ON THE MODEL
# Get predicted values
predicted_values = logit_model.predict(mydata)

print("Predicted probabilities:", predicted_values)

# SCATTERPLOT OF TIME AND PREDICTED VALUES
# Add the predicted values as a new column in mydata
mydata['predicted_values'] = predicted_values

dfg= pd.DataFrame(predicted_values.describe())
print(dfg.to_latex())



# In[9]:


# OBTAINING THE CONFUSION MATRIX

# Set the threshold for classification
threshold = 0.5

# Convert predicted probabilities to binary predictions
predicted_classes = (predicted_values > threshold).astype(int)

# Create the confusion matrix
conf_matrix = confusion_matrix(mydata['Churned'], predicted_classes)

# Calculate the error rate
error_rate = 1 - np.trace(conf_matrix) / np.sum(conf_matrix)

# Print the confusion matrix
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'],
                              columns=['Predicted 0', 'Predicted 1'])
print("\nConfusion Matrix:")
print(conf_matrix_df)

# Print the error rate
print("\nError Rate:", error_rate)


# In[10]:


# PERFORM SENSITIVITY


# Vary the threshold from 0.1 to 0.9
thresholds = np.arange(0.1, 1, 0.1)

# Initialize lists to store error rates, sensitivities, and specificities
error_rates = []
sensitivities = []
specificities = []

for threshold in thresholds:
    # Convert predicted probabilities to binary predictions using the current threshold
    predicted_classes = (predicted_values > threshold).astype(int)

    # Create the confusion matrix
    conf_matrix = confusion_matrix(mydata['Churned'], predicted_classes)

    # Calculate the error rate
    error_rate = 1 - np.trace(conf_matrix) / np.sum(conf_matrix)

    # Append error rate to the list
    error_rates.append(error_rate)

    # Calculate sensitivity (true positive rate)
    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])

    # Append sensitivity to the list
    sensitivities.append(sensitivity)

  

    # Calculate specificity (true negative rate)
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])


    # Append specificity to the list
    specificities.append(specificity)

# Print sensitivity and specificity values for each threshold in LaTeX format
for i, threshold in enumerate(thresholds):
    print(f"Threshold: {threshold:.1f}, Sensitivity: {sensitivities[i]:.4f}, Specificity: {specificities[i]:.4f}".replace(",", " &") + " \\\\")

# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(thresholds, error_rates, marker='o', color='green')
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. Threshold')
plt.grid(False)
plt.show()



# In[13]:


#PERFORMING ROC
# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(mydata['Churned'], predicted_values)

# Calculate the AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specifity')
plt.ylabel('Sensitivity')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()


# In[11]:


# PERFORMING KFOLD CROSS VALIDATION
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

# Define the features (X) and target variable (y)
X = mydata.drop(columns=['Churned'])
y = mydata['Churned']

# Initialize an empty list to store the accuracies for each fold
accuracies = []

# Initialize an empty list to store the error rates for each fold
error_rates = []

# Initialize an empty list to store the confusion matrices for each fold
conf_matrices = []

# Initialize an empty list to store the ROC AUC scores for each fold
roc_auc_scores = []

# Initialize K-Fold Cross-Validation with k=5
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform K-Fold Cross-Validation
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training data
    logit_model = GLM.from_formula(formula=formula_final, family=families.Binomial(),
                                   data=pd.concat([X_train, y_train], axis=1)).fit()

    # Print the logit summary for this fold
    print("\nLogit Summary for Fold", fold)
    print(logit_model.summary())

    # Get the predicted probability for the test samples
    predicted_prob = logit_model.predict(X_test)

    # Convert the predicted probabilities to binary predictions
    threshold = 0.5
    predicted_classes = (predicted_prob > threshold).astype(int)

    # Calculate the accuracy for this fold
    accuracy = accuracy_score(y_test, predicted_classes)
    accuracies.append(accuracy)

    # Calculate the confusion matrix for this fold
    conf_matrix = confusion_matrix(y_test, predicted_classes)
    conf_matrices.append(conf_matrix)

    # Calculate the error rate for this fold
    error_rate = 1 - np.trace(conf_matrix) / np.sum(conf_matrix)
    error_rates.append(error_rate)

    # Calculate the ROC AUC score for this fold
    roc_auc = roc_auc_score(y_test, predicted_prob)
    roc_auc_scores.append(roc_auc)
    print("ROC AUC score for Fold", fold, ":", roc_auc)

    # Print the confusion matrix for this fold
    print("\nConfusion Matrix for Fold", fold)
    print(conf_matrix)

# Create a summary table for all folds
summary_table = pd.DataFrame({
    'Fold': range(1, kf.get_n_splits() + 1),
    'Accuracy': accuracies,
    'Error Rate': error_rates,
    'ROC AUC Score': roc_auc_scores,
    'Confusion Matrix': conf_matrices
})

# Print the summary table
print("\nSummary Table:")


summary_table

# Calculate the average accuracy across all folds
avg_accuracy = np.mean(accuracies)
print("Average Accuracy:", avg_accuracy)

# Calculate the average ROC AUC score across all folds
avg_roc_auc = np.mean(roc_auc_scores)
print("Average ROC AUC Score:", avg_roc_auc)



# In[15]:


# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, cmap='viridis', fmt='g')

plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Visualizing the Confusion Matrix in a Heat-Map')
plt.show()


# In[ ]:


from pycaret.classification import *
selected_columns = ['Numbersubscriptions','Churned','Numberorders','Netspend', 'Tenure']
selected_data = mydata[selected_columns]


# Step 1: Setup the Environment
s = setup(selected_data , target='Churned')

# Step 2: Compare Models
best_model = compare_models(sort='AUC')

# Step 3: Create Model
final_model = create_model(best_model)

# Step 4: Evaluate Model
evaluate_model(final_model)


# In[ ]:


# Calculate the total Netspend
total_netspend = mydata['Netspend'].sum()

# Calculate the number of customers
num_customers = len(mydata)

# Calculate the CLV
clv = total_netspend / num_customers

print("Customer Lifetime Value (CLV):", clv)


# Calculate the total number of churned customers in your dataset
actual_churned_total = mydata['Churned'].sum()

# Calculate the accuracy rate from the error rate
accuracy_rate = 1 - error_rate

# Calculate the number of churned customers that your model predicted correctly
predicted_correct = round(accuracy_rate * actual_churned_total)

print("Actual number of churned customers:", actual_churned_total)
print("Number of churned customers predicted correctly:", predicted_correct)




# In[15]:


# Calculate 5% of the number of churned customers predicted correctly
retained_percentage = 0.05
retained_customers = round(predicted_correct * retained_percentage)

# Calculate the potential savings
potential_savings = retained_customers * clv

print("Potential savings:", potential_savings)


# In[16]:


# Calculate the total number of customers
total_customers = len(mydata)

# Calculate the number of customers who churned
churned_customers = mydata['Churned'].sum()

# Calculate the churn rate
churn_rate = churned_customers / total_customers

print("Churn Rate:", churn_rate)


# In[17]:


# Calculate the churn rate
churn_rate = churned_customers / total_customers

# Calculate the customer lifetime
customer_lifetime = 1 / churn_rate

print("Customer Lifetime:", customer_lifetime)



# In[18]:


sum_tenure = mydata['Tenure'].sum()
total_customers = len(mydata)
average_lifetime = sum_tenure / total_customers
print("Average Lifetime:", average_lifetime)


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns

# Group the data by 'Churned' and calculate the mean for each variable
grouped_data = mydata.groupby('Churned')[['Numbersubscriptions', 'Numberorders', 'Netspend', 'Tenure']].mean().reset_index()

# Set the figure size
plt.figure(figsize=(12, 8))

# Plot bar plots for each variable
plt.subplot(2, 2, 1)
sns.barplot(x='Churned', y='Numbersubscriptions', data=mydata, estimator=sum, ci=None)
plt.title('Total Number of Subscriptions')

plt.subplot(2, 2, 2)
sns.barplot(x='Churned', y='Numberorders', data=mydata, estimator=sum, ci=None)
plt.title('Total Number of Orders')

plt.subplot(2, 2, 3)
sns.barplot(x='Churned', y='Netspend', data=mydata, estimator=sum, ci=None)
plt.title('Total Netspend')

plt.subplot(2, 2, 4)
sns.barplot(x='Churned', y='Tenure', data=mydata, estimator=sum, ci=None)
plt.title('Total Tenure')

plt.tight_layout()
plt.show()


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt

# Filter the data for churned customers
churned_data = mydata[mydata['Churned'] == 1]

# Create a pairplot for all variables
sns.pairplot(churned_data[['Numbersubscriptions', 'Numberorders', 'Netspend', 'Tenure']])
plt.show()

# Create boxplots for each variable
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
sns.boxplot(x='Churned', y='Numbersubscriptions', data=mydata, ax=axs[0, 0])
sns.boxplot(x='Churned', y='Numberorders', data=mydata, ax=axs[0, 1])
sns.boxplot(x='Churned', y='Netspend', data=mydata, ax=axs[1, 0])
sns.boxplot(x='Churned', y='Tenure', data=mydata, ax=axs[1, 1])
plt.show()

