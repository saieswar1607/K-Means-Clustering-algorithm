# Implementation of K-Means Clustering Algorithm
## Aim
To write a python program to implement K-Means Clustering Algorithm.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation

## Algorithm:

### Step1:
<br>
Import pandas.



### Step2
<br>
Import matplotlib.pyplot.

### Step3
<br>

Import sklearn.cluster from KMeans module

### Step4
<br>
Import seaborn

### Step5
<br>
Import warnings

###step6
<br>
Declare warnings.filerwarning with ignore as argument

###step7
<br>
Declare a variable x1 and read a csv file(clustering.csv) in it.

###step8
<br>
Declare a variable x2 as index of x1 with arguments ApplicantIncome and LoanAmount.

###step9
<br>
Display x1.head(2) and x2.head(2).

###step10
<br>
Declare a variable x and store x2.values.

###step11
<br>
Declare sns.scatterplot for ApplicantIncome and LoanAmount by indexing

###step12
<br>
Plot Income , Loan and display them.

###step13
<br>
Declare a variable kmean = KMean(n_cluster_centers_) and execute kmean.fit(x).

###step14
<br>
Display kmean.cluster)centers

###step15
<br>
Display kmean.labels_

###step16
<br>
Declare a variable predcited_class to kmean.predict([[]]) and give two arguments in it.

###step17
<br>
Display the predicted_class



## Program:
~~~
#Developed by: Kandukuri sai eswar
#Register number: 212221240020
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
x1 = pd.read_csv('clustering.csv')
print(x1.head(2))
x2 = x1.loc[:,['ApplicantIncome','LoanAmount']]
print(x2.head(2))

x=x2.values
sns.scatterplot(x[:,0],x[:,1])
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()

kmean = KMeans(n_clusters=4)
kmean.fit(x)

print('Cluster Centers:',kmean.cluster_centers_)
print('Labels:',kmean.labels_)

predicted_class = kmean.predict([[9200,110]])
print('The cluster group for Applicant Income 9000 and Loanamount',predicted_class)







~~~
## Output:
![](sai.png)

### Insert your output

<br>

## Result
Thus the K-means clustering algorithm is implemented and predicted the cluster class using python program.
