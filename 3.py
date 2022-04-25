import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

#PLOTTING THE BAR GRAPH#
#Reading the CSV file using pandas
data = pd.read_csv("iris.csv")
#Plotting bar graph with frequencies
df=pd.DataFrame(data)
X=list(df.iloc[:,5])# Species
set_X=set(X)
l_X=list(set_X)
vesi_count=0
seto_count=0
virg_count=0
for i in X :
    if(i=="Iris-versicolor"):
        vesi_count=vesi_count+1
    elif(i=="Iris-setosa"):
        seto_count=seto_count+1
    elif(i=="Iris-virginica"):
        virg_count=virg_count+1
count_list=[vesi_count,seto_count,virg_count]
plt.bar(l_X,count_list,color='red')
plt.xlabel("Species")
plt.ylabel("Species count")
plt.title("Species frequency")
plt.show()

#PLOTTING SCATTER PLOT

#Preprocessing of the data#
#Appling PCA to get 2 principal compnents
from sklearn.preprocessing import StandardScaler
#Removing species column
x  = data.iloc[:,:-1]
print(x)
x = StandardScaler().fit_transform(x) #normalising the data
#After normalisation mean = 0 and std dvn = 1
print("Mean",np.mean(x))
print("STD Dvn : ",np.std(x))

#Calculating PCA to get 2 components
from sklearn.decomposition import PCA
pca_iris =  PCA(n_components=2)
print(pca_iris)
principal_components_iris = pca_iris.fit_transform(x)
print(principal_components_iris)
#Creating a data frame with PCA Values
principal_iris_DF = pd.DataFrame(data=principal_components_iris,
        columns = ['PC1','PC2'])
principal_iris_DF.tail()
print(principal_iris_DF)
#Plotting from the principal Components
plt.figure()
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
targets = ['Iris-setosa','Iris-versicolor','Iris-virginica']
colors = ['r','g','b']
for target,color in  zip(targets,colors):
    indices_to_Keep = data['Species'] == target
    plt.scatter(principal_iris_DF.loc[indices_to_Keep,'PC1'],
    principal_iris_DF.loc[indices_to_Keep,"PC2"],c = color)
plt.show()
  
  
def plot_hist(Att_name,Hcolor,title,xtitle):
    data[Att_name].hist(color = Hcolor)
    plt.xlabel(xtitle)
    plt.ylabel('frequency')
    plt.title(title)
    plt.show()

plot_hist('SepalLengthCm','r','Sepal length distribution','sepal length(cm)')
plot_hist('SepalWidthCm','g','Sepal Width distribution','sepal width(cm)')
plot_hist('PetalLengthCm','b','Petal length distribution','petal length(cm)')
plot_hist('PetalWidthCm','grey','Petal width distribution','petal width(cm)')