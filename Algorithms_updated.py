import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Placing a seed so the values dont change everytime I make a change
np.random.seed(4)


################ PRE PROCESSING DATA #####################


##Extracting the animals file
animals=pd.read_csv("animals",sep=" ",header=None)
#print(animals)

##Extracting the veggies file
veggies=pd.read_csv("veggies",sep=" ",header=None)
#print(len(veggies))
   
##Extracting the fruits file
fruits=pd.read_csv("fruits",sep= " ", header=None)
#print(len(fruits))
    
##Extracting the countries file
countries=pd.read_csv("countries",sep= " ",header=None)
#print(len(countries))
    
##Adding a column named Category
animals['Category'] = 'animals'
veggies['Category'] = 'veggies'
fruits['Category'] = 'fruits'
countries['Category'] = 'countries'
#print(animals)
#print(veggies)
#print(fruits)
#print(countries)

     
## Concatinating all files together
data_concat=pd.concat([animals,veggies,fruits,countries],ignore_index=True)
#print(len(data_concat))


maxAni=maxVeg=maxFruits=maxCountries=0
## Caluculating till which index each category is stored uptil
maxAni=data_concat.index[data_concat['Category']=='animals'][-1]
maxVeg=data_concat.index[data_concat['Category']=='veggies'][-1]
maxFruits=data_concat.index[data_concat['Category']=='fruits'][-1]
maxCountries=data_concat.index[data_concat['Category']=='countries'][-1]
# print(maxAni)
# print(maxVeg)
# print(maxFruits)
# print(maxCountries)
        

## Converting each labels to 0-3 where        
## 0-animals,1-veggies,2-fruits,3-countries
labels = (pd.factorize(data_concat.Category)[0])
#print(labels)
#print(data_concat)
#dataset=data_concat[0,:]

## Dropping the first and last column to use for implementation of K-means and K-medians
dataset = data_concat.drop([0,'Category'], axis=1).values
# print(len(dataset))
#print(dataset)


## Assignment Phase

##############3 CALCULATION FOR QUESTION1 ##################
def assign(centroids, x, clusters,distance_measure):
   
    k = len(centroids)
    dist = np.zeros([x.shape[0], k])
    if (distance_measure=="kMeans"):
        for j in range(len(centroids)):
            
            ##Euclidean distance calculation
            dist[:,j] = (np.linalg.norm(x-centroids[j],axis=1)**2)
            ##Taking the shortest distance as the cluster assignment
            clusters = np.argmin(dist, axis=1)
            
    elif (distance_measure=="kMedians"):
        for j in range(len(centroids)):
            
         ##Manhattan distance calculation
          dist[:,j]=(np.sum(np.abs(x-centroids[j]),axis=1))
          ##Taking the shortest distance as the cluster assignment
          clusters = np.argmin(dist, axis=1) 
           
    return clusters

    
def kAlgo(k,dataset,maxIter,algo_used,Normalisation=False):
    centroids=[]
    
    #For l2 regularisation - normalisation is required or not
    if Normalisation==True:
        #Assigning the nomralised function to the dataset
        dataset=dataset/np.linalg.norm(dataset)
    
    

    temp=np.random.randint(dataset.shape[0],size=k)
    
    print("-------------------------------------------------------------")
    print("\t\t\tK ALGORITHM FOR ", algo_used, "\n")
    print('-Initial centroids -' ,temp)
    
    for index in temp:
        centroids.append(dataset[index])
   
    ##Make an old centroid to do calculation before and after updation of centroid position 
    centroids_old=np.zeros(np.shape(centroids))
   
    ##Use the values of centroid for the intial calculation of centroid_new
    centroids_new=np.copy(centroids)
    
    ##Calculate the objective function by checking the intial difference between old centroid and new centroid values
    objFun=np.linalg.norm(centroids_new-centroids_old)
    
    
    #print("Initial objective function " ,objFun)

    ## Create a cluster to hold the loaction of where each datapoint would fall into
    cluster_group=np.zeros(dataset.shape[0])
    
    ##Use num_error to find out how many iterations took place before the centroid found the desired location to be placed
    num_errors=0
    
    for i in range(maxIter):
        
        num_errors=num_errors+1
        ass_clusters = assign(centroids_new, dataset, cluster_group,algo_used)
        centroids_old=np.copy(centroids_new)
        
        ## Optimisation Phase
        
        if(algo_used=="kMeans"):
            for x in range(k):
                ## Using the np function to calculate the mean for K-means
                centroids_new[x]=np.mean(dataset[ass_clusters==x],axis=0)
                
        elif (algo_used=="kMedians"):
           for x in range(k):
               ## Using the np function to calculate the median for K-medians
               centroids_new[x]=np.median(dataset[ass_clusters==x],axis=0)
            
        ## Calculation of new objective function
        new_objFun=np.linalg.norm(centroids_new-centroids_old)
        

        if new_objFun < objFun:
            #Checking if the calculated ObjFun has changed after distance calculation
            objFun=new_objFun
            #print("Changed Obj functions " ,objFun) 
        else:
            break
    
        
    
    print("Final Results of Clustering with", algo_used,"Distance Measurement")
    print("Number of Clusters: ", k)
    print("Number of Updates: ", num_errors)
    ##Final clusters gives the indices of which cluster each datapoint lies in 
    ass_clusters=np.array(ass_clusters)
    print("Final clusters: ",ass_clusters)
    #print(len(ass_clusters))
    ## Final centroid- gives the cluster values
    centroids_new=np.array(centroids_new)
    # print("Final Centroid Location: ",centroids_new)
    print("-------------------------------------------------------------------------------------")

    return dataset, ass_clusters, centroids_new
      
############ CALCULATION FOR QUESTION 3-6 ################# 
        
def getMetrics(dataset, clusters, centroids, labels):
    
    avg_precision=np.zeros(dataset.shape[0])
    avg_recall=np.zeros(dataset.shape[0])
    avg_fscore=np.zeros(dataset.shape[0])
    
    # print('metrics')
    #print(clusters)
    for i in range(len(dataset)):
        ## Assigning the new found cluster location and labels of each category to a variable
        current_cluster=clusters[i]
        current_category=labels[i]
        
        ## Creating a new list for reference and adding values that match into it
        new_data=[]
        
        for j in range(len(clusters)):
            if(clusters[j]==current_cluster):
                new_data.append(j)
        ##Shows all the datapoints in the current cluster
        
        count=0
        for x in new_data:
            if labels[x] == current_category:
                ##Gives how many of the same datapoints are together and increases a counter for it
                count=count+1
        
        ##Calculation of individual precision, recall and f_score
        avg_precision[i]=count/len(new_data)
        avg_recall[i]=count/len(labels[labels==current_category])
        avg_fscore[i]=(2*avg_precision[i]*avg_recall[i])/(avg_precision[i]+avg_recall[i])
    
    ## Calculating the mean of all the individual precision,recall and f_score    
    mean_precision=np.mean(avg_precision)
    mean_recall=np.mean(avg_recall)
    mean_fscore=np.mean(avg_fscore)
        
    print('Precision ',mean_precision)
    print('Recall ',mean_recall)
    print('Fscore ',mean_fscore)
        
        
        
    return mean_precision,mean_recall,mean_fscore
    
def plotting(precision,recall,fScore,k,algo_used,normalisation=False):
    plt.plot(k,precision,label="Precision")
    plt.plot(k,recall,label="Recall")
    plt.plot(k,fScore,label="fScore")
    title="Clustering with "+ algo_used    
    if normalisation==True:
        title+=" with Normalisation"
    plt.title(title)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

precision = []
recall = []
fScore = []
kList = []
for k in range(1,10):
    kList.append(k)
    x, clusters, centroids =kAlgo(k,dataset,100,"kMedians")
    
    tot_precision,avg_recall,avg_fscore=getMetrics(x,clusters,centroids, labels)
    
    precision.append(tot_precision)
    recall.append(avg_recall)
    fScore.append(avg_fscore)
plotting(precision,recall,fScore,kList,"kMedians")

precision = []
recall = []
fScore = []
kList = []
for k in range(1,10):
    kList.append(k)
    x, clusters, centroids =kAlgo(k,dataset,100,"kMedians",True)
    
    tot_precision,avg_recall,avg_fscore=getMetrics(x,clusters,centroids, labels)
    
    precision.append(tot_precision)
    recall.append(avg_recall)
    fScore.append(avg_fscore)
plotting(precision,recall,fScore,kList,"kMedians", True)

precision = []
recall = []
fScore = []
kList = []
for k in range(1,10):
    kList.append(k)
    x, clusters, centroids =kAlgo(k,dataset,100,"kMeans")
    
    tot_precision,avg_recall,avg_fscore=getMetrics(x,clusters,centroids, labels)
    
    precision.append(tot_precision)
    recall.append(avg_recall)
    fScore.append(avg_fscore)
plotting(precision,recall,fScore,kList,"kMeans")

precision = []
recall = []
fScore = []
kList = []
for k in range(1,10):
    kList.append(k)
    x, clusters, centroids =kAlgo(k,dataset,100,"kMeans",True)
    
    tot_precision,avg_recall,avg_fscore=getMetrics(x,clusters,centroids, labels)
    
    precision.append(tot_precision)
    recall.append(avg_recall)
    fScore.append(avg_fscore)
plotting(precision,recall,fScore,kList,"kMeans", True)
    
##### For calculation of QUESTION 1 & 2 ####################
# k,clusters=kAlgo(4,dataset,100,"kMeans")
# k,clusters= kAlgo(4,dataset,100,"kMedians")

            
