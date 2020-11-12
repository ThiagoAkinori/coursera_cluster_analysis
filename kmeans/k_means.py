import pandas as pd
from scipy.spatial import distance
import numpy as np 


class KMeans():
    def __init__(self, dataset, k=3):
        self.df = dataset
        self.k = k
    
    def choose_centroids(self, dataset:pd.DataFrame, k: int):

        max_index = len(dataset.index)-1
        centroids = np.random.randint(max_index, size = k)
        return centroids

    def create_centroid_df (self, dataset, centroids):
        i = 0
        df_centroid = pd.DataFrame({},columns=dataset.columns.tolist()+['cluster'])
        for centroid in centroids:
            row = dataset[dataset.index==centroid].copy()
            row[ 'cluster'] = int(i)
            df_centroid = pd.concat([df_centroid, row])
            i = i+1
        return df_centroid

    def calculate_distance(self, row, centroid):
        return distance.euclidean(row, centroid)

    def create_distance_rows(self, dataset:pd.DataFrame, dataset_centroids, features):

        columns = features.tolist()
        centroids = dataset_centroids['cluster'].tolist()
        #print(dataset_centroids)
        for centroid in centroids:
            
            dataset['centroid_'+str(centroid)] = 0
            centroid_row = dataset_centroids.loc[dataset_centroids['cluster']==centroid, columns]

            for index, row in dataset.iterrows():
                dataset.loc[index, 'centroid_'+str(centroid)] = self.calculate_distance(row[columns], centroid_row)
        return dataset

    def choose_row_centroid(self, row, k =3):
        min_centroid_value = row['centroid_0']
        min_centroid = 0
        #print(row)
        for centroid in range(k):
            if min_centroid_value > row['centroid_'+str(centroid)]:
                min_centroid = centroid
                min_centroid_value = row['centroid_'+str(centroid)]
        
        return min_centroid

    def assign_cluster(self, dataset, k = 3):
        df = dataset.copy()
        df['cluster'] = 0

        for index, row in df.iterrows():
            df.loc[index, 'cluster'] = self.choose_row_centroid(row, 3)

        return df

    def calculate_new_centroid(self, dataset:pd.DataFrame, features:list):
        
        new_centroids = pd.DataFrame(pd.unique(dataset['cluster']), columns=['cluster'])

        for feature in features:
            
            new_centroid = pd.DataFrame(dataset.groupby('cluster')[feature].mean()).reset_index()
            #print(new_centroid)
            new_centroids = new_centroids.join(new_centroid, on='cluster', rsuffix='_'+str(feature))
            new_centroids = new_centroids.drop('cluster_'+str(feature), axis =1 )

        return new_centroids

    def k_means(self, dataset:pd.DataFrame, k=3):

        centroids = self.choose_centroids(dataset, k)

        df = dataset.copy()
        
        features = df.columns
        df_centroids = self.create_centroid_df(df, centroids)
        print(df_centroids)
        for i in range(100):
            
            df_centroid = self.create_distance_rows(df, df_centroids, features)
            df_clusters = self.assign_cluster(df_centroid, k)
            
            df_centroids = self.calculate_new_centroid(df_clusters,features)
            
        return df_clusters

    def fit_predict(self):
        return self.k_means(self.df, self.k)


df = pd.read_csv('places.txt', sep=',', header=None)
kmeans = KMeans(df, 3)

df_clusters = kmeans.fit_predict()
df_clusters = df_clusters.reset_index()
print(df_clusters[['index', 'cluster']])

df_clusters[['index', 'cluster']].to_csv('clusters.txt', sep=' ', index=False,header=False)