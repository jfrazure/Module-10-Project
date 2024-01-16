# Module-10-Project






Python 3 (ipykernel)
Module 10 Application

Challenge: Crypto Clustering

In this Challenge, you’ll combine your financial Python programming skills with the new unsupervised learning skills that you acquired in this module.

The CSV file provided for this challenge contains price change data of cryptocurrencies in different periods.

The steps for this challenge are broken out into the following sections:

Import the Data (provided in the starter code)
Prepare the Data (provided in the starter code)
Find the Best Value for k Using the Original Data
Cluster Cryptocurrencies with K-means Using the Original Data
Optimize Clusters with Principal Component Analysis
Find the Best Value for k Using the PCA Data
Cluster the Cryptocurrencies with K-means Using the PCA Data
Visualize and Compare the Results
Import the Data

This section imports the data into a new DataFrame. It follows these steps:

Read the “crypto_market_data.csv” file from the Resources folder into a DataFrame, and use index_col="coin_id" to set the cryptocurrency name as the index. Review the DataFrame.

Generate the summary statistics, and use HvPlot to visualize your data to observe what your DataFrame contains.

Rewind: The Pandasdescribe()function generates summary statistics for a DataFrame.


# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
​
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    Path("Resources/crypto_market_data.csv"),
    index_col="coin_id")
​
# Display sample data
df_market_data.head(10)
price_change_percentage_24h	price_change_percentage_7d	price_change_percentage_14d	price_change_percentage_30d	price_change_percentage_60d	price_change_percentage_200d	price_change_percentage_1y
coin_id							
bitcoin	1.08388	7.60278	6.57509	7.67258	-3.25185	83.51840	37.51761
ethereum	0.22392	10.38134	4.80849	0.13169	-12.88890	186.77418	101.96023
tether	-0.21173	0.04935	0.00640	-0.04237	0.28037	-0.00542	0.01954
ripple	-0.37819	-0.60926	2.24984	0.23455	-17.55245	39.53888	-16.60193
bitcoin-cash	2.90585	17.09717	14.75334	15.74903	-13.71793	21.66042	14.49384
binancecoin	2.10423	12.85511	6.80688	0.05865	36.33486	155.61937	69.69195
chainlink	-0.23935	20.69459	9.30098	-11.21747	-43.69522	403.22917	325.13186
cardano	0.00322	13.99302	5.55476	10.10553	-22.84776	264.51418	156.09756
litecoin	-0.06341	6.60221	7.28931	1.21662	-17.23960	27.49919	-12.66408
bitcoin-cash-sv	0.92530	3.29641	-1.86656	2.88926	-24.87434	7.42562	93.73082

# Generate summary statistics
df_market_data.describe()
price_change_percentage_24h	price_change_percentage_7d	price_change_percentage_14d	price_change_percentage_30d	price_change_percentage_60d	price_change_percentage_200d	price_change_percentage_1y
count	41.000000	41.000000	41.000000	41.000000	41.000000	41.000000	41.000000
mean	-0.269686	4.497147	0.185787	1.545693	-0.094119	236.537432	347.667956
std	2.694793	6.375218	8.376939	26.344218	47.365803	435.225304	1247.842884
min	-13.527860	-6.094560	-18.158900	-34.705480	-44.822480	-0.392100	-17.567530
25%	-0.608970	0.047260	-5.026620	-10.438470	-25.907990	21.660420	0.406170
50%	-0.063410	3.296410	0.109740	-0.042370	-7.544550	83.905200	69.691950
75%	0.612090	7.602780	5.510740	4.578130	0.657260	216.177610	168.372510
max	4.840330	20.694590	24.239190	140.795700	223.064370	2227.927820	7852.089700

# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)
Prepare the Data

This section prepares the data before running the K-Means algorithm. It follows these steps:

Use the StandardScaler module from scikit-learn to normalize the CSV file data. This will require you to utilize the fit_transform function.

Create a DataFrame that contains the scaled data. Be sure to set the coin_id index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.


# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)

# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)
​
# Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index
​
# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")
​
# Display sample data
df_market_data_scaled.head()
price_change_percentage_24h	price_change_percentage_7d	price_change_percentage_14d	price_change_percentage_30d	price_change_percentage_60d	price_change_percentage_200d	price_change_percentage_1y
coin_id							
bitcoin	0.508529	0.493193	0.772200	0.235460	-0.067495	-0.355953	-0.251637
ethereum	0.185446	0.934445	0.558692	-0.054341	-0.273483	-0.115759	-0.199352
tether	0.021774	-0.706337	-0.021680	-0.061030	0.008005	-0.550247	-0.282061
ripple	-0.040764	-0.810928	0.249458	-0.050388	-0.373164	-0.458259	-0.295546
bitcoin-cash	1.193036	2.000959	1.760610	0.545842	-0.291203	-0.499848	-0.270317
Find the Best Value for k Using the Original Data

In this section, you will use the elbow method to find the best value for k.

Code the elbow method algorithm to find the best value for k. Use a range from 1 to 11.

Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.

Answer the following question: What is the best value for k?


# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))

# Create an empy list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    k_model = KMeans(n_clusters=i, random_state=1)
    k_model.fit(df_market_data_scaled)
    inertia.append(k_model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, "inertia": inertia}
elbow_data
​
# Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)

# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_plot = df_elbow.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)
​
elbow_plot
Answer the following question: What is the best value for k?

Question: What is the best value for k?

Answer: 4

Cluster Cryptocurrencies with K-means Using the Original Data

In this section, you will use the K-Means algorithm with the best value for k found in the previous section to cluster the cryptocurrencies according to the price changes of cryptocurrencies provided.

Initialize the K-Means model with four clusters using the best value for k.

Fit the K-Means model using the original data.

Predict the clusters to group the cryptocurrencies using the original data. View the resulting array of cluster values.

Create a copy of the original data and add a new column with the predicted clusters.

Create a scatter plot using hvPlot by setting x="price_change_percentage_24h" and y="price_change_percentage_7d". Color the graph points with the labels found using K-Means and add the crypto name in the hover_cols parameter to identify the cryptocurrency represented by each data point.


# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)

# Fit the K-Means model using the scaled data
model.fit(df_market_data_scaled)

KMeans
KMeans(n_clusters=4)

# Predict the clusters to group the cryptocurrencies using the scaled data
k_4 = model.predict(df_market_data_scaled)
​
# View the resulting array of cluster values.
k_4
array([0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 2, 0, 2, 2, 0, 2, 2, 0,
       2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 1, 0, 2, 2, 3, 2, 2, 2, 2],
      dtype=int32)

# Create a copy of the DataFrame
market_data_predictions_df = df_market_data_scaled.copy()

# Add a new column to the DataFrame with the predicted clusters
market_data_predictions_df['predicted clusters'] = k_4
​
# Display sample data
market_data_predictions_df.head()
price_change_percentage_24h	price_change_percentage_7d	price_change_percentage_14d	price_change_percentage_30d	price_change_percentage_60d	price_change_percentage_200d	price_change_percentage_1y	predicted clusters
coin_id								
bitcoin	0.508529	0.493193	0.772200	0.235460	-0.067495	-0.355953	-0.251637	0
ethereum	0.185446	0.934445	0.558692	-0.054341	-0.273483	-0.115759	-0.199352	0
tether	0.021774	-0.706337	-0.021680	-0.061030	0.008005	-0.550247	-0.282061	2
ripple	-0.040764	-0.810928	0.249458	-0.050388	-0.373164	-0.458259	-0.295546	2
bitcoin-cash	1.193036	2.000959	1.760610	0.545842	-0.291203	-0.499848	-0.270317	0

# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
predictions_scatter_plot = market_data_predictions_df.hvplot.scatter(
    title="Scatter Plot by Crypto Currency",
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="predicted clusters",
    hover_cols=['coin_id'],
    frame_width=700,
    frame_height=600
)
​
predictions_scatter_plot
Optimize Clusters with Principal Component Analysis

In this section, you will perform a principal component analysis (PCA) and reduce the features to three principal components.

Create a PCA model instance and set n_components=3.

Use the PCA model to reduce to three principal components. View the first five rows of the DataFrame.

Retrieve the explained variance to determine how much information can be attributed to each principal component.

Answer the following question: What is the total explained variance of the three principal components?

Create a new DataFrame with the PCA data. Be sure to set the coin_id index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.


# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)

# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
crypto_pca_data = pca.fit_transform(df_market_data_scaled)
​
# View the first five rows of the DataFrame. 
crypto_pca_data[0:5]
array([[-0.60066733,  0.84276006,  0.46159457],
       [-0.45826071,  0.45846566,  0.95287678],
       [-0.43306981, -0.16812638, -0.64175193],
       [-0.47183495, -0.22266008, -0.47905316],
       [-1.15779997,  2.04120919,  1.85971527]])

# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
variance = pca.explained_variance_ratio_
​
print(f'PC1 holds {variance[0]:.2f}%, PC2 holds {variance[1]:.2f}%, and PC3 holds {variance[2]: .2f}%. The primary components hold a total of {sum(variance) * 100: .2f}% of the original data.')
PC1 holds 0.37%, PC2 holds 0.35%, and PC3 holds  0.18%. The primary components hold a total of  89.50% of the original data.
Answer the following question: What is the total explained variance of the three principal components?

Question: What is the total explained variance of the three principal components?

Answer: 89.5%


# Create a new DataFrame with the PCA data.
df_market_data_pca = pd.DataFrame(crypto_pca_data, columns=['PC1', 'PC2', 'PC3'])
​
# Copy the crypto names from the original data
df_market_data_pca['coin_id'] = df_market_data_scaled.index
​
# Set the coinid column as index
df_market_data_pca = df_market_data_pca.set_index('coin_id')
​
# Display sample data
df_market_data_pca.head()
PC1	PC2	PC3
coin_id			
bitcoin	-0.600667	0.842760	0.461595
ethereum	-0.458261	0.458466	0.952877
tether	-0.433070	-0.168126	-0.641752
ripple	-0.471835	-0.222660	-0.479053
bitcoin-cash	-1.157800	2.041209	1.859715
Find the Best Value for k Using the PCA Data

In this section, you will use the elbow method to find the best value for k using the PCA data.

Code the elbow method algorithm and use the PCA data to find the best value for k. Use a range from 1 to 11.

Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.

Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?


# Create a list with the number of k-values to try
# Use a range from 1 to 11
pca_k_values = list(range(1, 11))

# Create an empy list to store the inertia values
pca_inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for k in pca_k_values: 
    model = KMeans(n_clusters=k)
    model.fit(df_market_data_pca)
    pca_inertia.append(model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
pca_elbow_data = {
    'k': pca_k_values,
    'Inertia': pca_inertia
}
​
# Create a DataFrame with the data to plot the Elbow curve
df_pca_elbow = pd.DataFrame(pca_elbow_data)

# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
pca_elbow_plot = df_pca_elbow.hvplot(x='k', y='Inertia', title='PCA K Values Elbow Curve')
pca_elbow_plot
Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?

Question: What is the best value for k when using the PCA data?

Answer: 4
Question: Does it differ from the best k value found using the original data?

Answer: No
Cluster Cryptocurrencies with K-means Using the PCA Data

In this section, you will use the PCA data and the K-Means algorithm with the best value for k found in the previous section to cluster the cryptocurrencies according to the principal components.

Initialize the K-Means model with four clusters using the best value for k.

Fit the K-Means model using the PCA data.

Predict the clusters to group the cryptocurrencies using the PCA data. View the resulting array of cluster values.

Add a new column to the DataFrame with the PCA data to store the predicted clusters.

Create a scatter plot using hvPlot by setting x="PC1" and y="PC2". Color the graph points with the labels found using K-Means and add the crypto name in the hover_cols parameter to identify the cryptocurrency represented by each data point.


# Initialize the K-Means model using the best value for k
pca_model = KMeans(n_clusters=4)

# Fit the K-Means model using the PCA data
pca_model.fit(df_market_data_pca)

KMeans
KMeans(n_clusters=4)

# Predict the clusters to group the cryptocurrencies using the PCA data
crypto_segments_pca = pca_model.predict(df_market_data_pca)
​
# View the resulting array of cluster values.
print(crypto_segments_pca)
[2 2 0 0 2 2 2 2 2 0 0 0 0 2 0 2 0 0 2 0 0 2 0 0 0 0 0 0 2 0 0 0 3 2 0 0 1
 0 0 0 0]

# Create a copy of the DataFrame with the PCA data
market_data_pca_predictions = df_market_data_pca.copy()
​
# Add a new column to the DataFrame with the predicted clusters
market_data_pca_predictions['CryptoClusters'] = crypto_segments_pca
​
# Display sample data
market_data_pca_predictions.head()
PC1	PC2	PC3	CryptoClusters
coin_id				
bitcoin	-0.600667	0.842760	0.461595	2
ethereum	-0.458261	0.458466	0.952877	2
tether	-0.433070	-0.168126	-0.641752	0
ripple	-0.471835	-0.222660	-0.479053	0
bitcoin-cash	-1.157800	2.041209	1.859715	2

# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
pca_predictions_scatter_plot = market_data_pca_predictions.hvplot.scatter(
    title='Scatter Plot by Crypto Currency Segment Using Primary Clusters - k=4',
    x='PC1',
    y='PC2',
    by='CryptoClusters',
    hover_cols=['coin_id'],
    frame_width=700,
    frame_height=600
)
​
pca_predictions_scatter_plot
Visualize and Compare the Results

In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

Create a composite plot using hvPlot and the plus (+) operator to contrast the Elbow Curve that you created to find the best value for k with the original and the PCA data.

Create a composite plot using hvPlot and the plus (+) operator to contrast the cryptocurrencies clusters using the original and the PCA data.

Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

Rewind: Back in Lesson 3 of Module 6, you learned how to create composite plots. You can look at that lesson to review how to make these plots; also, you can check the hvPlot documentation.


# Composite plot to contrast the Elbow curves
elbow_plot + pca_elbow_plot

# Compoosite plot to contrast the clusters
predictions_scatter_plot + pca_predictions_scatter_plot
Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

Question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

Answer: The inertia for the clusters with the PCA plot are lower than that of the original plot. The values within each cluster are much closer together and easier to distinguish compared to the ones in the original data. The PCA plot separated the two outliers a little better at a glance, but accuracy seems fair for both scenarios.