## Song-Recommender-System
# Create a music recommender based on a set of ‘liked’ songs
# Spotify Song Recommender System in Python
![image](https://user-images.githubusercontent.com/99494058/201613151-6a5645fd-97f6-4f25-b549-754023b52e1d.png)
![image](https://user-images.githubusercontent.com/99494058/201613185-84a65829-4cad-4e71-aadd-6dd37be0d56f.png)
Building a recommendation system is a common task that is faced by Amazon, Netflix, Spotify and Google. The underlying goal of the recommendation system is to personalize content and identify relevant data for our audiences. These contents can be articles, movies, games, etc
There are 3 types of recommendation system: content-based, collaborative and popularity.

In this exercise, we will learn how to build a music recommendation system using real data. Our Million Songs Dataset contains of two files: triplet_file and metadata_file. The triplet_file contains user_id, song_id and listen time. The metadat_file contains song_id, title, release_by and artist_name. Million Songs Dataset is a mixture of song from various website with the rating that users gave after listening to the song. A few examples are Last.fm, thisismyjam, musixmatch, etc

Our first job is to integrate our dataset, which is very important every time we want to build a data processing pipeline.To integrate both triplet_file and metadata_file, we are going to use a popular Python library called pandas

We first define the two files we are going to work with:

triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'
We then read the table of triplet_file using pandas and define the 3 columns as user_id, song_id and listen_count ( df here means dataframe)

song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']
We also read the metadat_file and going to combine the metadata_file with triplets_file. Whenever you combine 2 or more datasets, there will be duplicate columns. Here we drop the duplicates between 2 datasets using song_id


song_df_2 =  pandas.read_csv(songs_metadata_file)

song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
Using command song_df.head()allows us to visualize the combined data set:


Here we have the song index, user_id, song_id, listen_count, title, release and artist_name. Running len(song_df) returns the the total length of this dataset indexed by song are 2,000,000.

The second step of this exercise is data transformation, where we’re going to select a subset of this data (the first 10,000 songs). We then merge the song and artist_name into one column, aggregated by number of time a particular song is listened too in general by all users. The first line in the code below group the song_df by number of listen_count ascending. The second line calculate the group_sum by summing the listen_count of each song. The third line add a new column called percentage, and calculate this percentage by dividing the listen_count by the sum of listen_count of all songs and then multiply by 100. The last line list the song in the ascending order of popularity for a given song

song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])
Below are the example of our dataset after transformation step:


Doing data transformation allows us to further simplify our dataset and make it easy and simple to understand.

Next step, we’re going to follow a naive approach when building a recommendation system. We’re going to count the number of unique users and songs in our subset of data

users = song_df['user_id'].unique()
len(users) ## return 365 unique users
songs = song_df['song'].unique()
len(songs) ## return 5151 unique songs
We then create a song recommender by splitting our dataset into training and testing data. We use the train_test_split function of scikit-learnlibrary. It’s important to note that whenever we build a machine learning system, before we train our model, we always want to split our data into training and testing dataset

train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
We arbitrarily pick 20% as our testing size. We then used a popularity based recommender class as a blackbox to train our model. We create an instance of popularity based recommender class and feed it with our training data. The code below achieves the following goal: based on the popularity of each song, create a recommender that accept a user_id as input and out a list of recommended song of that user

pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')
#user the popularity model to make some prediction
user_id = users[5]
pm.recommend(user_id)

The code for the Recommender Systems model is below. This system is a naive approach and not personalized. It first get a unique count of user_id (ie the number of time that song was listened to in general by all user) for each song and tag it as a recommendation score. The recommend function then accept a user_id and output the top ten recommended song for any given user. Keeping in my that since this is the naive approach, the recommendation is not personalized and will be the same for all users.

#Class for Popularity based Recommender System modelclass popularity_recommender_py():    
    def __init__(self):        
    self.train_data = None        
    self.user_id = None        
    self.item_id = None        
    self.popularity_recommendations = None            
    #Create the popularity based recommender system model    
    def create(self, train_data, user_id, item_id): 
        self.train_data = train_data
        self.user_id = user_id        
        self.item_id = item_id         
        
        #Get a count of user_ids for each unique song as   recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()        
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)            
        #Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])            
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)     
        #Use the popularity based recommender system model to    
        #make recommendations    
    def recommend(self, user_id):            
        user_recommendations = self.popularity_recommendations                 
        #Add user_id column for which the recommendations are being generated        
        user_recommendations['user_id'] = user_id            
        #Bring user_id column to the front        
        cols = user_recommendations.columns.tolist()        
        cols = cols[-1:] + cols[:-1]        
        user_recommendations = user_recommendations[cols]
        return user_recommendations
The second part of this exercise is to create a ML personalized song recommender system by leveraging the item similarity based collaborative filtering model. Recall that recommender system is divided into 2 types: content based and collaborative based. Content based system predicts what a user like based on what that user like in the past. Collaborative based system predict what a particular user like based on what other similar users like. Most companies like Netflix and Hulu use the hybrid approach, which provide recommendation based on the combination of what content a user like in the past as well as what other similar user like.

According to Agnes Jóhannsdóttir (Twitter: @agnesjohanns) at Cambridge Coding Academy, Memory-based collaborative filtering can be divided into two main approaches: user-item filtering and item-item filtering.

Item-item filtering approach involves defining a co-occurrence matrix based on a song a user likes. We are seeking to answer a question, for each song, what a number of time a user, who have listened to that song, will also listen to another set of other songs. To further simplify this, based on what you like in the past, what other similar song that you will like based on what other similar user have liked. Let’s apply this to our code. First we create an instance item similarity based recommender class and feed it with our training data.

is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')
Notice that inside the recommender system’s source code, the generate_top_recommendations function calculated a weighted average of the scores in cooccurence matrix for all user song. This cooccurence matrix will tend to be sparse matrix because it’s not possible to predict if a user like a particular song, whether or not he/she will like a million other song. The possibility is so vast. Using our model, we will be able to predict the list of song that a user will like

#Print the songs for the user in training data
user_id = users[5]
user_items = is_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id)
output:

------------------------------------------------------------------------------------
Training data songs for the user userid: 4bd88bfb25263a75bbdd467e74018f4ae570e5df:
------------------------------------------------------------------------------------
Just Lose It - Eminem
Without Me - Eminem
16 Candles - The Crests
Speechless - Lady GaGa
Push It - Salt-N-Pepa
Ghosts 'n' Stuff (Original Instrumental Mix) - Deadmau5
Say My Name - Destiny's Child
My Dad's Gone Crazy - Eminem / Hailie Jade
The Real Slim Shady - Eminem
Somebody To Love - Justin Bieber
Forgive Me - Leona Lewis
Missing You - John Waite
Ya Nada Queda - Kudai
----------------------------------------------------------------------
Recommendation process going on:
----------------------------------------------------------------------
No. of unique songs for the user: 13
no. of unique songs in the training set: 4483
Non zero values in cooccurence_matrix :2097
We can also use our item similarity based collaborative filtering model to find similar songs to any songs in our dataset:

is_model.get_similar_items(['U Smile - Justin Bieber'])
this output

no. of unique songs in the training set: 4483
Non zero values in cooccurence_matrix :271

It’s worth to note that this method is not Deep Learning but purely based on linear algebra.

To recap, in this exercise we discussed 2 models. The first model is popularity based recommender, meaning it is not personalized toward any user and will output the same list of recommended songs. The second model is personalized recommender leveraging the item similarity based collaborative filtering model (ie the cooccurence matrix) to find a personalized list of song that a user might like based on what other similar user have liked.

Next we will discuss how to measure the performance of these two models using a precision recall curve to quantitatively compare the popularity based model and personalized collaborative filtering model.

To quantitatively measure the performance of recommender system, we use three different metrics: Precision , Recall and F-1 Score


Source: http://aimotion.blogspot.com/2011/05/evaluating-recommender-systems.html
According to Marcel Caraciolo, Precision is “the proportion of top results that are relevant, considering some definition of relevant for your problem domain”. In our case, the definition of relevant for our problem domain is the length that a song is listened to, a number of user have all liked the song. Recall would “measure the proportion of all relevant results included in the top results”.In our case, it means precision seeks to measure the relevancy of songs in relation to the top ten results of recommended song, whereas recall seeks to measure the relevancy of songs in relation to all the songs


Observing the precision recall curve of both our popularity based model and personalized item similarity model, item similarity model perform better (ie having higher number of recall and precision) up to certain point in precision-recall curve.

The last type of recommender system is Matrix Factorization based Recommender System. This type of recommender system uses what is called a Singular Value Decomposition (SVD) factorized matrix of the original similarity matrix to build recommender system.

To compute SVD and recommendations, we use the following code:

#constants defining the dimensions of our User Rating Matrix (URM) MAX_PID = 4 
MAX_UID = 5  
#Compute SVD of the user ratings matrix 
def computeSVD(urm, K):     
    U, s, Vt = sparsesvd(urm, K)      
    dim = (len(s), len(s))     
    S = np.zeros(dim, dtype=np.float32)     
    for i in range(0, len(s)):         
        S[i,i] = mt.sqrt(s[i])      
        U = csc_matrix(np.transpose(U), dtype=np.float32)     
        S = csc_matrix(S, dtype=np.float32)     
        Vt = csc_matrix(Vt, dtype=np.float32)          
        return U, S, Vt
In this code, U represents user vector, S represents the item vector.Vt represent the joint of these two vectors as collection of points (ie vector) in 2 dimensional spaces. We’re going to use these vectors to measure the distance from one user’s preferences to another user’s preferences.

In another word, we are vectorizing matrices in order to compute the distance between matrices. To further clarify this, we’re going to talk through an example. Assume we have a user song matrix below:

        Song0   Song1   Song2   Song3 
User0   3       1       2       3
User1   4       3       4       3
User2   3       2       1       5
User3   1       6       5       2
User4   0       0       5       0
Once we perform SVD, the output is going to be vectors and measuring distance between vectors gives us recommendation

#Compute estimated rating for the test user
def computeEstimatedRatings(urm, U, S, Vt, uTest, K, test):
    rightTerm = S*Vt
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm
        #we convert the vector to dense format in order to get the     #indices
        #of the movies with the best estimated ratings 
        estimatedRatings[userTest, :] = prod.todense()
        recom = (-estimatedRatings[userTest, :]).argsort()[:250]
    return recom

#Used in SVD calculation (number of latent factors)
K=2
#Initialize a sample user rating matrix
urm = np.array([[3, 1, 2, 3],[4, 3, 4, 3],[3, 2, 1, 5], [1, 6, 5, 2], [5, 0,0 , 0]])
urm = csc_matrix(urm, dtype=np.float32)
#Compute SVD of the input user ratings matrix
U, S, Vt = computeSVD(urm, K)
#Test user set as user_id 4 with ratings [0, 0, 5, 0]
uTest = [4]
print("User id for whom recommendations are needed: %d" % uTest[0])
#Get estimated rating for test user
print("Predictied ratings:")
uTest_recommended_items = computeEstimatedRatings(urm, U, S, Vt, uTest, K, True)
print(uTest_recommended_items)
will output:

User id for whom recommendations are needed: 4
Predictied ratings:
[0 3 2 1]
Next, we discuss the real world example of how Hulu applying Deep Learning to Collaborative Filtering to build its industry leading recommendation system. At Hulu, features like Personalized Masthead, Watchlist and Top Picks are all powered by collaborative filtering.

The method Hulu used is CF-NADE. Let’s take an example. Suppose we have 4 movies: “Transformers”, “SpongeBob”, “Teenage Mutant Ninja Turtles” and “Interstellar”, with scores 4,2,3 and 5. In CF-NADE, the joint probability of vector (4,2,3,5) is factorized as a product of conditionals by chain rule, which are the following:

1/ The probability that the user gives “Transformers” 4-star conditioned on nothing;
2/ The probability that the user gives “SpongeBob” 2-star conditioned on giving “Transformers” 4-star;
3/ The probability that the user gives “Teenage Mutant Ninja Turtles” a 3-star conditioned on giving 4-star and 2-star to “Transformers” and “SpongeBob”, respectively;
4/ The probability that the user gives “Interstellar” a 5-star conditioned on giving 4-star, 2-star and 3-star to “Transformers”, “SpongeBob” and “Teenage Mutant Ninja Turtles”, respectively;
To summarize, this is the chain of probability based on what previously have occurred. Each conditional is modeled by its own neural network and the parameter for all of these neural networks are shared amongst all models.

or
## Build a Song Recommender System using Content-Based Filtering in Python.

With the rapid growth in online and mobile platforms, lots of music platforms are coming into the picture. These platforms are offering songs lists from across the globe. Every individual has a unique taste for music. Most people are using Online music streaming platforms such as Spotify, Apple Music, Google Play, or Pandora.

Online Music listeners have lots of choices for the song. These customers sometimes get very difficult in selecting the songs or browsing the long list. The service providers need an efficient and accurate recommender system for suggesting relevant songs. As data scientists, we need to understand the patterns in music listening habits and predict the accurate and most relevant recommendations.

In this tutorial, we are going to cover the following topics:

# Contents  hide 
1 Content-Based Song Recommender System
2 Loading Dataset
3 Understanding the Dataset
4 Perform Feature Scaling
5 Building Recommender System using Cosine Similarity
6 Recommeding songs
7 Song Recommendations using Sigmoid Kernel
8 Summary
Content-Based Song Recommender System
The content-based filtering method is based on the analysis of item features. It determines which features are most important for suggesting the songs. For example, if the user has liked a song in the past and the feature of that song is the theme and that theme is party songs then Recommender System will recommend the songs based on the same theme. So the system adapts and learns the user behavior and suggests the items based on that behavior. In this article, we are using the Spotify dataset to discover similar songs for recommendation using cosine similarity and sigmoid kernel.

# Song Recommender System
Loading Dataset
In this tutorial, you will build a book recommender system. You can download this dataset from here.

Let’s load the data into pandas dataframe:


import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

df=pd.read_csv("data.csv")

df.head(
Output:


Understanding the Dataset
Let’s understand the dataset. In this dataset, we have 15 columns: acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, valence, target, song_title, artist.

Acosticness confidence measure from 0.0 to 1.0 of whether the track is acoustic.
Danceability measure describes how suitable a track is for dancing.
duration_ms is the duration of the song track in milliseconds.
Energy represents a perceptual measure of intensity and activity.
Instrumentalness predicts whether a track contains vocals or not.
Loudness of a track in decibels(dB).
Liveness detects the presence of an audience in the recording.
Speechiness detects the presence of spoken words in a track
Time_signature is an estimated overall time signature of a track.
Key the track is in. Integers map to pitches using standard Pitch Class notation.
Valence measures from 0.0 to 1.0 describing the musical positiveness conveyed by a track.
Target value describes the encoded value of 0 and 1. 0 means listener has not saved the song and 1 means listener have saved the song.
Tempo is in beats per minute (BPM).
Mode indicates the modality(major or minor) of the song.
Song_title is the name of the song.
Artist is the singer of the song.

df.info()
Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2017 entries, 0 to 2016
Data columns (total 17 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   Unnamed: 0        2017 non-null   int64  
 1   acousticness      2017 non-null   float64
 2   danceability      2017 non-null   float64
 3   duration_ms       2017 non-null   int64  
 4   energy            2017 non-null   float64
 5   instrumentalness  2017 non-null   float64
 6   key               2017 non-null   int64  
 7   liveness          2017 non-null   float64
 8   loudness          2017 non-null   float64
 9   mode              2017 non-null   int64  
 10  speechiness       2017 non-null   float64
 11  tempo             2017 non-null   float64
 12  time_signature    2017 non-null   float64
 13  valence           2017 non-null   float64
 14  target            2017 non-null   int64  
 15  song_title        2017 non-null   object 
 16  artist            2017 non-null   object 
dtypes: float64(10), int64(5), object(2)
memory usage: 268.0+ KB
Perform Feature Scaling
Before building the model, first we normalize or scale the dataset. For scaling it we are using MinMaxScaler of Scikit-learn library.


Min-Max Scaler

feature_cols=['acousticness', 'danceability', 'duration_ms', 'energy',
              'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
              'speechiness', 'tempo', 'time_signature', 'valence',]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_df =scaler.fit_transform(df[feature_cols])

print(normalized_df[:2])
Building Recommender System using Cosine Similarity
In this section, we are building a content-based recommender system using similarity measures such as Cosine and Sigmoid Kernel. Here, we will find the similarities among items or songs feature set and pick the top 10 most similar songs and recommend them.

Cosine similarity measures the cosine angle between two feature vectors. Its value implies that how two records are related to each other. Cosine similarity can be computed for the non-equal size of text documents.



# Create a pandas series with song titles as indices and indices as series values 
indices = pd.Series(df.index, index=df['song_title']).drop_duplicates()

# Create cosine similarity matrix based on given matrix
cosine = cosine_similarity(normalized_df)

def generate_recommendation(song_title, model_type=cosine ):
    """
    Purpose: Function for song recommendations 
    Inputs: song title and type of similarity model
    Output: Pandas series of recommended songs
    """
    # Get song indices
    index=indices[song_title]
    # Get list of songs for given songs
    score=list(enumerate(model_type[indices['Parallel Lines']]))
    # Sort the most similar songs
    similarity_score = sorted(score,key = lambda x:x[1],reverse = True)
    # Select the top-10 recommend songs
    similarity_score = similarity_score[1:11]
    top_songs_index = [i[0] for i in similarity_score]
    # Top 10 recommende songs
    top_songs=df['song_title'].iloc[top_songs_index]
    return top_songs
In the above code, we have computed the similarity using Cosine similarity and returned the Top-10 recommended songs.

Recommeding songs
Let’s make a forecast using computed cosine similarity on the Spotify song dataset.


print("Recommended Songs:")
print(generate_recommendation('Parallel Lines',cosine).values)
In the above code, we have generated the Top-10 song list based on cosine similarity.

Song Recommendations using Sigmoid Kernel
Let’s make a forecast using computed Sigmoid kernel on Spotify song dataset.


# Create sigmoid kernel matrix based on given matrix
sig_kernel = sigmoid_kernel(normalized_df)

print("Recommended Songs:")
print(generate_recommendation('Parallel Lines',sig_kernel).values)
In the above code, we have generated the Top-10 song list based on Sigmoid Kernel.

Summary
Congratulations, you have made it to the end of this tutorial!

In this tutorial, we have built the song recommender system using cosine similarity and Sigmoid kernel. This developed recommender system is a content-based recommender system. In another article, we have developed the recommender system using collaborative filtering. You can check that article here Book Recommender System using KNN. You can also check another article on the NLP-based recommender system.

# Authors 
Amen Mengstu <AmenMengstu> <https://github.com/AmenMengstu>
   

