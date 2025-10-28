import pandas as pd
import sqlite3
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

DATABASE = 'database.sqlite'


# FUNCTIONS

def find_optimal_K(range_max, corpus, dictionary, bow_list):
    optimal_coherence = -100
    optimal_lda = None
    optimal_k = 0
    for K in range(2, range_max):    
        # Train LDA model
        lda = LdaModel(corpus, num_topics=K, id2word=dictionary, passes=10, random_state=2)

        # Now that the LDA model is done, let's see how good it is by computing its 'coherence score'
        coherence_model = CoherenceModel(model=lda, texts=bow_list, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        
        if(coherence_score > optimal_coherence):
            print(f'Trained LDA with {K} topics. Average topic coherence (higher is better): {coherence_score} which is the best so far!')
            optimal_coherence = coherence_score
            optimal_lda = lda
            optimal_k = K
        else: 
            print(f'Trained LDA with {K} topics. Average topic coherence (higher is better): {coherence_score} which is not very good.')

    return optimal_k, optimal_lda

def sentiment_by_sentiment_score(score):
    sentiment = 'Neutral'
    if score > 0.05:
        sentiment = 'Positive'
    elif score < -0.05:
        sentiment = 'Negative'
    
    return sentiment

def main():

    # --- Exercise 4.1 ---
    print(f'--- Exercise 4.1 ---')
    print('\n')
    # Download necessary NLTK data, without these the below functions wouldn't work
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Database connection and load data
    conn = sqlite3.connect(DATABASE)
    posts = pd.read_sql_query("SELECT id, content FROM posts", conn)

    # Get a basic stopword list
    stop_words = stopwords.words('english')

    # Add extra words to make our analysis even better
    stop_words.extend(['would', 'best', 'always', 'amazing', 'bought', 'quick' 'people', 'new', 'fun', 'think', 'know', 'believe', 'many', 'thing', 'need', 'small', 'even', 'make', 'love', 'mean', 'fact', 'question', 'time', 'reason', 'also', 'could', 'true', 'well',  'life', 'said', 'year', 'going', 'good', 'really', 'much', 'want', 'back', 'look', 'article', 'host', 'university', 'reply', 'thanks', 'mail', 'post', 'please'])

    # Lemmatise words
    lemmatizer = WordNetLemmatizer()

    # after the below for loop, we will transform each post into "bags of words" where each BOW is a set of words from one post 
    bow_list = []

    for _, row in posts.iterrows():
        text = row['content']
        tokens = word_tokenize(text.lower()) # tokenise (i.e. get the words from the post)
        tokens = [lemmatizer.lemmatize(t) for t in tokens] # lemmatise
        tokens = [t for t in tokens if len(t) > 2]  # filter out words with less than 3 letter s
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words] # filter out stopwords
        # if there's at least 1 word left for this post, append to list
        if len(tokens) > 0:
            bow_list.append(tokens)

    # Create dictionary and corpus
    dictionary = Dictionary(bow_list)

    # Filter words that appear less than 2 times or in more than 30% of posts
    dictionary.filter_extremes(no_below=2, no_above=0.3)
    corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]

    optimal_lda = None
    optimal_k = 85 # Set after first running, saves time when running again..

    # if optimal_k does not set, have to find it
    if optimal_k == 0 and optimal_lda == None:
        optimal_k, optimal_lda =  find_optimal_K(100, corpus, dictionary, bow_list)
    else:
        optimal_lda = LdaModel(corpus, num_topics=optimal_k, id2word=dictionary, passes=10, random_state=2)

    # Save topics, top 6 most representative words per topic
    topic_list = [0] * optimal_k 
    for i, topic in optimal_lda.print_topics(num_words=6, num_topics=-1):
        topic_list[i] = topic

    # Then, let's determine how many posts we have for each topic
    # Count the dominant topic for each document
    topic_counts = [0] * optimal_k  # one counter per topic
    for bow in corpus:
        topic_dist = optimal_lda.get_document_topics(bow)  # list of (topic_id, probability)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0] # find the top probability
        topic_counts[dominant_topic] += 1 # add 1 to the most probable topic's counter

    # Sort based on counts and then print those
    sorted_topics = []
    for i, count in enumerate(topic_counts):
        sorted_topics.append((i,count))
    
    sorted_topics.sort(key=lambda x: x[1], reverse=True)

    for i, info in enumerate(sorted_topics):
        print(f"Number {i+1} topic is ({topic_list[info[0]]}) with {info[1]} posts")

    # --- Exercise 4.2 ---
    print('\n')
    print(f'--- Exercise 4.2 ---')
    
    nltk.download('vader_lexicon')
    print('\n')

    # Combine all posts and comments
    comments = pd.read_sql_query("SELECT id, content FROM comments", conn)
    posts_and_comments = pd.concat([posts, comments], ignore_index=True)

    # do sentiment analysis
    sia = SentimentIntensityAnalyzer()

    # Calculate sentiment scores for each content
    posts_and_comments['sentiment_score'] = posts_and_comments['content'].apply(lambda content: sia.polarity_scores(content)['compound'])

    sentiment_score_avg = posts_and_comments['sentiment_score'].mean()
    overall_sentiment = sentiment_by_sentiment_score(sentiment_score_avg)

    print(f'Average sentiment score overall in the platform (posts+comments): {sentiment_score_avg}')
    print(f'Which means that the overall tone is: {overall_sentiment}')

    # Find all posts(and their comments) to each top 10 topics
    top_10_topics = [t[0] for t in sorted_topics[:10]]

    topic_sentiments = []

    for topic_id in top_10_topics:
        # Find posts inside this topic
        topic_post_ids = []
        for idx, bow in enumerate(corpus):
            topic_dist = optimal_lda.get_document_topics(bow)
            dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
            if dominant_topic == topic_id:
                topic_post_ids.append(posts.iloc[idx]['id'])

        # Get comments of those posts
        post_ids_str = ','.join(map(str, topic_post_ids))
        query = f"SELECT id, content FROM comments WHERE post_id IN ({post_ids_str})"
        topic_comments = pd.read_sql_query(query, conn) if len(topic_post_ids) > 0 else pd.DataFrame(columns=['id', 'content'])

        # Combine posts and their comments
        topic_texts = pd.concat([
            posts[posts['id'].isin(topic_post_ids)][['id', 'content']],
            topic_comments
        ], ignore_index=True)

        # Skip if no text
        if topic_texts.empty:
            continue
        
        # Calculate sentiment score for this topic
        topic_texts['sentiment_score'] = topic_texts['content'].apply(lambda content: sia.polarity_scores(content)['compound'])
        avg_sentiment_score = topic_texts['sentiment_score'].mean()
        topic_sentiments.append((topic_id, avg_sentiment_score, sentiment_by_sentiment_score(avg_sentiment_score)))
    
    print('\n')
    print('Sentiments for top 10 topics:')
    for topic_id, avg_score, label in topic_sentiments:
        print(f"Topic ({topic_list[topic_id]}): sentiment score is {avg_score:.2f} so sentiment is {label}")

    
    conn.close()

if __name__ == '__main__':
    main()