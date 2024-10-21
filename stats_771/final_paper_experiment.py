# %%
# importing the needed packages to explore the data
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

import nltk

nltk.download("stopwords")
nltk.download("wordnet")
import stopwordsiso

from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
from nltk.stem import PorterStemmer

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import re

# %%
# read in the bbc data
bbc_data = pd.read_csv("bbc-news-data.csv", sep="\t")
# Removing duplicated articles from the database
bbc_data.drop_duplicates(
    subset=["category", "title", "content"], keep="last", inplace=True
)
# remove columns from the dataframe that are not needed
bbc_data["text"] = bbc_data["title"] + "" + bbc_data["content"]
bbc_data = bbc_data[["category", "text"]]
# dataset description
bbc_data.info()

# %%
# displaying the unique categories of articles present in the data
print(bbc_data["category"].unique())
# factorizing the categories
label_encoder = preprocessing.LabelEncoder()
bbc_data["cat_id"] = label_encoder.fit_transform(bbc_data["category"])
# creating a temporary dataframe 'df' to determine the distribution of the categories
df = bbc_data["category"].value_counts().to_frame().reset_index()
df = df.rename(columns={"index": "Category", "category": "Article Count"})

# determining how varied the category distribution is from balanced distribution
df["Actual Ratio"] = df["Article Count"] / df["Article Count"].sum()

# plotting the distribution of the categories
fig = px.bar(
    df,
    x="Category",
    y="Article Count",
    color="Category",
    text="Article Count",
    title="Article Category Count",
)
fig.update_layout(yaxis_title="Article Count", xaxis_title="Category")
fig.show()
fig.write_image("plots\\barplot.png")
fig = px.pie(
    df, names="Category", values="Actual Ratio", title="Article Category Distribution"
)
fig.show()
fig.write_image("plots\\pieplot.png")
# Determining the difference between actual ratio distribution and expected
df["Expected Ratio"] = 1 / len(bbc_data["category"].unique())
df["Difference"] = abs(df["Actual Ratio"] - df["Expected Ratio"])
df[["Category", "Expected Ratio", "Actual Ratio", "Difference"]]

# %% [markdown]
# From the above we can see that sports and business articles make up most of the dataset; with tech having the smallest article count. However, the overall difference between the categories is relatively small. The dataset is fairly balanced, with the largest difference from the expected ratio being 4%.

# %%
for cat in bbc_data["category"].unique():
    df = bbc_data.loc[bbc_data["category"] == cat]["text"].str.cat(sep=" ")

    # create word cloud for each category based on the article content
    wordcloud = WordCloud(
        stopwords=STOPWORDS, background_color="white", max_words=50
    ).generate(df)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{cat.title()} Articles")
    plt.savefig(f"plots\\{cat}_notcleaned_wordcloud.png")
    plt.show()


# %% [markdown]
# From the above we can see that there are some high frequency words that need to be dropped from all categories.
# These include Said, Will, UK, etc... These are words that are not stop words but do not provide any real context
#
# There is also a corpus of stopwords, that is, high-frequency words such as the, to, and
# also that we sometimes want to filter out of a document before further processing.
# Stopwords usually have little lexical content, and their presence in a text fails to distinguish
# it from other texts.  "o'relly's nlp book "
#
# I may need create my own specific stopword list to in order capture common words in this domain
#
# "Another way to reduce the size of the feature space is to eliminate stopwords such
# as the, to, and and, which may seem to play little role in expressing the topic, sentiment,
# or stance. This is typically done by creating a stoplist (e.g., NLTK.CORPUS.STOPWORDS),
# and then ignoring all terms that match the list. However, corpus linguists and social psychologists have shown that seemingly inconsequential words can offer surprising insights
# about the author or nature of the text (Biber, 1991; Chung and Pennebaker, 2007). Furthermore, high-frequency words are unlikely to cause overfitting in discriminative classifiers.
# As with normalization, stopword filtering is more important for unsupervised problems,
# such as term-based document retrieval." - Jacob Eisenstein

# %%
nltk_stops = set(nltk.corpus.stopwords.words("english"))
print(len(nltk_stops))
stops = nltk_stops
git_stop = stopwordsiso.stopwords("en")
stops.update(git_stop)
print(len(stops))

# %% [markdown]
# preprocess steps:
#     remove symbols and numbers
#     lower case everything but the 'US' >> 'united states'
#     remove stop words >> had to look for a more comprehensive list of stop words
#     lemmatize the token
#     and lower case again to get 'US' >> 'us'


# %%
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.split(" ")
    text = [w if w in ["US", "DS", "OS", "PS"] else w.lower() for w in text]
    return " ".join(text)


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stops]
    return " ".join(no_stopword_text)


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    lem = [lemmatizer.lemmatize(w) for w in text.split()]
    return " ".join(lem)


def stemming(text):
    stemmer = PorterStemmer()
    lem = [stemmer.stem(w) for w in text.split()]
    return " ".join(lem)


bbc_data["clean_text"] = bbc_data["text"].apply(clean_text)
bbc_data["clean_text"] = bbc_data["clean_text"].apply(remove_stopwords)
bbc_data["clean_text_lem"] = bbc_data["clean_text"].apply(lemmatization)
bbc_data["clean_text_stem"] = bbc_data["clean_text"].apply(stemming)
bbc_data["clean_text_lem"] = bbc_data["clean_text_lem"].str.lower()
bbc_data["clean_text_stem"] = bbc_data["clean_text_stem"].str.lower()

# %%
bbc_data.shape

# %%
for cat in bbc_data["category"].unique():
    df = bbc_data.loc[bbc_data["category"] == cat]["clean_text_stem"].str.cat(sep=" ")
    # create word cloud for each category based on the article content
    wordcloud = WordCloud(
        stopwords=stops, background_color="white", max_words=50
    ).generate(df)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{cat.title()} Articles")
    plt.show()

    freq_dist = nltk.FreqDist(df.split())

    freq_dist = pd.DataFrame(
        {"word": list(freq_dist.keys()), "count": list(freq_dist.values())}
    )

    fig = px.bar(
        freq_dist.nlargest(20, "count").sort_values("count"),
        y="word",
        x="count",
        title=f"Frequency of Words in {cat.title()}",
        color="count",
    )
    fig.update_layout(yaxis=dict(dtick=1))
    fig.show()

# %%
for cat in bbc_data["category"].unique():
    df = bbc_data.loc[bbc_data["category"] == cat]["clean_text_lem"].str.cat(sep=" ")

    # create word cloud for each category based on the article content
    wordcloud = WordCloud(
        stopwords=stops, background_color="white", max_words=50
    ).generate(df)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{cat.title()} Articles Word Cloud")
    plt.savefig(f"plots\\{cat}_wordcloud.png")
    plt.show()

    freq_dist = nltk.FreqDist(df.split())

    freq_dist = pd.DataFrame(
        {"word": list(freq_dist.keys()), "count": list(freq_dist.values())}
    )

    fig = px.bar(
        freq_dist.nlargest(20, "count").sort_values("count"),
        y="word",
        x="count",
        title=f"Word Frequencies {cat.title()} Articles",
        color="count",
    )
    fig.update_layout(yaxis=dict(dtick=1))
    fig.write_image(f"plots\\{cat}_freq.png")
    fig.show()

# %%
temp = bbc_data["clean_text_lem"].str.cat(sep=" ")
temp = set(temp.split())
temp_l = list(temp)
temp_l.sort()

# %%
clean_words = bbc_data["clean_text_lem"].str.cat(sep=" ").split()
words = bbc_data["text"].str.cat(sep=" ").split()

# %%
print(len(set(clean_words)), len(set(words)))


# %%
temp2 = bbc_data["text"].apply(clean_text).str.cat(sep=" ")
temp2 = set(temp2.split())
temp_l2 = list(temp2)
temp_l2.sort()


# %%
vectorizer_bow = CountVectorizer()
bow = vectorizer_bow.fit_transform(bbc_data["clean_text_lem"])
vectorizer_tfidf = TfidfVectorizer()
tfidf = vectorizer_tfidf.fit_transform(bbc_data["clean_text_lem"])
idf = vectorizer_tfidf.idf_
bow_tfidf = np.multiply(bow.toarray(), idf)


# %%
Y = bbc_data[["category", "cat_id"]]
bow_df = pd.DataFrame(bow.toarray(), columns=vectorizer_bow.get_feature_names_out())
tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vectorizer_bow.get_feature_names_out())
bow_tfidf_df = pd.DataFrame(bow_tfidf, columns=vectorizer_bow.get_feature_names_out())


# %%
max_tfidf = bow_tfidf_df.max().sort_values(ascending=False)
df = pd.DataFrame(max_tfidf.head(20)).reset_index()
df.rename(columns={"index": "word", 0: "tfidf_score"}, inplace=True)
fig = px.bar(
    df,
    x="tfidf_score",
    y="word",
    title="Top 20 Highest TF-IDF Score Words",
    color="tfidf_score",
)
fig.update_yaxes(autorange="reversed")
fig.update_layout(xaxis_title="TF-IDF score", yaxis_title="word")
fig.update_layout(yaxis=dict(dtick=1))
fig.write_image(f"plots\\tfidf.png")
fig.show()


# %%
computational_time = {}
reduced_data = {}

# %%
size = [
    int(round(0.02 * bow.shape[1], 0)),
    int(round(0.01 * bow.shape[1], 0)),
    int(round(0.001 * bow.shape[1], 0)),
    4,
]

# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    bow_tfidf_df, Y.iloc[:, 1], test_size=0.3, random_state=32, stratify=Y.iloc[:, 1]
)

tf_idf_train = bbc_data.iloc[X_train.index]["clean_text_lem"]
tf_idf_test = bbc_data.iloc[X_test.index]["clean_text_lem"]
# %%
np.random.seed(32)
for i in size:
    print(i)
    start_time = time.time()
    vectorizer_tfidf = TfidfVectorizer(
        max_features=i,
    )
    tfidf = vectorizer_tfidf.fit_transform(tf_idf_train)
    stop_time = time.time()
    idf = vectorizer_tfidf.idf_

    computational_time[f"tf-idf_{i}"] = stop_time - start_time
    reduced_data[f"tf-idf_{i}"] = tfidf
    reduced_data[f"idf_{i}"] = idf
    reduced_data[f"vectorizer_tfidf_{i}"] = vectorizer_tfidf


# %%
from sklearn.decomposition import TruncatedSVD

np.random.seed(32)
for i in size:
    print(i)
    start_time = time.time()
    lsa_obj = TruncatedSVD(n_components=i, n_iter=50, random_state=32)
    tfidf_lsa_data = lsa_obj.fit_transform(X_train)
    stop_time = time.time()
    computational_time[f"lsa_{i}"] = stop_time - start_time
    reduced_data[f"lsa_{i}"] = tfidf_lsa_data
    reduced_data[f"lsa_sigma_{i}"] = lsa_obj.singular_values_
    reduced_data[f"las_v_{i}"] = lsa_obj.components_.T
    reduced_data[f"lsa_obj_{i}"] = lsa_obj

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

np.random.seed(32)
start_time = time.time()
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
X_lda = lda.transform(X_train)
stop_time = time.time()
computational_time[f"lda_{i}"] = stop_time - start_time
reduced_data[f"lda_{i}"] = X_lda
reduced_data[f"lda_sigma_{i}"] = lda.n_components
reduced_data[f"lda_obj_{i}"] = lda

# %%
from keras.layers import Input, Dense
from keras.models import Model

np.random.seed(32)
for i in size:
    print(i)
    start_time = time.time()
    input_layer = Input(shape=(tfidf_df.shape[1],))
    encoder = Dense(i, activation="relu")(input_layer)
    decoder = Dense(tfidf_df.shape[1], activation="sigmoid")(encoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.fit(
        X_train,
        X_train,
        epochs=10,
        batch_size=64,
        shuffle=True,
        validation_data=(X_test, X_test),
    )
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    decoder_model = Model(inputs=input_layer, outputs=decoder)
    X_train_compressed = encoder_model.predict(X_train)
    stop_time = time.time()
    computational_time[f"ae_{i}"] = stop_time - start_time
    reduced_data[f"ae_{i}"] = X_train_compressed
    reduced_data[f"encoder_{i}"] = encoder_model
    reduced_data[f"decoder_{i}"] = decoder_model


# %%
for i in size:
    start_time = time.time()
    tfidf = reduced_data[f"vectorizer_tfidf_{i}"].transform(tf_idf_test)
    stop_time = time.time()
    idf = vectorizer_tfidf.idf_
    computational_time[f"tf-idf_test_{i}"] = stop_time - start_time
    reduced_data[f"tf-idf_test_{i}"] = tfidf
    reduced_data[f"idf_test_{i}"] = idf

    start_time = time.time()
    tfidf_lsa_data = reduced_data[f"lsa_obj_{i}"].transform(X_test)
    stop_time = time.time()
    computational_time[f"lsa_test_{i}"] = stop_time - start_time
    reduced_data[f"lsa_test_{i}"] = tfidf_lsa_data
    reduced_data[f"lsa_sigma_test_{i}"] = reduced_data[f"lsa_obj_{i}"].singular_values_
    reduced_data[f"las_v_test_{i}"] = reduced_data[f"lsa_obj_{i}"].components_.T

    start_time = time.time()
    X_test_compressed = reduced_data[f"encoder_{i}"].predict(X_test)
    stop_time = time.time()
    computational_time[f"ae_test_{i}"] = stop_time - start_time
    reduced_data[f"ae_test_{i}"] = X_test_compressed

start_time = time.time()
lda.fit(X_test, y_test)
X_lda = lda.transform(X_test)
stop_time = time.time()
computational_time[f"lda_test_{i}"] = stop_time - start_time
reduced_data[f"lda_test_data{i}"] = X_lda
reduced_data[f"lda_sigma_test_{i}"] = lda.n_components


# %%
sorted_dict = dict(sorted(computational_time.items(), key=lambda x: x[1], reverse=True))
data_time_dict = {i: j for i, j in sorted_dict.items()}

data = pd.DataFrame(list(data_time_dict.items()), columns=["DRT", "time (s)"])
fig = px.bar(data, x="DRT", y="time (s)", color="DRT", title="DRT Time Comparison")
fig.update_layout(xaxis=dict(dtick=1))
fig.update_layout(showlegend=False)
fig.write_image(f"plots\\time.png")
fig.show()


# %%
from sklearn.metrics.pairwise import cosine_similarity

similar_cosine_dict = {}


def cal_cs(df, real):
    # concatenate the rows of the matrices
    vector1 = df.flatten()
    vector2 = real.values.flatten()
    # pad the shorter vector with zeros to match the length of the longer vector
    if len(vector1) > len(vector2):
        vector2 = np.concatenate((vector2, np.zeros(len(vector1) - len(vector2))))
    else:
        vector1 = np.concatenate((vector1, np.zeros(len(vector2) - len(vector1))))

    # calculate the cosine similarity between the two vectors
    similarity = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    return similarity


# for i in size:
#     similar_cosine_dict[f"lsa_cs_train_{i}"] = cal_cs(
#         reduced_data[f"lsa_test_{i}"], X_test
#     )
#     similar_cosine_dict[f"lsa_cs_test_{i}"] = cal_cs(reduced_data[f"lsa_{i}"], X_train)
#     similar_cosine_dict[f"ae_cs_train_{i}"] = cal_cs(reduced_data[f"ae_{i}"], X_train)
#     similar_cosine_dict[f"ae_cs_test_{i}"] = cal_cs(
#         reduced_data[f"ae_test_{i}"], X_test
#     )
#     similar_cosine_dict[f"tfidf_cs_test_{i}"] = cal_cs(
#         reduced_data[f"tf-idf_test_{i}"].toarray(), X_test
#     )
#     similar_cosine_dict[f"tfidf_cs_train_{i}"] = cal_cs(
#         reduced_data[f"tf-idf_{i}"].toarray(), X_train
#     )

for i in size:
    X_train_reconstructed = reduced_data[f"lsa_obj_{i}"].inverse_transform(
        reduced_data[f"lsa_{i}"]
    )
    similar_cosine_dict[f"lsa_mse_{i}"] = cosine_similarity(
        X_train.values, X_train_reconstructed
    ).mean()
    X_test_reconstructed = np.dot(
        (reduced_data[f"lsa_test_{i}"]), reduced_data[f"lsa_obj_{i}"].components_
    )
    similar_cosine_dict[f"lsa_mse_test_{i}"] = cosine_similarity(
        X_test.values, X_test_reconstructed
    ).mean()

    train_recon_loss = cosine_similarity(
        X_train.values, reduced_data[f"decoder_{i}"].predict(X_train)
    ).mean()
    similar_cosine_dict[f"ae_mse_{i}"] = train_recon_loss

    test_recon_loss = cosine_similarity(
        X_test.values, reduced_data[f"decoder_{i}"].predict(X_test)
    ).mean()
    similar_cosine_dict[f"ae_mse_test_{i}"] = test_recon_loss

# similar_cosine_dict[f"lda_cs_test_{i}"] = cal_cs(
#     reduced_data[f"lda_test_data{i}"], X_test
# )
# similar_cosine_dict[f"lda_cs_train_{i}"] = cal_cs(reduced_data[f"lda_{i}"], X_train)

sorted_dict = dict(
    sorted(similar_cosine_dict.items(), key=lambda x: x[1], reverse=True)
)
data_time_dict = {i: j for i, j in sorted_dict.items()}

data = pd.DataFrame(list(data_time_dict.items()), columns=["DRT", "score"])
fig = px.bar(data, x="DRT", y="score", color="DRT", title="DRT Cosine Similarity")
fig.update_layout(xaxis=dict(dtick=1))
fig.update_layout(showlegend=False)
fig.write_image(f"plots\\cs.png")
fig.show()


# %%
from sklearn.metrics import mean_squared_error


reconstruction_error_dict = {}
for i in size:
    X_train_reconstructed = reduced_data[f"lsa_obj_{i}"].inverse_transform(
        reduced_data[f"lsa_{i}"]
    )
    reconstruction_error_dict[f"lsa_mse_{i}"] = mean_squared_error(
        X_train.values, X_train_reconstructed
    )
    X_test_reconstructed = np.dot(
        (reduced_data[f"lsa_test_{i}"]), reduced_data[f"lsa_obj_{i}"].components_
    )
    reconstruction_error_dict[f"lsa_mse_test_{i}"] = mean_squared_error(
        X_test.values, X_test_reconstructed
    )

    train_recon_loss = mean_squared_error(
        X_train.values, reduced_data[f"decoder_{i}"].predict(X_train)
    )
    reconstruction_error_dict[f"ae_mse_{i}"] = train_recon_loss

    test_recon_loss = mean_squared_error(
        X_test.values, reduced_data[f"decoder_{i}"].predict(X_test)
    )
    reconstruction_error_dict[f"ae_mse_test_{i}"] = test_recon_loss

# %%

sorted_dict = dict(
    sorted(reconstruction_error_dict.items(), key=lambda x: x[1], reverse=True)
)
data_time_dict = {i: j for i, j in sorted_dict.items()}

data = pd.DataFrame(list(data_time_dict.items()), columns=["DRT", "score"])
fig = px.bar(
    data, x="DRT", y="score", color="DRT", title="DRT Reconstruction Error (MSE)"
)
fig.update_layout(xaxis=dict(dtick=1))
fig.update_layout(showlegend=False)
fig.write_image(f"plots\\re.png")
fig.show()

# %%
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

np.random.seed(32)
clf = SVC()
rfc = RandomForestClassifier()


accuracy_dict = {}
for i in size:
    clf.fit(reduced_data[f"tf-idf_{i}"], y_train)
    accuracy = clf.score(reduced_data[f"tf-idf_test_{i}"], y_test)
    accuracy_dict[f"tf-idf_svm_{i}"] = accuracy

    # accuracy = clf.score(reduced_data[f"tf-idf_{i}"], y_train)
    # accuracy_dict[f"tf-idf_svm_tr_{i}"] = accuracy

    rfc.fit(reduced_data[f"tf-idf_{i}"], y_train)
    accuracy = rfc.score(reduced_data[f"tf-idf_test_{i}"], y_test)
    accuracy_dict[f"tf-idf_rf_{i}"] = accuracy

    # accuracy = rfc.score(reduced_data[f"tf-idf_{i}"], y_train)
    # accuracy_dict[f"tf-idf_rf_tr_{i}"] = accuracy

    clf.fit(reduced_data[f"lsa_{i}"], y_train)
    accuracy = clf.score(reduced_data[f"lsa_test_{i}"], y_test)
    accuracy_dict[f"lsa_svm_{i}"] = accuracy

    # clf.fit(reduced_data[f"lsa_{i}"], y_train)
    # accuracy = clf.score(reduced_data[f"lsa_{i}"], y_train)
    # accuracy_dict[f"lsa_svm_tr_{i}"] = accuracy

    rfc.fit(reduced_data[f"lsa_{i}"], y_train)
    accuracy = rfc.score(reduced_data[f"lsa_test_{i}"], y_test)
    accuracy_dict[f"lsa_rf_{i}"] = accuracy

    # accuracy = rfc.score(reduced_data[f"lsa_{i}"], y_train)
    # accuracy_dict[f"lsa_rf_tr_{i}"] = accuracy

    clf.fit(reduced_data[f"ae_{i}"], y_train)
    accuracy = clf.score(reduced_data[f"ae_test_{i}"], y_test)
    accuracy_dict[f"ae_svm_{i}"] = accuracy

    # accuracy = clf.score(reduced_data[f"ae_{i}"], y_train)
    # accuracy_dict[f"ae_svm_tr_{i}"] = accuracy

    rfc.fit(reduced_data[f"ae_{i}"], y_train)
    accuracy = rfc.score(reduced_data[f"ae_test_{i}"], y_test)
    accuracy_dict[f"ae_rf_{i}"] = accuracy

    # accuracy = rfc.score(reduced_data[f"ae_{i}"], y_train)
    # accuracy_dict[f"ae_rf_tr_{i}"] = accuracy


clf.fit(reduced_data[f"lda_{i}"], y_train)
accuracy = clf.score(reduced_data[f"lda_test_data{i}"], y_test)
accuracy_dict[f"lda_svm_{i}"] = accuracy

# accuracy = clf.score(reduced_data[f"lda_{i}"], y_train)
# accuracy_dict[f"lda_svm_tr_{i}"] = accuracy

rfc.fit(reduced_data[f"lda_{i}"], y_train)
accuracy = rfc.score(reduced_data[f"lda_test_data{i}"], y_test)
accuracy_dict[f"lda_rf_{i}"] = accuracy

# rfc.fit(reduced_data[f"lda_{i}"], y_train)
# accuracy = rfc.score(reduced_data[f"lda_{i}"], y_train)
# accuracy_dict[f"lda_rf_tr_{i}"] = accuracy
# %%
sorted_dict = dict(sorted(accuracy_dict.items(), key=lambda x: x[1], reverse=True))
data_time_dict = {i: j for i, j in sorted_dict.items()}

data = pd.DataFrame(list(data_time_dict.items()), columns=["DRT", "score"])
fig = px.bar(data, x="DRT", y="score", color="DRT", title="Classification Accuracy")
fig.update_layout(xaxis=dict(dtick=1))
fig.update_layout(showlegend=False)
fig.write_image(f"plots\\ca.png")
fig.show()
# %%
