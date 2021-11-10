from sklearn.feature_extraction.text import CountVectorizer;

documents = ["I love Lilies", "Lilies are so beautiful", "Lilies are Lilies "]

# create a Vectorizer Object
vectorizer = CountVectorizer()
vectorizer.fit(documents)

# return a dict, key: index, value: word
print(vectorizer.vocabulary_)

# encode the document
vector = vectorizer.transform(documents)
print(vector.toarray())


