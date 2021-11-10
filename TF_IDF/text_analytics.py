from sklearn.feature_extraction.text import TfidfVectorizer

d0 = 'Geeks for geeks'
d1 = 'Geeks'
d2 = 'r2j'

string = [d0, d1, d2]

tfidf = TfidfVectorizer()
result = tfidf.fit_transform(string)

for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
    print(ele1, ':', ele2)

print(tfidf.vocabulary_)
