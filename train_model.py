import pickle
import string
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    tokens = []
    for token in text:
        if token.isalnum():
            tokens.append(token)

    text = tokens[:]
    tokens.clear()

    for token in text:
        if token not in stopwords.words('english') and token not in string.punctuation:
            tokens.append(token)

    text = tokens[:]
    tokens.clear()

    stemmer = PorterStemmer()
    for token in text:
        tokens.append(stemmer.stem(token))

    return ' '.join(tokens)


def main():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    spam = pd.read_csv('data/spam.csv', encoding='latin1')
    spam = spam[['v1', 'v2']].rename(columns={'v1': 'target', 'v2': 'text'})
    spam['transformed_text'] = spam['text'].apply(transform_text)

    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(spam['transformed_text']).toarray()
    y = (spam['target'] == 'spam').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print('accuracy', accuracy_score(y_test, predictions))
    print('precision', precision_score(y_test, predictions))

    Path('model').mkdir(exist_ok=True)
    with open('model/vectorizer.pkl', 'wb') as file_handle:
        pickle.dump(vectorizer, file_handle)
    with open('model/model.pkl', 'wb') as file_handle:
        pickle.dump(model, file_handle)

    check_vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
    check_model = pickle.load(open('model/model.pkl', 'rb'))
    sample = check_model.predict(
        check_vectorizer.transform([transform_text('lottery for spam')]).toarray()
    )[0]
    print('reloaded', hasattr(check_model, 'classes_'))
    print('sample', int(sample))


if __name__ == '__main__':
    main()