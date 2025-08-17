import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,accuracy_score
# download nltk data
nltk.download('stopwords')
nltk.download('punkt')
#step_1:load dataset
df=pd.read_csv('text_data.csv')
print("Dataset samples:\n",df.head())
#step_2: clean and preprocess the data
stop_words=set(stopwords.words('english'))

def clean_text(text):
    text=text.lower()
    text=re.sub(r"http\S+|www\S+|https\S+",'',text,flags=re.MULTILINE)
    text=re.sub(r'\@\w+|\#','',text)
    text=re.sub(r'[^a_zA-Z\s]','',text)
    tokens=nltk.word_tokenize(text)
    filtered=[word for word in tokens if word not in stop_words]
    return''.join(filtered)
df['clean_text']=df['text'].apply(clean_text)

#step_3: Vectorize the text
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df['clean_text'])
y=df['label']

#step_4: train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,
test_size=0.2,random_state=42)
#step_5: train a simple model
model=MultinomialNB()
model.fit(X_train,y_train)
#step_6: Evaluate the model
y_pred=model.predict(X_test)
print("\nAccuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))

def predict_sentiment(new_text):
    cleaned=clean_text(new_text)
    vect=vectorizer.transform([cleaned])
    return model.predict(vect)[0]
print("\nSample Prediction:")
sample="horrible airline"
print(f"Text: {sample}")
print(f"Predicted Sentiment:{predict_sentiment(sample)}")
