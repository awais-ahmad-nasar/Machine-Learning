#Import Libraries
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
# Load the SMS spam dataset
message_data = pd.read_csv("B:\MY Documents\spam.csv", encoding="latin")
print(message_data.head())

# Rename columns for clarity
message_data = message_data.rename(columns={'v1': 'Spam/Not_Spam', 'v2': 'message'})

# Data Cleaning
message_data_copy = message_data['message'].copy()

# Text Preprocessing
def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

message_data_copy = message_data_copy.apply(text_preprocess)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
message_mat = vectorizer.fit_transform(message_data_copy)

# Split the data into training and testing sets
message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(
    message_mat, message_data['Spam/Not_Spam'], test_size=0.3, random_state=20
)

# Train Multinomial Naive Bayes classifier
Spam_model = MultinomialNB()
Spam_model.fit(message_train, spam_nospam_train)

# Make predictions and calculate accuracy
pred = Spam_model.predict(message_test)
confusion = confusion_matrix(spam_nospam_test, pred)
accuracy = accuracy_score(spam_nospam_test, pred)*100

color = sns.color_palette(["lightgreen", "red"])

# Display the confusion matrix as a visual image
sns.heatmap(confusion, annot=True, fmt='d', cmap=color)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\n------------------------------------------------------")
print("------------------------------------------------------")

print(f'Accuracy: {accuracy:.2f}%')