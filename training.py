import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

print("1. Membaca dataset")
df = pd.read_excel('youtube-comment-groundtruth.xlsx')
df = df.dropna(subset=['komentar_bersih', 'label'])

df.to_csv('dataset_dashboard.csv', index=False)
print("Dataset bersih disimpan sebagai 'dataset_dashboard.csv'")

X = df['komentar_bersih'].astype(str)
y = df['label']
labels_list = sorted(y.unique())

print("2. Memproses teks menjadi angka (TF-IDF)")
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

print("3. Membagi data latih dan data uji")
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels_list)
    }

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("4. Melatih algoritma Naive Bayes")
model_nb = MultinomialNB()
model_nb.fit(X_train_smote, y_train_smote)
y_pred_nb = model_nb.predict(X_test)
metrics_nb = calculate_metrics(y_test, y_pred_nb)
print(f"Akurasi Naive Bayes: {metrics_nb['accuracy'] * 100:.2f}%")

print("5. Melatih algoritma Support Vector Machine (SVM)")
model_svm = SVC(kernel='linear', probability=True) 
model_svm.fit(X_train_smote, y_train_smote)
y_pred_svm = model_svm.predict(X_test)
metrics_svm = calculate_metrics(y_test, y_pred_svm)
print(f"Akurasi SVM: {metrics_svm['accuracy'] * 100:.2f}%")

print("6. Menyimpan model menjadi file .pkl")
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

with open('model_nb.pkl', 'wb') as file:
    pickle.dump(model_nb, file)

with open('model_svm.pkl', 'wb') as file:
    pickle.dump(model_svm, file)

evaluation_results = {
    'labels': labels_list,
    'nb': metrics_nb,
    'svm': metrics_svm
}
with open('eval_metrics.pkl', 'wb') as file:
    pickle.dump(evaluation_results, file)
    
print("\nSelesai.")