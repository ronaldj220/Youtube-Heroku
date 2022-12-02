from flask import Flask, request, render_template, session
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key='login'


# Kembalikan dalam bentuk index
@app.route('/')
def index():
    return render_template('index.html')

# Untuk login
data={'Admin':'1234', 'admin':'admin123'}
@app.route('/login',methods=['POST', 'GET'])
def login():
    user = request.form['username']
    pwd = request.form['password']
    if user not in data:
        return render_template('index.html', info='User tidak ada')
    else:
        if data[user] != pwd:
            return render_template('index.html', info='Password tidak ada')
        else:
            return render_template('home.html', name=user)

# Untuk logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return render_template('index.html')

# Untuk melakukan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data
    df = pd.read_csv('YoutubeSpamMergedData.csv')
    # Ambil kolom konten dan kelas
    df_data = df[['CONTENT', 'CLASS']]
    # Gunakan pembagian berdasarkan fitur(df_x) dan label(df_y)
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS
    # Lakukan ekstraksi fitur menggunakan Count Vectorizer
    corpus = df_x
    cv = CountVectorizer()
    # Fit data
    X = cv.fit_transform(corpus)
    # Lakukan pembagian data sebesar 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.30, random_state=42)
    # Implementasi Multinomial Naive bayes
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        prediksi_saya = clf.predict(vect)
        
    return render_template('home.html', prediction = prediksi_saya)

if __name__ == '__main__':
    app.run(debug=True)