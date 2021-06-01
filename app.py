from flask import Flask , render_template , request
import numpy as np
import pickle

filename = 'sentiment.pkl'
classifier = pickle.load(open(filename,'rb'))
cv = pickle.load(open('text_convert1.pkl','rb'))

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    Review = request.form['Review']
    data = [Review]
    vect = cv.transform(data).toarray() 

    my_prediction = classifier.predict(vect)

    return render_template('predict.html', prediction = my_prediction)

if __name__=='__main__':
    app.run(debug=True)
