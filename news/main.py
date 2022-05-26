from flask import *
import pandas as pd
#import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score,precision,recall


d=pd.read_csv(r'static/finalproject.csv')


app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=["POST","GET"])
def home():
    catego = ""
    if request.method=="POST":
        tle = request.form['title']
        #return tle
        
        a=[]
        for i in range(len(d)):
            a.append(d.loc[i])
        b=[]
        for i in a:
            b.append(str(list(i)).split(':'))

        x=[]
        y=[]
        for i in range(len(b)):
            x.append(b[i][1])
            y.append(b[i][6])

        x1=[]
        y1=[]
        for i in range(len(x)):
            x1.append(x[i].split(',')[0][2:-1])
            y1.append(y[i][2:-8])
            
        data=pd.DataFrame()
        data['category']=x1
        data['title']=y1
        """----"""

        dt = data
        print(dt.head())

        dt.isnull().sum()

        dt["category"].value_counts()

        dt = dt[["title", "category"]]

        x = np.array(dt["title"])
        y = np.array(dt["category"])

        cv = CountVectorizer()
        X = cv.fit_transform(x)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


        model = RandomForestClassifier()
        model.fit(X_train,y_train)
        ypred=model.predict(X_test)

        print(accuracy_score(y_test,ypred))
        precision = sklearn.metrics.precision_score(y_test, y_pred, pos_label="positive")
        print(precision)

        with open('sample.obj','wb') as  fp:
            pickle.dump(model,fp)
        with open('sample.obj','rb') as fp:
            model=pickle.load(fp)

        user = tle
        dt = cv.transform([user]).toarray()
        output = model.predict(dt)
        print(output)
        catego = output
        return catego
    return render_template("index.html", data=catego)

if __name__ == "__main__":
    app.run(debug=True)
