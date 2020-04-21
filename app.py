from flask import Flask,render_template,request
# from src.get_reviews import get_review

# For Checking Locally
from Review_Ranker import * 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/result')
def result():
    url = request.args.get('userurl')
    df = get_review(url)
    df=features(df)
    X,y=predictor(df)
    df['y_pred']=rank(X,y)
    df=df.sort_values(by='y_pred',ascending=False)
    positive = df[df.Sentiment=="pos"]
    negative = df[df.Sentiment=="neg"]
    positive = positive.to_html(columns=["Review_Text"],index=False,classes=["table"],header=False).replace('\\n','<br>')
    negative = negative.to_html(columns=["Review_Text"],index=False,classes=["table"],header=False).replace('\\n','<br>')
    # Render template expects a df with one coloumn, ranked df which will be displayed
    return render_template('result.html',positive=positive,negative=negative)

@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True,host="127.0.0.1", port=5000)