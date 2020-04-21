from flask import Flask,render_template,request
from src.get_reviews import get_review
# from Review_Ranker import get_review

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/result')
def result():
    url = request.args.get('userurl')
    df = get_review(url)
    df=df.to_html()
    return render_template('result.html',df=df)

@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True,host="127.0.0.1", port=5000)