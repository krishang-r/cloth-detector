from flask import Flask, render_template, request, url_for
import os

app = Flask(__name__)

app.config['SECRET_KEY'] = os.urandom(24)

list_of_styles = []

@app.route("/", methods = ['GET', 'POST'])
@app.route("/home", methods = ['GET', 'POST'])
def home():
    global list_of_styles

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug = True)