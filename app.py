from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=["GET","POST"])
def submit():
    datas = request.form.get("data")
    return render_template("sa.html", data=datas)


if __name__ == "__main__":
    app.run(debug=True)