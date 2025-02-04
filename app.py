from flask import Flask, render_template, request
import os
from cnnClassifier.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')

@app.route("/login")
def login():
    return render_template('login.html')   

@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def submit():
    predict_result = None
    img_path = None

    if request.method == 'POST':
        img = request.files['my_image']
        model_name = request.form['model']
        print(f"Selected Model: {model_name}")

        img_path = os.path.join("static", "tests", img.filename)
        img.save(img_path)

        if model_name == 'MobileNetV2':
            pipeline = PredictionPipeline(img_path)
            predict_result = pipeline.predict_label()

    return render_template("prediction.html", prediction=predict_result, img_path=img_path, model=model_name)

@app.route("/chart")
def chart():
    return render_template('chart.html')

@app.route("/performance")
def performance():
    return render_template('performance.html')

if __name__ == '__main__':
    app.run(debug=True)



