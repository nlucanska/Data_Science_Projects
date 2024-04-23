from flask import Flask, jsonify, request, render_template
from utilities import predict_churn
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=['POST'])
def predict():

    if request.method == 'POST':
        customer = request.form.to_dict()
        prediction = predict_churn(customer)

        # data to dataframe
        df = pd.Series(customer, index=customer.keys())

    return render_template("result.html", prediction = prediction, customer = df)

    # or    
    #return jsonify({'result': prediction, 'customer' : customer}) 


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)