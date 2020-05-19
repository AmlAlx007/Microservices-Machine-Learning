from flask import Flask, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import pandas as pd
import math
import pickle
import random

app = Flask(__name__)

config = {
    'user': 'admin',
    'password': 'fAgitod3v',
    'host': 'fagitodev.cw0t204dkdrj.eu-west-1.rds.amazonaws.com',
    'database': 'fagito'
}

model = pickle.load(open('logmodel.pkl', 'rb'))


@app.route('/orderTime', methods=['POST'])
def home():
    request_object = request.json
    df = pd.read_csv("./data.csv")
    df = df.drop(columns=["Unnamed: 0", "kitchen_food_prep_time"])
    distance = []
    for val in df.values:
        dist = calculate_distance(val[0], val[1], val[2], val[3])
        distance.append(dist)

    df = df.drop(columns=['cust_latitude'])
    df['distance'] = distance

    X = (df.iloc[:, 0:4]).values
    y = df.iloc[:, 4].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)

    dist = calculate_distance(request_object["rest_latitude"], request_object["rest_longitude"], request_object["cust_latitude"], request_object["cust_longitude"])

    s = [[request_object["rest_latitude"], request_object["rest_longitude"], request_object["cust_longitude"], dist]]

    order_time_result = logmodel.predict(s)
    str = np.array_str(order_time_result)
    str = str.strip("[")
    str = str.strip("]")
    return str


def calculate_distance(*val):

    theta = val[1] - val[3]
    dist = math.sin(math.radians(val[0])) * math.sin(math.radians(val[2])) + math.cos(
                math.radians(val[0])) * math.cos(math.radians(val[2])) * math.cos(math.radians(theta))
    dist = math.acos(dist)
    dist = math.degrees(dist)
    dist = dist * 60 * 1.1515
    dist = dist * 1.609344
    dist = dist / 100
    return dist


if __name__ == "__main__":
    app.run(port=80, host="0.0.0.0")
