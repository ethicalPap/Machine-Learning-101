import streamlit as st
from sklearn.linear_model import Perceptron
from sklearn import datasets

# app for classification based on 2 features only (petal length and petal width)

# load data into cache for app
@st.cache_data
def load_data():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    return X, y, iris.target_names

features, target, target_names = load_data()

# Train data
model=Perceptron()
model.fit(features, target)

# Add classification sliders
st.sidebar.title("Iris Features")
petal_length = st.sidebar.slider("Petal length", float(features[:, 0].min()), float(features[:, 0].max()))
petal_width = st.sidebar.slider("Petal width", float(features[:, 0].min()), float(features[:, 0].max()))

input_data = [[petal_length, petal_width]]

# make prediction based on the feature sliders
prediction = model.predict(input_data)
predicted_iris_species = target_names[prediction[0]]

# output prediction
st.title("Iris Prediction")
st.write(f"Predicted Species: **{predicted_iris_species}** :sunglasses:")

# also output a picture for the prediction
if predicted_iris_species == 'setosa':
    st.image("./images/setosa.jpg", caption="Iris Setosa")
elif predicted_iris_species == 'versicolor':
    st.image(".images/versicolor.jpg", caption="Iris Versicolor")
elif predicted_iris_species == 'virginica':
    st.image("./images/virginica.jpg", caption="Iris Virginica")