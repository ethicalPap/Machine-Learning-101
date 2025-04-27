# imports 
import streamlit as st
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# load our csv and save it to memory
@st.cache_data
def load_data():
    df = pd.read_csv('iris_dataset.csv')
    encoded_data = df.copy()
    encoded_data['species_encoded'] = encoded_data['species'].map({'setosa': 1, 'versicolor': 2, 'virginica': 3})

    # define our features
    X = encoded_data.drop(['species_encoded', 'species'], axis=1)
    y = encoded_data['species_encoded']
    return X, y, encoded_data['species'].unique()

# define feature, label, and label name
features, target, target_names = load_data()

# init scaler
scaler = StandardScaler()
feature_scaled = scaler.fit_transform(features)

# define our model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(feature_scaled, target)

# add classification sliders
feature_names = features.columns.to_list()
if len(feature_names) >= 4:
    # add slide bar with column anmes
    st.sidebar.title("Iris Flower Features")

    feature_value = []
    for i, name in enumerate(feature_names[:4]):
        feature_value.append(
            st.sidebar.slider(f"{name}",
                            float(features[name].min()),
                            float(features[name].max()),
                            float(features[name].mean()))
        )

    # create input data.

    input_data = [feature_value]
    input_df = pd.DataFrame([feature_value], columns=feature_names[:4])
    input_data_scaled = scaler.transform(input_df)

    # make predictions
    prediction = model.predict(input_data_scaled)

    # convert predictions to species names
    species_map = {1: 'setosa', 2: 'versicolor', 3: 'virginica'}
    predicted_iris_species = species_map[prediction[0]]

    # output the prediction to the app
    st.title("Iris Prediction")
    st.write(f"**Predicted Species:** *{predicted_iris_species}* :sunglasses:")

    # show the probabilities (sigmoid output)
    probabilities = model.predict_proba(input_data_scaled)[0]
    st.write(f"**Prediction Confidence**")
    st.write(f"**Setosa:** *{probabilities[0]:.2f}*")
    st.write(f"**Versicolor:** *{probabilities[1]:.2f}*")
    st.write(f"**Virginica:** *{probabilities[2]:.2f}*")

    # also output a picture for the prediction (optional)
    if predicted_iris_species == 'setosa':
        st.image("./images/setosa.jpg", caption="Iris Setosa")
    elif predicted_iris_species == 'versicolor':
        st.image("./images/versicolor.jpg", caption="Iris Versicolor")
    elif predicted_iris_species == 'virginica':
        st.image("./images/virginica.jpg", caption="Iris Virignica")
else:
    st.error("There are not enough features in the dataset. Expect at least 4.")