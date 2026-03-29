import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

st.title("Spaceship Titanic Model Deployment")
st.write("Please input your data here to predict whether you will be transported or not.")

@st.cache_resource
def load_pipeline():
    return joblib.load(Path("artifacts/pipeline.pkl"))

model = load_pipeline()

PassengerId = st.text_input("Passenger ID", "0003_01")

HomePlanet = st.selectbox("Home Planet", ["Earth", "Europa", "Mars"])

CryoSleep = st.selectbox("CryoSleep", [True, False])

Cabin = st.text_input("Cabin", "A/0/S")

Destination = st.selectbox(
    "Destination",
    ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]
)

Age = st.number_input("Age", min_value=0, max_value=100, value=20)

VIP = st.selectbox("VIP", [True, False])

RoomService = st.number_input("Room Service", min_value=0.0, value=2.0)
FoodCourt = st.number_input("Food Court", min_value=0.0, value=1.0)
ShoppingMall = st.number_input("Shopping Mall", min_value=0.0, value=0.0)
Spa = st.number_input("Spa", min_value=0.0, value=10.0)
VRDeck = st.number_input("VR Deck", min_value=0.0, value=2.0)

Name = st.text_input("Name", "Altark Susent")

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "PassengerId": PassengerId,
        "HomePlanet": HomePlanet,
        "CryoSleep": CryoSleep,
        "Cabin": Cabin,
        "Destination": Destination,
        "Age": Age,
        "VIP": VIP,
        "RoomService": RoomService,
        "FoodCourt": FoodCourt,
        "ShoppingMall": ShoppingMall,
        "Spa": Spa,
        "VRDeck": VRDeck,
        "Name": Name
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.metric("Transported Probability", f"{probability*100:.2f}%")

    if prediction:
        st.success("Passenger will be Transported")
    else:
        st.error("Passenger will NOT be Transported")