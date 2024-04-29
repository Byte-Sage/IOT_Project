import streamlit as st
import pickle
import joblib
from sklearn.preprocessing import StandardScaler

def rec_mq7():
    import requests
    url = 'https://api.thingspeak.com/channels/2220148/fields/1.json?results=2'  # Replace this with your JSON API URL
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the JSON response
        json_data = response.json()
        return json_data['feeds'][0]['field1']  # Print the JSON data
    else:
        print('Failed to fetch data:', response.status_code)

def rec_mq2():
    import requests
    url = 'https://api.thingspeak.com/channels/2215347/fields/1.json?results=2'  # Replace this with your JSON API URL
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the JSON response
        json_data = response.json()
        return json_data['feeds'][0]['field1']  # Print the JSON data
    else:
        print('Failed to fetch data:', response.status_code)

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    scaler = joblib.load('scaler_model.joblib')
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Loaded scaler object is not an instance of StandardScaler")
    print("Type of scaler:", type(scaler))

except Exception as e:
    st.error(f"An error occurred while loading the model or scaler: {e}")
    st.stop()

def main():
    st.title('Classifying whether the gas is Hazardous or Safe')
    val1 = rec_mq7()
    val2 = rec_mq2()
    # input1 = st.number_input('MQ7', value=0.0)
    input1 = val1
    # input2 = st.number_input('MQ8', value=0.0)
    input2 = val2
    scaled_inputs = scaler.transform([[input1, input2]])    
    
    if st.button('Prediction'):
        try:
            st.write("MQ7 ",val1)
            st.write("MQ2 ",val2)
            prediction = model.predict(scaled_inputs)[0]
        
            if prediction == 1:
                st.write('Prediction: Hazardous')
            else:
                st.write('Prediction: Safe')
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
