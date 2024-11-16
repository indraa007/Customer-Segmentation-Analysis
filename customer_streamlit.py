import numpy as np
import joblib
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate some example data for demonstration
X, y = make_classification(n_samples=100, n_features=9, n_informative=3, n_classes=3, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example of defining and training model_knn
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)  # Fit the model with training data

# Save the trained KNN model
joblib.dump(model_knn, 'knn_model.joblib')

# Load the trained KNN model
model = joblib.load('knn_model.joblib')

# Define the Streamlit app
def main():
    st.title('KNN Cluster Predictor')

    education = st.selectbox('Education', ['1', '2', '3', '4'])
    marital_status = st.selectbox('Marital Status', ['0', '1'])
    income = st.number_input('Income')
    kids = st.number_input('Kids')
    expenses = st.number_input('Expenses')
    totalacceptedcmp = st.number_input('Total Accepted CMP')
    numtotalpurches = st.number_input('Num Total Purches')
    customer_age = st.number_input('Customer Age')
    customer_for = st.number_input('Customer For')

    data_point = [education, marital_status, income, kids, expenses, totalacceptedcmp, numtotalpurches, customer_age, customer_for]

    # Convert all values in data_point to float
    data_point = [float(i) for i in data_point]

    if st.button('Predict Cluster'):
        cluster = predict_cluster(data_point, model)
        st.write('Predicted Cluster:', cluster)

def predict_cluster(data_point, model):
    data_point_array = np.asarray(data_point)
    data_point_reshaped = data_point_array.reshape(1, -1)
    cluster = model.predict(data_point_reshaped)[0]
    return cluster

if __name__ == '__main__':
    main()
