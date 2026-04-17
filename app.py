import streamlit as st
import pickle
import numpy as np  

#Load the saved model
model = pickle.load(open (r'D:\FSDS  with   GEN AI   And Agent AI\Mera Maal\SIMPLE LINEAR REGRESSION\Linear_regression_model.pkl','rb'))
#set the title of the streamlit App
st.title ("salary prediction app")
#Add a brief description
st.write("This app predict the salary based on year on experience  using a simple linear regression model .")
#Add input widget for user to enter years of experiance 
years_experiance= st.number_input ("Enter  year of experiance :",min_value =0.0, max_value =50.0, value=1.0,step=0.5)
#When the button is clicked ,make predictions
if st.button("prediction slary "):
     #Make a prediction using the trained model
     experience_input=np.array([[years_experiance]]) #convert the input to a 2D Array prediction
     prediction =model.predict(experience_input)
     #Display the result
     st.success(f"The predicted salary for {years_experiance} years of experiance is : ${prediction[0]:,.2f}")
#Display information about the model
st.write("The model was trained using a dataset of salaries and years of   experience.built model by Pravashis Biswal")