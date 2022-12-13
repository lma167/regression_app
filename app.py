import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
#cmatrix shows how many predictions are correct

#add header to describe app 
st.markdown("#Welcome to my Logist Regression App: Predicitng a Linkedin User")

#user inputs

#inget data
s = pd.read_csv("social_media_usage.csv")
 

#clean Data
def clean_sm(input):
    x = np.where(input == 1, 1, 0)
    return x

toy_df = pd.DataFrame({
    "identifier": [1,2,3],
    "input": [1,1,0]
})

toy_df_test = pd.DataFrame({
    "x": clean_sm(toy_df.input)
})



#ss dataframe 
ss = pd.DataFrame({
    "sm_li": clean_sm(s.web1h),
    "income": np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent": clean_sm(s.par ),
    "married": clean_sm(s.marital),
    "female": np.where(input == 2, 1, 0),
    "age":np.where(s["age"] > 98, np.nan, s["age"]),
})


#dropna
ss = ss.dropna()


#create target and feature set 
y = ss["sm_li"]
X = ss[["income",
    "education",
    "parent",
    "married",
    "female",
    "age"]]


#split into train and test 
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility


#initiate lr model
lr = LogisticRegression(class_weight = "balanced")
# Fit algorithm to training data
#passing the data to our model
#we fit the model - we don't know yet how it performs
lr.fit(X_train, y_train)

#evaluate model performance
y_pred = lr.predict(X_test)




income = st.selectbox("Select Income",[1,2,3,4,5,6,7,8,9])
eduction = st.selectbox("Select Education Level",[1,2,3,4,5,6,7,8])
parent = st.radio("Parent? (1 indicates Yes)",[1,0])
married = st.radio("Married? (1 indicates Yes)",[1,0])
female = st.radio("Female? (1 indicates Yes)",[1,0])
age = st.slider("What is your age?", 0,98)
#use model to make predictions
newdata =[income,eduction,parent,married,female,age]
# Predict class, given input features classified as 1 or 0 if prob is over 0.5
predicted_class = lr.predict([newdata])

# Generate probability of positive class (=1)
probs = lr.predict_proba([newdata])

st.markdown(f"Predicted class: **{predicted_class[0]}**") # 0=not pro-environment, 1=pro-envronment
st.markdown(f"Probability that this person is pro-environment: **{probs[0][1]}**")


         