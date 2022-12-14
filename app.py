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
st.markdown("# Welcome to my Logist Regression App: Predicitng a Linkedin User")
st.markdown("## Please answer the following questions")
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
    "parent": np.where(s.par == 2, False, np.where(s.par < 2, True, np.nan)),
    "married": np.where(s["marital"] > 6, np.nan, np.where(s.marital < 2, True, False)),
    "female": np.where(s.gender == 2, True, np.where(s.gender < 4, False, np.nan)),
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




income = st.selectbox("Select Income",["<$10K",
                                       "$10K-$20K",
                                       "$20K-$30K",
                                       "$30K-$40K",
                                       "$40K-$50K",
                                       "$50K-$75K",
                                       "$75K-$100K",
                                       "$100K-$150K",
                                       ">=$150K"])
if income =="<$10K":
 income = 1
elif income == "$10K-$20K":
 income = 2
elif income == "$20K-$30K":
 income = 3
elif income == "$30K-$40K":
 income = 4 
elif income == "$40K-$50K":
 income = 5
elif income == "$50K-$75K":
 income = 6
elif income == "$75K-$100K":
 income = 7
elif income == "$100K-$150K":
 income = 8
else:
 income =9

education = st.selectbox("Select Education Level",["Less than Highschool",
                                                  "Highschool Incomplete",
                                                  "Highschool Graduate",
                                                  "Some college, no degree",
                                                  "Two-year associate degree from a college or university",
                                                  "Four-year college or university degree/Bachelor's degree",
                                                  "Some post-graduate or professional schooling, no postgrad degree",
                                                  "Postgraducation or professional degree (masters, medical, doctorate, or law)"])
if education =="Less than Highschool":
 education = 1
elif education == "Highschool Incomplete":
 education = 2
elif education == "Highschool Graduate":
 education = 3
elif education == "Some college, no degree":
 education = 4 
elif education == "Two-year associate degree from a college or university":
 education = 5
elif education == "Four-year college or university degree/Bachelor's degree":
 education = 6
elif education == "Some post-graduate or professional schooling, no postgrad degree":
 education = 7
else:
 education = 8

parent = st.radio("Parent?",["Yes","No"])
if parent == "Yes":
 parent = 1
else:
 parent = 0
 

married = st.radio("Married?",["Yes","No"])
if married == "Yes":
 married =1 
else:
 married = 0


female = st.radio("Female?",["Yes","No"])
 
if female == "Yes":
 female =1 
else:
 female = 0
 
age = st.slider("What is your age?", 0,98)
#use model to make predictions
newdata =[income,education,parent,married,female,age]
# Predict class, given input features classified as 1 or 0 if prob is over 0.5
predicted_class = lr.predict([newdata])
if predicted_class[0] == 1:
 predicted_class = "Linkedin user"
else: predicted_class = "Not a Linkedin user"

# Generate probability of positive class (=1)
probs = lr.predict_proba([newdata])

probins = round(probs[0][1],4)*100


    
 




if st.button("Predict"):
    st.markdown(f"Predicted class: **{predicted_class}**") # 0=not pro-environment, 1=pro-envronment
    st.markdown(f"Probability that this person is a linkedin user is: **{probins}%**")
     
else:
    st.markdown("Awaiting prediction")




         