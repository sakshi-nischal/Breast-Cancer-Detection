import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
@st.cache
def load_data():
    cancer_data = load_breast_cancer()
    
    df = pd.DataFrame(np.c_[cancer_data['data'],cancer_data['target']],columns = np.append(cancer_data['feature_names'],['target']))
    
    return cancer_data['DESCR'] , df

desc , df = load_data()
st.write("Data used for training")
st.write(df)
if st.checkbox("Show data DESC"):
    st.write(desc)
    
def model_selection():
    model_options=['Logistic','SVM','Navie Bayes','KNN','Decision Tree','XG_boost']
    model_type = st.selectbox('Select the model',model_options)
    if model_type == 'Logistic':
        model = pickle.load(open('Model/logistic_model.pickle','rb'))
    elif model_type=='SVM':
        model = pickle.load(open('Model/svm_model.pickle','rb'))
    elif model_type=='Navie Bayes':
        model = pickle.load(open('Model/nb_model.pickle','rb'))
    elif model_type=='KNN':
        model = pickle.load(open('Model/Knn_model.pickle','rb'))
    elif model_type=='Decision Tree':
        model = pickle.load(open('Model/DecisionTree_model.pickle','rb'))
    elif model_type=='XG_boost':
        model = pickle.load(open('Model/XG_boost.pickle','rb'))
    return model

model = model_selection()
st.write(model)
st.write(df.columns)
mean_radius = st.number_input("mean radius")
mean_texture = st.number_input("mean texture")
mean_perimeter = st.number_input("mean perimeter")
mean_area = st.number_input("mean area")
mean_smoothness = st.number_input("mean smoothness")
mean_compactness = st.number_input("mean compactness")
mean_concavity = st.number_input("mean concavity")
mean_concave_ponts = st.number_input("mean concave points")
mean_symmetry = st.number_input("mean symmetry")
mean_fractal_dimension = st.number_input("mean fractal dimension")

worst_radius = st.number_input("worst radius")
worst_texture = st.number_input("worst texture")
worst_perimeter = st.number_input("worst perimeter")
worst_area = st.number_input("worst area")
worst_smoothness = st.number_input("worst smoothness")
worst_compactness = st.number_input("worst compactness")
worst_concavity = st.number_input("worst concavity")
worst_concave_ponts = st.number_input("worst concave points")
worst_symmetry = st.number_input("worst symmetry")
worst_fractal_dimension = st.number_input("worst fractal dimension")

radius_error = st.number_input('radius error')
texture_error = st.number_input("texture error")
perimeter_error = st.number_input("perimeter error")
area_error = st.number_input("area error")
smoothness_error = st.number_input("smoothness error")
compactness_error = st.number_input("compactness error")
concavity_error = st.number_input("concavity error")
concave_points_error = st.number_input("concave point error")
symmetry_error = st.number_input("symmetry error")
fractal_dimension_error = st.number_input("fracture dimension error")

test_data = np.array([mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,\
    mean_concave_ponts,mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,\
        concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,\
        worst_concave_ponts,worst_symmetry,worst_fractal_dimension])
if st.checkbox("predict"):
    if model.predict(np.reshape(test_data,(1,-1)))[0] ==1:
        st.warning("According to the model your report is Positive")
    else:
        st.write("You are safe but be ")
    