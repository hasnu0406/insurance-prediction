# Health Care Insurance Premium Prediction

![insurance_premium_poster](/images/insurance_premium_poster.jpg)

## Context
Now a days,  healthcare firms utilize personal health data of insurance holders to predict individual health insurance premiums. The premium amount varies from person to person, as numerous factors influence the cost of a health insurance policy. For instance, age plays a significant role, where younger individuals are less likely to have major health problems compared to older ones, leading to higher treatment expenses for older individuals. Consequently, older individuals are required to pay higher premiums than younger ones. Apart from age, several other factors also impact health insurance premiums.

![insurance_premium_context](/images/insurance_premium_context.jpg)

## Content
We have developed a comprehensive system that utilizes various techniques to estimate the required amount of insurance premium based on individual health situations. The dataset consists of 1338 records with 7 attributes, and the data is stored in a structured format within a CSV file.

#### Data Source
KAGGLE - https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction

## Abstract
This project aims to predict an individual's medical bill expenses based on their health conditions, providing them with the ability to focus on their well-being without the burden of financial worries. We evaluated the performance of five regression models, namely Linear Regression, Decision Tree Regression, Random Forest Regression, Gradient Boosting Regression, and KNN, to identify the most effective one. The models were trained using a comprehensive dataset, enabling accurate predictions. Comparison with actual data verified the model's accuracy, and the best-performing model was selected for deployment. This approach ensures a reliable and efficient solution for predicting medical bill expenses.

## __There is rise in demand for Health Care Insurance Premium in the recent years !!__

![rise in demand for insurance premium](/images/rising_demand.jpg)

- According to a study by the Peterson Center on Healthcare and the Kaiser Family Foundation (KFF), __U.S. healthcare spending rose nearly a trillion dollars from 2009 to 2019,__ when adjusted for inflation.
- A recent study reported that __U.S. healthcare spending during 2019 was nearly $3.8 trillion, or $11,582 per person__. By 2028, these costs are expected to climb to $6.2 trillion—roughly $18,000 per person.
- According to the American Medical Association (AMA), __healthcare costs are rising by about 4.5% a year.__

![Investopedia page](/images/rising_cost_insurance.jpg)

## Objective
The primary objective of this project is to create an application that predicts medical expenses for individuals based on their health information. This prediction is achieved using machine learning-based regression models on input data. The proposed solution aims to estimate insurance premiums based on people's health data, providing valuable insights for insurance companies and individuals.

## Attribute Information
- age - Insurance holder's age in years
- sex - Gender of the insurance holder (Male or Female)
- bmi - BMI stands for Body Mass Index, the ideal range according to height and weight is 18.5 to 24.9
- children - Number of children
- smoker - Whether the insurance holder is a smoker or not
- region - Residential area of the person
- expenses - Individual medical costs billed by health insurance

## Tasks Performed
### 1. Data exploration: 
Explored the dataset using Python libraries such as pandas, numpy, matplotlib, and seaborn to gain insights into the data.

### 2. Exploratory Data Analysis : 
Conducted EDA by plotting various graphs to better understand the relationships between dependent and independent features.

### 3. Feature Engineering : 
Performed feature engineering, including scaling numerical data and encoding categorical data to prepare it for model building.

### 4. Model Building : 
Divided the data into training and testing sets and used multiple algorithms for model training, including:<br>
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- KNN

### 5. Model Selection : 
Evaluated all the models using performance metrics such as Root Mean Squared Error (RMSE) and R-squared, and selected the best-performing model for deployment.

### 6. Web Application & Deployment : 
Developed a web application using the Streamlit API to take user inputs and display the predicted output.

## Conclusion
The system demonstrates the utilization of various techniques to estimate the premium amount based on an individual's health condition. It evaluates the impact of factors such as smoking habits and gender on the premium amount. By employing different regression models for prediction, Gradient Boosting emerged as the most accurate model for this task. Our predictions provide users with insights into the expected medical expenses required for purchasing insurance premium and also assist healthcare organizations in determining insurance charges based on an individual's current health condition. This solution aims to offer transparency and informed decision-making for both individuals and healthcare providers in the insurance domain.

# Helper
This project consists of three main components(jupyter notebooks):

### 🟢 Exploratory Data Analysis (EDA):
The notebook __"abc_EDA.ipynb"__ contains the code for exploring the dataset using libraries like pandas, numpy, matplotlib, and seaborn. Various graphs and visualizations have been plotted to gain insights into the dependent and independent features.
### 🟢 Model Building:
In the notebook __"abc_model.ipynb"__ different regression models including Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and KNN, have been trained on the data. The notebook demonstrates the process of data preprocessing, feature engineering, model training, and evaluation using metrics such as RMSE and R-squared. The best-performing model is selected for further use.
### 🟢 Model Deployment:
The project is deployed as a web application using the streamlit API. You can access the application by visiting the link: __https://insurance-hasna.streamlit.app/__ The web application allows users to input relevant information and obtain predictions on the required medical expenses and insurance premium based on their health data.

This project offers a complete data science workflow, from initial data exploration to model training and ultimately presenting predictions through an intuitive web application.

# How to start and use streamlit?

Simply run the following commands on command line and let it do the magic
```
pip3 install -r requirements.txt
streamlit run app.py
```

__This is how our Streamlit Application looks:__

![streamlitapp page](/images/app_1.jpg)
"# insurance-prediction" 
"# insurance-prediction" 
"# insurance-prediction" 
