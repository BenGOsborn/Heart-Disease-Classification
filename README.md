# Heart disease EDA, classification and understanding

<br />

## This project features exploratory data analysis on the [heart disease dataset](https://www.kaggle.com/ronitf/heart-disease-uci), as well as a model that can predict if a patient has heart disease with an 84% accuracy on the validation set, and breaks down the importance of the features the model uses to make its predictions to help us better understand the factors that lead to heart disease.

<br />

## Dataset details:

https://www.kaggle.com/ronitf/heart-disease-uci
<br />
<br />
"This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. The names and social security numbers of the patients were recently removed from the database, replaced with dummy values. One file has been "processed", that one containing the Cleveland database."
<br />
<br />
Creators:
 1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
 2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
 3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
 4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
<br />
<br />
Donors:
 - David W. Aha (aha '@' ics.uci.edu) (714) 856-8779

#### Raw feature details (https://www.kaggle.com/ronitf/heart-disease-uci/discussion/105877):
 1. age (the age of the patient in years)
 2. sex (0 female, 1 male)
 3. chest pain type (0 typical angina, 1 atypical angina, 2 non-anginal pain, 3 asymptomatic)
 4. resting blood pressure (normal blood pressure of the patient while they are not moving)
 5. serum cholestoral in mg/dl (the serum cholestoral level of the patient)
 6. fasting blood sugar > 120 mg/dl (if the patients blood sugar when fasting is greater than 120)
 7. resting electrocardiographic results (0 showing probable or definite left ventricular hypertrophy by Estes' criteria, 1 normal, 2 having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV))
 8. Maximum heart rate achieved
 9. exercise induced angina (whether the patient experienced excercise induced angina)
 10. oldpeak = ST depression induced by exercise relative to rest
 11. the slope of the peak exercise ST segment (0 downsloping, 1 flat, 2 upsloping)
 12. number of major vessels (0-3) colored by flouroscopy (4 is NaN)
 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect (0 NaN, 1 fixed, 2 normal, 3 reversable)

<br />

### Requirements:
 - Python 3.7.4
 - pandas==0.25.1
 - numpy==1.16.5
 - matplotlib==3.1.0
 - seaborn==0.11.0
 - scikit-learn==0.21.3
 - eli5==0.11.0
 - shap==0.38.1

<br />

### Instructions:
 1. Clone the repository
 2. Install the requirements
 3. Download the [Dataset](https://www.kaggle.com/ronitf/heart-disease-uci), rename it 'heart.csv' and place it in the directory that contains 'heart-disease-notebook.ipynb'
 4. Open the 'heart-disease-notebook.ipynb' notebook
 5. Run the notebook

<br />

## Summary:
#### Age and heart disease:
It was observed that the older the patient, the more at risk they were of having heart disease.

#### Gender and heart disease:
It was observed that males are at a much higher risk of heart disease compared to females.

#### Age, blood pressure, heart rate, and heart disease:
It was found the older the patient, the less the patients heart rate and the greater the patients blood pressure. It was also found that patients that were diagnosed with heart disease had a lower heat rate and higher blood pressure than patients who did not, for all age groups.

#### Feature encoding/engineering:
 - Creating a new feature 'elderly' 1 if the patient is above 60, otherwise 0
 - One hot encoding chest pain type
 - One hot encoding flouroscopy coloured vessels
 - One hot encoding excercise st segment slope
 - One hot encoding the defect type
 - One hot encoding the electrocardiographic results
 - Standardizing the serum cholestoral
 - Standardizing the resting blood pressure
 - Standardzing the max heart rate
 - Standardizing the st depression excercising after resting

##### Final features:
sex, resting_blood_pressure, serum_cholestoral, fasting_blood_sugar>120, max_heart_rate,
excercise_induced_angina, st_depression_excercise_after_rest, elderly, typical_angina, atypical_angina, non_anginal_pain, asymptomatic,
0_flouroscopy_coloured_vessels, 1_flouroscopy_coloured_vessels, 2_flouroscopy_coloured_vessels, 3_flouroscopy_coloured_vessels,
negative_st_slope, flat_st_slope, positive_st_slope, fixed_defect, no_defect, reversable_defect, left_ventricle_hypertrophy,
normal_st_waves, st_wave_abnormality

#### Model building and training:
The model used is a random forrest classifier with 100 estimators. The model was cross validated using K-Fold cross validation over 6 folds. The average accuracy for the models was roughly 80%, and the accuracy for the best model on the validation set is roughly 84%. The model was trained on 246 training examples and was validation on 50 validation examples.

<br />

The model has a 79% accuracy when classifying if a patient does not have heart disease, and an 88% accuracy when classifying if a patient does have heart disease, leading to more false positives than vice versa, which is a positive when trying to predict such a disease.

#### Interpreting the model:
When doing a permutation importance test, it was found that the most impactful features to the model included hether the patient experienced typical angina, followed by the patients maximum heart rate, and if they had no vessels show up in a flouroscopy. Other features deemed important by the model include the patients resting blood pressure, whether they have a reversable defect and their ST depression levels while they are excercising after they have rested. It was found that the features that had a negative impact on the model include if the patient had non anginal pain, and if the patient had left ventricle hypertrophy. 

<br />

Partial dependence plots showed that:
 - The sex of the patient, observing that being male increases the likelihood of a patient has heart disease according to the model
 - The resting blood pressure of the patient, observing that high pressure increases the likelihood of a patient having heart disease according to the model
 - The serum cholestoral levels of the patient, observing that higher serum cholestoral levels increase the likelihood of a patient having heart disease according to the model
 - The maximum heart rate of the patient, observing that as a general trend the lower the heart rate the higher the chance of a patient having heart disease according to the model
 - A patients ST depression excercising after they have rested, observing that the higher the ST depression excercising after they have rested increases the chance of a patient having heart disease according to the model
 - Whether a patient has typical angina, observing that if a patient has typical angina they are more likely to have heart disease according to the model
 - The number of vessels that show in a flouroscopy, observing that the more coloured vessels the higher the patients chances of having heart disease according to the model

<br />

A summary plot using SHAP values concluded that the contributions of the features to the models predictions include (Positive impacts mean increases the likelihood of the model heart disease, and negative impacts decrease the likelihood of the model predicting heart disease):

 - Large positive impact if a patient has typical angina, and a large negative impact if a patient does not
 - Negative impact if a patient has no coloured vessels in a flouroscopy, and a large positive impact if a patient has more than 0 coloured vessels show up in a flouroscopy
 - Large negative impact if a patient had a medium to high max heart rate, and a large positive impact if a patient had a low max heart rate
 - Positive impact if the ST slope is flat, and a negative impact if it is not a flat ST slope
 - Large positive impact if a patient has a high ST depression excercising after resting, and a high negative impact if they have a low ST depression excercising after resting
 - Large negative impact if a patient has no heart defect, and a large positive impact if a patient has a defect
 - Negative impact if a patient has a positive ST slope, and a positive slope if the patient does not have a positive ST slope
 - Positive impact if the patient has a reversable defect, and a negative impact if the npatient does not have a reversable defect
 - Positive impact if the patient has experienced excercise induced angina, and a negative impact if a patient has not experienced excercise induced angina
 - Positive impact if a patient is male, and a negative impact if a patient is female

<br />

A summary plot of how the model interprets the features from the validation data

<br />

![Summary plot](https://imgur.com/a/3qiXHtp)
