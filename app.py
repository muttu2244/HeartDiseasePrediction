import gradio
import joblib
import numpy as np
import pandas as pd

model = joblib.load('xgboost-model.pkl')

def handle_outliers(df, colm):
    '''Change the values of outlier to upper and lower whisker values '''
    q1 = df.describe()[colm].loc["25%"]
    q3 = df.describe()[colm].loc["75%"]
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for i in range(len(df)):
        if df.loc[i,colm] > upper_bound:
            df.loc[i,colm]= upper_bound
        if df.loc[i,colm] < lower_bound:
            df.loc[i,colm]= lower_bound
    return df

def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                        high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):

    yes_no_map = {'Yes':1, 'No':0}
    gender_map = {'M':1, 'F':0}

    outlier_colms = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']


    inputs_from_user = [age, yes_no_map[anaemia], creatinine_phosphokinase, yes_no_map[diabetes], ejection_fraction,
                        yes_no_map[high_blood_pressure], platelets, serum_creatinine, serum_sodium, gender_map[sex], yes_no_map[smoking], time]


    cols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                        'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']

    input_df = pd.DataFrame([inputs_from_user], columns=cols)

    for colm in outlier_colms:
        input_df = handle_outliers(input_df, colm)



    #inputs_to_model = np.array(inputs_from_user).reshape(1, -1)

    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df) * 100

    if prediction[0]==1:
        return f"Subject patient will not survive with a probability of {prob[0][1]:.1f} %"
    elif prediction[0]==0:
        return f"Subject patient will survive with a probability of {prob[0][0]:.1f} %"
    else:
        return f"Error observed while making prediction"


# Inputs from user
inputs = [gradio.Slider(1, 100, label="Enter the age of the patient:"),
          gradio.Radio(["Yes", "No"], label="Whether patient is Anaemic or not?:"),
          gradio.Slider(200, 10000, label="Enter the level of CPK enzyme in the patient's blood (mcg/L):"),
          gradio.Radio(["Yes", "No"], label="Whether patient is diabetic or not?:"),
          gradio.Slider(10, 100, label="Enter the % of blood leaving the patient's heart at each contraction:"),
          gradio.Radio(["Yes", "No"], label="Whether patient is Hypertensive or not?:"),
          gradio.Slider(10000, 100000, label="Enter the No. of platelets in the patient's blood (kiloplatelets/mL):"),
          gradio.Slider(0.1, 10, label="Enter the level of serum creatinine in the patient's blood (mg/dL):"),
          gradio.Slider(100, 200, label="Enter the level of serum sodium in the patient's blood (mEq/L): "),
          gradio.Radio(["M", "F"], label="Choose the sex of the patient:"),
          gradio.Radio(["Yes", "No"], label="Whether the patient smokes or not?:"),
          gradio.Slider(1, 100, label="Enter the follow-up period (days):"),
          ]

# Output response
outputs = gradio.Textbox(type="text", label='Will the patient survive?')

# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs = inputs,
                         outputs = outputs,
                         title = title,
                         description = description,
                         allow_flagging='never')

iface.launch(server_name="0.0.0.0", server_port = 7860)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface


