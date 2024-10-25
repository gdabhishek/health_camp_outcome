import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
from sklearn.preprocessing import OneHotEncoder
data_dir = "Dataset/"


model = joblib.load("health_camp_outcome.pkl")
sc = model.sc
oh_map = model.oh_map

def merge_data(new_data):
    new_data.drop(["Var2","Var3","Var4","Var5"], axis = 1,inplace = True)
    camp_det = pd.read_csv(data_dir +"/Health_Camp_Detail.csv")
    patient_profile = pd.read_csv(data_dir +"/Patient_Profile.csv")
    fhc = pd.read_csv(data_dir +"/First_Health_Camp_Attended.csv")
    patient_profile = patient_profile.replace("None",np.nan)
    dataset = pd.merge(new_data, camp_det, how = "left", on = "Health_Camp_ID")
    dataset = pd.merge(dataset, patient_profile, how = "left", on = "Patient_ID")
    fhc.drop(["Health_Score", "Unnamed: 4"], axis = 1,inplace = True)
    dataset = pd.merge(dataset, fhc, how = "left", on = ["Patient_ID","Health_Camp_ID"])
    return dataset

def derive_columns(dataset):
    dataset["Donation"].fillna(0,inplace = True)
    dataset["Registration_Date"] = pd.to_datetime(dataset["Registration_Date"], format='%d-%b-%y')
    dataset["Camp_Start_Date"] = pd.to_datetime(dataset["Camp_Start_Date"],format='%d-%b-%y')
    dataset["Camp_End_Date"] = pd.to_datetime(dataset["Camp_End_Date"],format='%d-%b-%y')
    dataset["First_Interaction"] = pd.to_datetime(dataset["First_Interaction"],format='%d-%b-%y')

    dataset["Camp_Duration"] = dataset["Camp_End_Date"] - dataset["Camp_Start_Date"]
    dataset["Camp_Duration"] = dataset["Camp_Duration"].dt.days
    dataset['Registration_Date'].fillna(dataset['Camp_Start_Date'], inplace = True)
    dataset["dur_rf"] = dataset["Registration_Date"] - dataset["First_Interaction"]
    dataset["dur_rf"] = dataset["dur_rf"].dt.days

    dataset = dataset.drop(["Patient_ID","Health_Camp_ID","City_Type",
                            "Employer_Category", "Income","Education_Score","Age"],axis=1)

    dataset["RD"] = dataset["Registration_Date"].dt.day
    dataset["RM"] = dataset["Registration_Date"].dt.month
    dataset["RY"] = dataset["Registration_Date"].dt.year
    dataset["RW"] = dataset["Registration_Date"].dt.dayofweek

    dataset["CSD"] = dataset["Camp_Start_Date"].dt.day
    dataset["CSM"] = dataset["Camp_Start_Date"].dt.month
    dataset["CSY"] = dataset["Camp_Start_Date"].dt.year
    dataset["CSW"] = dataset["Camp_Start_Date"].dt.dayofweek

    dataset["CED"] = dataset["Camp_End_Date"].dt.day
    dataset["CEM"] = dataset["Camp_End_Date"].dt.month
    dataset["CEY"] = dataset["Camp_End_Date"].dt.year
    dataset["CEW"] = dataset["Camp_End_Date"].dt.dayofweek

    dataset["FID"] = dataset["First_Interaction"].dt.day
    dataset["FIM"] = dataset["First_Interaction"].dt.month
    dataset["FIY"] = dataset["First_Interaction"].dt.year
    dataset["FIW"] = dataset["First_Interaction"].dt.dayofweek

    dataset.drop(["Registration_Date","Camp_Start_Date","Camp_End_Date","First_Interaction"],axis=1,inplace=True)

   
    return dataset
def one_hot_encode(new_data, categorical_features,oh_map, drop_first = True):
    for col in categorical_features:
        oh_encode=pd.DataFrame(oh_map[col].transform(new_data[col].values.reshape(-1,1)).toarray())
        if drop_first:
            oh_encode=oh_encode.iloc[:,1:]
        new_data=new_data.drop(col,axis=1)
        new_data=pd.concat([new_data,oh_encode],axis=1)
    return new_data

def pre_process_data(dataset,oh_map, sc):
    
    dataset = one_hot_encode(dataset, ["Category1","Category2","Category3"],oh_map)
    dataset = sc.transform(dataset)
    return dataset

def main():
    st.title("Health Camp Outcome Analysis")

    # File upload
    st.header("Upload the Health camp registration Data")
    file = st.file_uploader("Upload Health camp registration Data", type=["csv"])

    if file is not None:
        print("file", file  )
        # Read CSV file
        test_data = pd.read_csv(file)
       
        final_data = merge_data(test_data)
        final_data = derive_columns(final_data)
        final_data = pre_process_data(final_data, oh_map, sc)
        prediction = model.predict(final_data)

        test_data["outcome"] = prediction
        # Download button
        st.write("### Download Health Camp Outcome Analysis")
        st.download_button(
            label="Download CSV file",
            data=test_data.to_csv(index=False),
            file_name="processed_data.csv",
            mime="text/csv",
        )
        pie_chart_col, bar_chart_col = st.columns(2)
        with pie_chart_col:
            st.write("### Outcome Analysis")
            test_data["outcome"] = test_data["outcome"].replace({0.0:"Might not Attended", 1.0:"High chances of Attending"})
            fig = px.pie(test_data["outcome"], values=test_data["outcome"].value_counts().values, names=test_data["outcome"].value_counts().index)
            st.plotly_chart(fig)



if __name__ == "__main__":
    main()
