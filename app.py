import sys
import streamlit as st
import pickle
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('.\\pipe.pkl', 'rb'))
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame(
        {'batting_team': [str(batting_team)], 'bowling_team': [str(bowling_team)], 'city': [str(selected_city)],
         'runs_left': [int(runs_left)], 'balls_left': [int(balls_left)], 'wickets': [int(wickets)],
         'total_runs_x': [int(target)], 'crr': [float(crr)], 'rrr': [float(rrr)]})

    loaded_pipe = joblib.load(".\\iplpred.joblib")

    result = loaded_pipe.predict_proba(input_df.iloc[[0]])

    print(f"------------------------> {result}")
    print(f"------------------------> {type(result)}")

    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win * 100)) + "%")
    st.header(bowling_team + "- " + str(round(loss * 100)) + "%")


def main():
    # Parse command-line arguments
    args = sys.argv[1:]  # Exclude the script name itself

    # Check if arguments were provided
    if len(args) == 0:
        st.error("No arguments provided. Please provide the required arguments.")
        return

    # Extract the arguments
    pipe_file = args[0]  # Assuming the first argument is the pipe file

    # Your Streamlit application logic here
    # Use the pipe_file variable to load your pickled file

    # Example usage
    st.write(f"Loading pipe file: {pipe_file}")


if __name__ == "__main__":
    main()


