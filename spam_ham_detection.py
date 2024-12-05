import streamlit as st
import pickle

# Custom CSS for background, text, and button
def customize_ui(background_color, text_color, button_color, button_text_color):
    st.markdown(
        f"""
        <style>
        /* Background color */
        .stApp {{
            background-color: {background_color};
        }}

        /* Text color */
        div, h1, h2, h3, h4, h5, h6, p {{
            color: {text_color};
        }}

        /* Button styling */
        div.stButton > button {{
            background-color: {button_color};
            color: {button_text_color};
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }}

        /* Button hover effect */
        div.stButton > button:hover {{
            background-color: #555555;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Customize colors
background_color = "#000000"  # Black background
text_color = "#FFFFFF"       # White text
button_color = "#1E90FF"     # Blue button
button_text_color = "#FFFFFF" # White button text

customize_ui(background_color, text_color, button_color, button_text_color)

# load the model and vector files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def main():
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or ham.")
    st.subheader("Classification")
    # input text
    user_input=st.text_area("Enter an email to classify" ,height=150)
	
    if st.button('Classify'):
        if user_input:
            # preprocess
            transformed_email =[user_input]
            # vectorize
            vec_input = tfidf.transform(transformed_email)
            # predict
            result = model.predict(vec_input)
            # display
            if result[0]==0:
                st.success("This is not Spam Email (Ham Email)")
            else:
                st.error("This is A Spam Email")
        else:
            st.write("Please Enter an Email to classify.")

main()
