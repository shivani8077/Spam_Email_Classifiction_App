import streamlit as st
import pickle
import os

# Custom CSS for background, text, and button
def customize_ui(background_color, text_color, button_color, button_text_color):
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {background_color}; }}
        div, h1, h2, h3, h4, h5, h6, p {{ color: {text_color}; }}
        div.stButton > button {{
            background-color: {button_color};
            color: {button_text_color};
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }}
        div.stButton > button:hover {{ background-color: #555555; }}
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

# Load model and vectorizer
# Change base_dir to current working directory for compatibility with Streamlit Cloud
base_dir = os.getcwd()  # Use current working directory
vectorizer_path = os.path.join(base_dir, 'vectorizer.pkl')
model_path = os.path.join(base_dir, 'model.pkl')

def load_pickle(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading pickle file: {e}")
            return None
    else:
        st.error(f"Error: {file_path} not found. Please check the file path.")
        return None

# Load the vectorizer and model
tfidf = load_pickle(vectorizer_path)
model = load_pickle(model_path)

def main():
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or ham.")
    st.subheader("Classification")
    
    # User input for email to classify
    user_input = st.text_area("Enter an email to classify", height=150)
    
    if st.button('Classify'):
        if user_input:
            if tfidf and model:
                # Preprocess input and classify
                transformed_email = [user_input]
                vec_input = tfidf.transform(transformed_email)
                result = model.predict(vec_input)
                
                # Display result
                if result[0] == 0:
                    st.success("This is not a Spam Email (Ham Email)")
                else:
                    st.error("This is a Spam Email")
            else:
                st.error("Model or Vectorizer not loaded correctly.")
        else:
            st.warning("Please enter an email to classify.")

# Run the app
if __name__ == "__main__":
    main()
