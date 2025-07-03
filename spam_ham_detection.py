import streamlit as st
import os
import joblib
import nltk

# === Download required NLTK packages ===
nltk_packages = ['punkt', 'stopwords']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# === Custom UI Styling ===
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

# === UI Color Setup ===
customize_ui("#000000", "#FFFFFF", "#1E90FF", "#FFFFFF")

# === Load Model and Vectorizer ===
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

model, vectorizer = load_model_and_vectorizer()

# === Streamlit App Layout ===
def main():
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or ham.")
    st.subheader("Classification")
    
    # User Input
    user_input = st.text_area("Enter an email to classify", height=150)
    
    # Classification Button
    if st.button("Classify"):
        if not user_input:
            st.warning("Please enter an email to classify.")
        elif model is None or vectorizer is None:
            st.error("Model or Vectorizer not loaded correctly.")
        else:
            # Transform and Predict
            transformed_email = [user_input]
            vec_input = vectorizer.transform(transformed_email)
            result = model.predict(vec_input)
            if result[0] == 0:
                st.success("This is not a Spam Email (Ham Email)")
            else:
                st.error("This is a Spam Email")

# === Run App ===
if __name__ == "__main__":
    main()
