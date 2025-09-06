import streamlit as st

st.set_page_config(page_title="ANN Algorithms 🫧", page_icon="🫧", layout="wide")

# Header
st.markdown("""
<div style="background-color: #C7CCD5; width: 100%; padding: 10px; border-radius: 10px;">
   <h1 style="color:#FFFFFF; text-align: center;">ANN Girls 🫧</h1>
</div>
""", unsafe_allow_html=True)

# Welcome text
st.markdown("""
<div style="text-align: center; font-family: Arial, sans-serif; margin: 40px 0;">
   <h1 style="font-size: 40px; color: #EDD3D5;">Welcome to ANN Girls 🫧 </h1>
   <p style="font-size: 18px; color: #4B4B4B;">
       Explore the ANN Girls 🫧 To navigate the world of algorithms 📈.
   </p>
</div>
""", unsafe_allow_html=True)

# Add some spacing
st.write("")

# First row of buttons
col1, space1, col2 = st.columns([1, 0.2, 1])

with col1:
    if st.button("Go to PCA Algorithm 🩵", key="pca_button"):
        st.switch_page("Pages/st_PCA.py")

with col2:
    if st.button("Go to SOM Algorithm ♥️", key="som_button"):
        st.switch_page("Pages/SOM_IS.py")

# Add spacing between rows
st.write("")
st.write("")

# Second row of buttons
col3, space2, col4 = st.columns([1, 0.2, 1])

with col3:
    if st.button("RBF Algorithm 💖", key="backprop_button"):
        st.switch_page("Pages/RPF.py")

with col4:
    if st.button("Go to ART1 Algorithm 🩶", key="cnn_button"):
        st.switch_page("Padges/ART22.py")

# Add spacing between rows
st.write("")
st.write("")

# Third row (centered)
col5, col5_space2, col6 = st.columns([1, 0.2, 1])

with col5:
    if st.button("Go to Genetic Algorithm 🌸", key="rnn_button"):
        st.switch_page("Pages/genetic_algorithm_lab.py")

with col6:
    if st.button("Go to Fuzzy Algorithm 👩‍🎓✨", key="fuzzy_button"):
        st.switch_page("Pages/Fuzzy_python.py")

# Custom styling for better appearance with larger buttons and more spacing
st.markdown("""
<style>
    /* Button styling */
    div[data-testid="stButton"] button {
        background-color: #EDD3D5;
        color: #4B4B4B;
        padding: 20px 30px;
        font-size: 22px;
        border-radius: 12px;
        font-family: Arial, sans-serif;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        width: 100%;
        transition: all 0.3s ease;
        margin-top: 20px;
    }
    
    /* Hover effect */
    div[data-testid="stButton"] button:hover {
        background-color: #D8BDBF;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    /* Active effect */
    div[data-testid="stButton"] button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)




