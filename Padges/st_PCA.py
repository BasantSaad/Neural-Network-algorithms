import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="PCA Clustering", page_icon="💙", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #89CFF0;
        background-size: cover;
    }
    .stApp {
        color: #000080;
        font-family: Arial, sans-serif;
    }
    .stTitle {
        font-size: 30px;
        font-weight: bold;
        color: #000080; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Welcome message
st.title("🌃 Principal Component Analysis (PCA) 💙")
st.write("Welcome! ,**Principal Component Analysis (PCA)** simulation. This will include all the necessary steps: data standardization, covariance matrix calculation, eigenvalue decomposition, and projection onto principal components.. Enjoy ❄️")

# Sidebar Inputs
st.sidebar.header("🌌 PCA Inputs")
data_input = st.sidebar.text_area("Enter data as comma-separated values (e.g., 1.4,1.65\n1.6,1.975)\n\n\n",
                          "1.4,1.65\n1.6,1.975\n-1.4,-1.775\n-2,-2.525\n-3,-3.95\n2.4,3.075\n1.5,2.025\n2.3,2.75\n-3.2,-4.05\n-4.1,-4.85")

# Process input data
data = np.array([list(map(float, row.split(','))) for row in data_input.strip().split('\n')])

# Display user input data
st.subheader("Input Data 📥")
st.write(data)

# Calculate mean
mean_values = np.mean(data, axis=0)

# Center the data (subtract mean)
X_centered = data - mean_values

# Calculate **Classical Covariance Formula** # Cov= Sum[ (X1-X1_mean)(X2-X2_mean)] /(n-1) if there are just two independent variables 
n = len(data)
var_x1 = np.sum(X_centered[:, 0]**2) / (n - 1)
var_x2 = np.sum(X_centered[:, 1]**2) / (n - 1)
cov_x1x2 = np.sum(X_centered[:, 0] * X_centered[:, 1]) / (n - 1)  

# Manual covariance matrix         **or**         # Cov_matrix= XCT. XC/(n-1) 
cov_matrix = np.array([
    [var_x1, cov_x1x2],
    [cov_x1x2, var_x2]
])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# Sort in descending order 
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

#note :
#       the LAPACK routine used internally by NumPy may return different sign conventions depending on floating-point precision.

eigenvectors[:, 0] = -eigenvectors[:, 0]
eigenvectors[:, 1] = -eigenvectors[:, 1]

# Recalculate transformed data                               #T_data= XC. eigenvectors
transformed_data = np.dot(X_centered, eigenvectors)

st.subheader("Mean values ⚖️")
st.write(mean_values)

st.subheader("Variance and Covariance 🎢 ")
st.write(f"- Variance of X1  : {round(var_x1,4)}")
st.write(f"- Variance of X2  : {round(var_x2,4)}")
st.write(f"- Covariance of X1 and X2  : {round(cov_x1x2,4)}")

st.subheader("Covariance Matrix  🔄")
st.write(cov_matrix)

st.subheader("Eigenvalues 📈")
st.write(eigenvalues)

st.subheader("Eigenvectors 🧭 ")
st.write(eigenvectors)

st.subheader("Transformed Data 🔀. ")
st.write(transformed_data)
# Plot original and transformed data
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(data[:, 0], data[:, 1], alpha=0.8)
ax[0].set_title('Original Data ')
ax[0].set_xlabel('X1')
ax[0].set_ylabel('X2')
ax[0].grid(True)
# Plot transformed data that is the principal components 
ax[1].scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.8)
ax[1].set_title('PCA Transformed Data')
ax[1].set_xlabel('First Principal Component')
ax[1].set_ylabel('Second Principal Component')
ax[1].grid(True)

st.pyplot(fig)

# Show principal axes on original data
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(data[:, 0], data[:, 1], alpha=0.8)

comp1 = eigenvectors[:, 0] * np.sqrt(eigenvalues[0]) * 0.5
comp2 = eigenvectors[:, 1] * np.sqrt(eigenvalues[1]) * 0.5

ax.arrow(mean_values[0], mean_values[1], comp1[0], comp1[1], color='r', width=0.05, 
          head_width=0.3, head_length=0.2, label='First PC')
ax.arrow(mean_values[0], mean_values[1], comp2[0], comp2[1], color='g', width=0.05, 
          head_width=0.3, head_length=0.2, label='Second PC')

ax.set_title('Principal Components Directions')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.grid(True)
ax.legend()
ax.axis('equal')

st.pyplot(fig)
