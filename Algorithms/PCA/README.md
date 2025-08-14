
🌃 PCA Clustering Simulation 💙

An interactive Streamlit app that demonstrates Principal Component Analysis (PCA) from scratch.
The app allows you to input your own dataset, performs mean centering, covariance matrix calculation,
eigenvalue decomposition, and projection onto principal components, then visualizes the results.

📌 Features
- Custom Data Input (via sidebar)
- Manual Covariance Calculation (step-by-step)
- Eigenvalues & Eigenvectors Computation
- PCA Transformation
- Visual Comparison between original and transformed data
- Principal Components Directions Plot

🖼 Demo
The app produces:
1. Original Data Scatter Plot
2. PCA Transformed Data Plot
3. Principal Axes over Original Data

🛠 Installation & Usage

1️⃣ Clone the Repository
    git clone https://github.com/yourusername/pca-streamlit-app.git
    cd pca-streamlit-app

2️⃣ Install Requirements
    Make sure you have Python 3.8+ installed, then run:
    pip install -r requirements.txt

3️⃣ Run the App
    streamlit run 0880be20-7816-43fc-a40c-e87229a27474.py

📂 Project Structure
    📁 pca-streamlit-app
     ├── 0880be20-7816-43fc-a40c-e87229a27474.py   # Main Streamlit app
     ├── requirements.txt                         # Python dependencies
     └── README.txt                                # Project documentation

📋 How It Works
1. Data Input – Enter your dataset in the sidebar as comma-separated values.
2. Mean Centering – Data is centered by subtracting column means.
3. Covariance Matrix – Calculated manually using the classical formula.
4. Eigen Decomposition – Extracts eigenvalues & eigenvectors.
5. Transformation – Projects data onto the new principal component axes.
6. Visualization – Displays both original and PCA-transformed datasets.

📦 Requirements
- streamlit
- numpy
- matplotlib

Install them manually:
    pip install streamlit numpy matplotlib

📜 License
This project is open-source and available under the MIT License.

