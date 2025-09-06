import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="SOM Clustering", page_icon="❤️", layout="wide")

# Custom CSS for pink theme and hearts
st.markdown(
    """
    <style>
    body {
        background-color: #ffccdd; /* Soft pink */
        background-image: url('https://www.transparenttextures.com/patterns/hearts.png'); /* Red hearts */
        background-size: cover;
    }
    .stApp {
        color: #d63384;
        font-family: Arial, sans-serif;
    }
    .stTitle {
        font-size: 30px;
        font-weight: bold;
        color: #ff1493; /* Deep pink */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Welcome message
st.title("💖 Self-Organizing Map (SOM) 💖")
st.write("Welcome! This is a **Self-Organizing Map (SOM)** simulation. You can edit input vectors, initial weights, and control the number of training iterations. Enjoy!")

# Sidebar Inputs
st.sidebar.header("🔧 SOM Settings")
st.sidebar.write("Edit input vectors and initial weights:")

# Editable input vectors
vectors_input = st.sidebar.text_area("Enter Vectors:", """1 1 0 0
0 0 0 1
1 0 0 0
0 0 1 1""")
vectors = np.array([list(map(float, line.split())) for line in vectors_input.splitlines()])

# Editable initial weights
initial_weights_input = st.sidebar.text_area("Enter Initial Weights:", """0.2 0.8
0.6 0.4
0.5 0.7
0.9 0.3""")
initial_weights = np.array([list(map(float, line.split())) for line in initial_weights_input.splitlines()])

# Slider for number of iterations
iterations = st.sidebar.slider("Select Number of Iterations", min_value=1, max_value=10, value=4, step=1)

# Train SOM function with correct calculations                                         #vectors.shape=(4,4) and weights.shape=(2,2)
def train_som(vectors, weights, iterations):
    history = [weights.copy()]
    learning_rate = 0.6

    for iteration in range(1, iterations + 1):
        for vec in vectors:
            distances = np.linalg.norm(weights - vec.reshape(-1, 1), axis=0)          # Compute distances = sum[Wij-Xi]2
            winner_idx = np.argmin(distances)                                         # Find closest cluster = min(distances)

            # Correct weight update formula
            weights[:, winner_idx] += learning_rate * (vec - weights[:, winner_idx])  # Update weights = Wij+Lr(Xi-Wij) for the winning cluster

        history.append(weights.copy())

        # Decay learning rate after 4 iterations that means *one epoch*
        if iteration % 4 == 0:
            learning_rate *= 0.5  

    return weights, history

# Train SOM dynamically based on selected iterations
final_weights, history = train_som(vectors, initial_weights.copy(), iterations=iterations)

# Display initial and final weights
st.write("### 🔢 Initial Weight Matrix ")
st.dataframe(pd.DataFrame(initial_weights, columns=["Unit 1", "Unit 2"]))

st.write("### 🎯 Final Weight Matrix")
st.dataframe(pd.DataFrame(final_weights, columns=["Unit 1", "Unit 2"]).style.format(precision=4))

# Visualization of weight evolution
def plot_weights(history):
    num_iterations = len(history)
    fig, axes = plt.subplots(1, num_iterations, figsize=(15, 3))
    
    for i, weights in enumerate(history):
        sns.heatmap(weights, annot=True, cmap='coolwarm', fmt='.3f', ax=axes[i])
        axes[i].set_title(f'Iteration {i}')
    
    st.pyplot(fig)

st.write("### 🔄 Weight Evolution Over Iterations")
plot_weights(history)

# Competitive Learning Visualization
def plot_competitive_learning(history, vectors):
    num_iterations = len(history)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    iterations_to_plot = [0, max(1, iterations // 3), max(2, iterations // 2), iterations - 1]
    
    for idx, iteration in enumerate(iterations_to_plot):
        ax = axes[idx]
        weights = history[iteration]

        ax.scatter(vectors[:, 0], vectors[:, 1], color='green', label="Input Vectors")
        ax.scatter(weights[0, :], weights[1, :], color='red', marker='*', s=150, label="Weights (Clusters)")

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(f"Iteration {iteration}")
        ax.legend()

    st.pyplot(fig)

st.write("### 📌 Competitive Learning Visualization")
plot_competitive_learning(history, vectors)

# Advanced visualization: Line plot for weight convergence
def plot_weight_convergence(history):
    num_iterations = len(history)
    weights_over_time = np.array(history)  # Convert to numpy array
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for i in range(initial_weights.shape[0]):  # Iterate over each feature
        for j in range(initial_weights.shape[1]):  # Iterate over each cluster
            ax.plot(range(num_iterations), weights_over_time[:, i, j], marker='o', label=f'Feature {i+1} - Cluster {j+1}')
    
    ax.set_title("📈 Weight Convergence Over Iterations")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Weight Value")
    ax.legend()
    
    st.pyplot(fig)

st.write("### 📊 Weight Convergence Over Training")
plot_weight_convergence(history)
