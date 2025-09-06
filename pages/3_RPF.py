import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="RBF Visualization", page_icon="❤️", layout="wide")

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
st.title("💖 Radial Basis Function (RBF) Visualization 💖")
st.write("Welcome! This is a **Radial Basis Function (RBF)** simulation. You can visualize how the RBF transforms data points based on two centers. Enjoy!")

# Sidebar Inputs
st.sidebar.header("🔧 RBF Settings")
st.sidebar.write("Enter the centers and data points:")

# Editable input centers
center1_input = st.sidebar.text_input("Enter Center 1 (x,y):", "0.0,0.0")
center2_input = st.sidebar.text_input("Enter Center 2 (x,y):", "2.5,2.5")

# Parse the centers
c1 = np.array([float(val.strip()) for val in center1_input.split(',')])
c2 = np.array([float(val.strip()) for val in center2_input.split(',')])

# Editable data points input
points_input = st.sidebar.text_area("Enter Data Points (x,y,label):", """0.0,0.0,0
0.0,2.0,0
0.5,1.0,1
1.0,0.5,1
1.0,1.5,1
1.5,1.0,1
2.0,0.0,0
2.0,2.0,0
2.0,3.0,0
2.5,2.5,1
3.0,2.0,0
3.0,3.0,0""")

# Parse the data points
points_lines = points_input.strip().splitlines()
points = []
numeric_labels = []  # 0 or 1
str_labels = []      # "Light" or "Dark"
for line in points_lines:
    if not line.strip():
        continue  # Skip empty lines
    parts = line.split(',')
    if len(parts) < 2:
        continue
    x_val = float(parts[0].strip())
    y_val = float(parts[1].strip())
    label_val = int(parts[2].strip()) if len(parts) >= 3 else 0
    label_str = "Dark" if label_val == 1 else "Light"
    points.append([x_val, y_val])
    numeric_labels.append(label_val)
    str_labels.append(label_str)

X = np.array(points)

# Helper functions for RBF calculation
def squared_distance(x, c):
    """Compute squared Euclidean distance between x and c."""      # r2 = sum[(xi-ci)2] , r2 = sum[(x1-c1)2+(x2-c2)2]
    return np.sum((x - c) ** 2)

def rbf_value(r2):
    """Compute RBF using phi = exp(-r^2)."""                       # phi = exp(-r2)
    return np.exp(-r2)

# Table for results
table_data = [["#", "x1", "x2", "r²(c1)", "r²(c2)", "φ1", "φ2", "Label"]]

# Calculate RBF values for each point
for i, point in enumerate(X):
    r2_c1 = squared_distance(point, c1)
    r2_c2 = squared_distance(point, c2)
    phi1 = rbf_value(r2_c1)
    phi2 = rbf_value(r2_c2)

    table_data.append([
        str(i + 1),
        f"{point[0]:.1f}",
        f"{point[1]:.1f}",
        f"{r2_c1:.2f}",
        f"{r2_c2:.2f}",
        f"{phi1:.3e}",
        f"{phi2:.3e}",
        str_labels[i]
    ])

# Grid for 3D & 2D visualization
grid_x = np.linspace(-1, 4, 60)
grid_y = np.linspace(-1, 4, 60)
Xg, Yg = np.meshgrid(grid_x, grid_y)
Zg = np.zeros_like(Xg)

# Compute Z = φ1 + φ2 at each grid point
for row in range(Xg.shape[0]):
    for col in range(Xg.shape[1]):
        r2_1 = squared_distance([Xg[row, col], Yg[row, col]], c1)
        r2_2 = squared_distance([Xg[row, col], Yg[row, col]], c2)
        phi1 = rbf_value(r2_1)
        phi2 = rbf_value(r2_2)
        Zg[row, col] = phi1 + phi2

# Create the figure and subplots
fig = plt.figure(figsize=(14, 10))

# Subplot (1,1): Scatter plot of data points
ax1 = fig.add_subplot(221)
for i, point in enumerate(X):
    if numeric_labels[i] == 1:  # Dark
        ax1.plot(point[0], point[1], 'ko', markersize=8)
    else:  # Light
        ax1.plot(point[0], point[1], 'ro', markerfacecolor='none', markersize=8)
ax1.set_title("Data Points")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_xlim([-1, 4])
ax1.set_ylim([-1, 4])
ax1.grid(True)

# Subplot (1,2): Table of results
ax2 = fig.add_subplot(222)
ax2.axis('off')  # Hide the axis so the table is prominent
table_plot = ax2.table(cellText=table_data, loc='center')
table_plot.auto_set_font_size(False)
table_plot.set_fontsize(9)
table_plot.scale(1, 1.5)

# Subplot (2,1): 3D surface plot (Corrected to 3D)
ax3 = fig.add_subplot(223, projection='3d')
surf = ax3.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.8)
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('RBF Value')
ax3.set_title('3D RBF Surface')
fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=10)

# Subplot (2,2): 2D contour plot
ax4 = fig.add_subplot(224)
contour = ax4.contourf(Xg, Yg, Zg, cmap='viridis', levels=50)
ax4.set_xlabel('x1')
ax4.set_ylabel('x2')
ax4.set_title('2D RBF Contour')
fig.colorbar(contour, ax=ax4, label='RBF Value')

# Mark the centers in both 3D and 2D plots
ax3.scatter([c1[0], c2[0]], [c1[1], c2[1]], [1, 1], color='red', s=100, marker='o')
ax4.scatter([c1[0], c2[0]], [c1[1], c2[1]], color='red', s=100, marker='o', label='Centers')

# Mark the original data points on 2D contour
ax4.scatter(X[:, 0], X[:, 1], color='white', s=50, marker='x', label='Data Points')
ax4.legend()

# Display the plot
plt.tight_layout()
st.pyplot(fig)
# Add a section for finding nearest cluster for a test point
st.write("## Find Nearest Cluster")
st.write("Enter coordinates to find the nearest cluster center:")

# Create a form for input coordinates
with st.form("cluster_form"):
    col1, col2 = st.columns(2)
    with col1:
        x1_test = st.number_input("x1 coordinate", value=0.0, step=0.1)
    with col2:    
        x2_test = st.number_input("x2 coordinate", value=0.0, step=0.1)
    
    submitted = st.form_submit_button("Find Nearest Cluster")
    
    if submitted:
        test_point = np.array([x1_test, x2_test])

        # Calculate distances to both centers
        dist_c1 = np.linalg.norm(test_point - c1)
        dist_c2 = np.linalg.norm(test_point - c2)

        # Determine nearest cluster
        if dist_c1 < dist_c2:
            nearest = "Cluster 1"
            nearest_center = c1
            distance = dist_c1
        else:
            nearest = "Cluster 2" 
            nearest_center = c2
            distance = dist_c2

        # Display results in a nice format
        st.success(f"""
        **Results:**
        - Nearest cluster: {nearest}
        - Distance to center: {distance:.3f}
        - Center coordinates: ({nearest_center[0]:.2f}, {nearest_center[1]:.2f})
        """)

        # Update the plots to show the test point
        ax1.plot(x1_test, x2_test, 'g*', markersize=15, label='Test Point')
        ax4.plot(x1_test, x2_test, 'g*', markersize=15, label='Test Point')
        ax1.legend()
        ax4.legend()
        st.pyplot(fig)
