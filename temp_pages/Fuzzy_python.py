import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Fuzzy Grade Predictor", page_icon="👩‍🎓", layout="wide")
st.title("🎓 Fuzzy Logic Grade Prediction")

st.markdown("Predict student grades using fuzzy logic based on:")
st.markdown("- Class participation (%)")
st.markdown("- Weekly study hours")

# Define fuzzy variables
participation = ctrl.Antecedent(np.arange(0, 101, 1), 'participation')
study_hours = ctrl.Antecedent(np.arange(0, 21, 1), 'study_hours')
grade = ctrl.Consequent(np.arange(0, 101, 1), 'grade')

# Membership functions
participation['low'] = fuzz.trimf(participation.universe, [0, 0, 50])
participation['medium'] = fuzz.trimf(participation.universe, [30, 50, 70])
participation['high'] = fuzz.trimf(participation.universe, [60, 100, 100])

study_hours['low'] = fuzz.trimf(study_hours.universe, [0, 0, 8])
study_hours['medium'] = fuzz.trimf(study_hours.universe, [5, 10, 15])
study_hours['high'] = fuzz.trimf(study_hours.universe, [12, 20, 20])

grade['poor'] = fuzz.trimf(grade.universe, [0, 0, 50])
grade['average'] = fuzz.trimf(grade.universe, [40, 60, 80])
grade['excellent'] = fuzz.trimf(grade.universe, [75, 100, 100])

# Rules that cover all cases
rule1 = ctrl.Rule(participation['low'] & study_hours['low'], grade['poor'])
rule2 = ctrl.Rule(participation['low'] & study_hours['medium'], grade['poor'])
rule3 = ctrl.Rule(participation['low'] & study_hours['high'], grade['average'])

rule4 = ctrl.Rule(participation['medium'] & study_hours['low'], grade['poor'])
rule5 = ctrl.Rule(participation['medium'] & study_hours['medium'], grade['average'])
rule6 = ctrl.Rule(participation['medium'] & study_hours['high'], grade['excellent'])

rule7 = ctrl.Rule(participation['high'] & study_hours['low'], grade['average'])
rule8 = ctrl.Rule(participation['high'] & study_hours['medium'], grade['excellent'])
rule9 = ctrl.Rule(participation['high'] & study_hours['high'], grade['excellent'])

# Fuzzy control system
grading_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

# Sidebar inputs
st.sidebar.header("📥 Inputs")
input_participation = st.sidebar.slider("Class Participation (%)", 0, 100, 75)
input_study_hours = st.sidebar.slider("Study Hours per Week", 0, 20, 10)

# Run simulation safely
try:
    sim = ctrl.ControlSystemSimulation(grading_ctrl)
    sim.input['participation'] = input_participation
    sim.input['study_hours'] = input_study_hours
    sim.compute()
    predicted = sim.output['grade']
    st.metric("🎯 Predicted Grade", f"{predicted:.2f}%")
except Exception as e:
    st.error(f"Error during fuzzy simulation: {e}")

# Visualization grid
p_vals = np.linspace(0, 100, 50)
s_vals = np.linspace(0, 20, 50)
P, S = np.meshgrid(p_vals, s_vals)
Z = np.zeros_like(P)

# Compute predictions across the grid
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        try:
            sim_grid = ctrl.ControlSystemSimulation(grading_ctrl)
            sim_grid.input['participation'] = P[i, j]
            sim_grid.input['study_hours'] = S[i, j]
            sim_grid.compute()
            Z[i, j] = sim_grid.output['grade']
        except:
            Z[i, j] = np.nan  # gracefully handle unreachable fuzzy combinations

# Plotting
fig = plt.figure(figsize=(14, 6))

# 3D plot
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(P, S, Z, cmap='viridis')
ax1.set_xlabel("Participation (%)")
ax1.set_ylabel("Study Hours")
ax1.set_zlabel("Grade")
ax1.set_title("3D Surface: Fuzzy Grade Output")
fig.colorbar(surf, ax=ax1, shrink=0.6)

# 2D contour
ax2 = fig.add_subplot(122)
contour = ax2.contourf(P, S, Z, cmap='viridis', levels=30)
ax2.set_xlabel("Participation (%)")
ax2.set_ylabel("Study Hours")
ax2.set_title("2D Contour: Fuzzy Grade Output")
fig.colorbar(contour, ax=ax2)

st.pyplot(fig)