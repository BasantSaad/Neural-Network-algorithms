# 🎓 Fuzzy Logic Grade Predictor

A Streamlit web application that predicts student grades using fuzzy logic based on class participation and weekly study hours.

![Demo Screenshot](demo-screenshot.png) *(Replace with actual screenshot)*

## Features

- 📊 Interactive fuzzy logic grade prediction
- 🎚️ Adjustable input sliders for:
  - Class participation (0-100%)
  - Weekly study hours (0-20 hours)
- 📈 Visualizations:
  - 3D surface plot showing grade outputs
  - 2D contour plot of the fuzzy logic system
- 🎯 Real-time grade prediction display

## How It Works

The application uses fuzzy logic with these components:

### Input Variables
1. **Class Participation (%)**
   - Membership functions: Low, Medium, High
2. **Weekly Study Hours**
   - Membership functions: Low, Medium, High

### Output Variable
- **Predicted Grade (%)**
   - Membership functions: Poor, Average, Excellent

### Fuzzy Rules
The system implements 9 rules covering all combinations of input membership functions to determine the output grade.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fuzzy-grade-predictor.git
   cd fuzzy-grade-predictor
   
