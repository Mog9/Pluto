# Pluto

**Pluto** is a no-code, web-based machine learning trainer that allows users to upload CSV files, select features and algorithms, visualize results, and train classical ML models end-to-end through a simple interface.

This project is built for **learning, experimentation, and rapid iteration**, not for production-scale workloads.

try it out: https://plutoml.streamlit.app/
---

## Demo

**Demo Video**  

https://github.com/user-attachments/assets/d873c497-3fe2-461a-9360-b6e9c8e7ca64

The demo shows the complete workflow: CSV upload → preprocessing → model selection → training → evaluation and plots.




---

## Features

- Upload CSV files (maximum size: **3 MB**)
- Automatic data cleaning and preprocessing
- Dataset statistics and shape summary
- Data preview after cleaning
- Manual target column selection
- Custom feature selection
- Automatic detection of problem type (regression or classification)
- Algorithm recommendations based on task type
- Multiple visualization modes
- End-to-end model training through a single interface

---

## Workflow

1. Upload a CSV file  
   - The data is automatically cleaned  
   - Dataset shape and cleaned data statistics are shown  
   - A preview of the cleaned dataset is displayed  

2. Select the target column  

3. Choose feature columns manually for better control over training  

4. Problem type is automatically detected:
   - **Regression** or **Classification**

5. Select algorithms to train

   **Regression Algorithms**
   - Linear Regression  
   - Decision Tree Regressor  
   - Random Forest Regressor  
   - KNN Regressor  
   - Evaluation metric: **R² score**

   **Classification Algorithms**
   - Logistic Regression  
   - Support Vector Machine (SVM)  
   - KNN Classifier  
   - Random Forest Classifier  
   - Evaluation metrics: **Accuracy**, **F1-score**

6. Choose visualization options
   - Individual plots for each selected model  
   - Combined comparison plot highlighting the best-performing model  
   - Both options can be enabled simultaneously  

7. Click **Run Models**

8. View results
   - Trained models
   - Evaluation metrics
   - Visualizations based on selected options

---

## Execution Model and Constraints

This project is intentionally constrained to ensure predictable behavior and fast iteration:

- Maximum CSV size: **3 MB**
- CPU-only training
- Classical machine learning algorithms only
- Training jobs typically complete within a few seconds

These constraints are deliberate and aligned with the educational goals of the project.

---

## What Pluto Is Not

- Not a production ML system  
- Not designed for large datasets or high concurrency  
- Not optimized for deep learning workloads  
- Not intended for GPU-based training  

Pluto focuses on clarity, learning, and controlled experimentation.

---

## Deployment Notes

Pluto is hosted on a free-tier platform.  
Cold starts after inactivity are expected and acceptable for demo and learning purposes.

