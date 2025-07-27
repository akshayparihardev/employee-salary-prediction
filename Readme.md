## 👨‍💻 Employee Income Prediction 💰

### 📚 About

This project provides an end-to-end solution for predicting whether an individual's annual income exceeds $50K. It showcases a complete machine learning workflow, starting from comprehensive data preprocessing and rigorous model evaluation to deployment as a fully interactive web application. The goal is to build a reliable and accessible tool for income prediction.
<hr>

### ✨ Features

- **Interactive Web App 🌐:** A user-friendly interface built with Streamlit that allows for:

  - Single Prediction 👤: Enter an individual's details into a form for an instant result.
  - Batch Prediction 📄: Upload a CSV file to get predictions for multiple records at once.

- **Robust Data Preprocessing 🧹:** A complete pipeline that handles missing values, caps outliers, and performs one-hot encoding for categorical features.

- **Advanced Model Evaluation 📈:**
  - Trains five different classification models.
  - Performs in-depth comparison using key metrics like Accuracy, F1-score, and AUC.
  - Includes interactive visualizations to easily identify the best-performing model.

- **Deployment Ready 📦:** All essential assets, including the trained model, scaler, and encoders, are saved for easy reuse and deployment.
  <hr>

### 🛠️ Tech Stack

- 🐍 Language: Python
- 🧮 Data Libraries: Pandas, NumPy
- 🤖 ML Library: Scikit-learn
- 🌐 Web Framework: Streamlit
- 📊 Visualization: Plotly Express
- 📓 Environment: Jupyter Notebook
- 🔗 Deployment Tunneling: ngrok

<hr>

### 🚀 Installation

Follow these steps to get the project up and running on your local machine.

1. **Clone the repo** 📥
   ```bash 
   git clone https://github.com/akshayparihardev/employee-salary-prediction.git
   ```
3. **Move into the project directory** 📁
   ```bash
   cd employee-salary-prediction
   ```
5. **Install dependencies** ⚙️  
   _(It's recommended to use a virtual environment)_  
   ```bash
   pip install -r requirements.txt
   ```

<hr>

### 💡 Usage

This project is designed for maximum convenience. You can launch the interactive web application directly from the Jupyter Notebook.

- Open the Notebook 📓: Launch the `Employee_Salary_Prediction_App.ipynb` file.
- Run All Cells 🚀: Simply run all the cells from top to bottom.
- Click the ngrok Link ✨: The final cell will automatically start the Streamlit application and output a public ngrok URL. Click this link to open the live web app in your browser.

<hr>

### 📈 Model Performance

The project evaluates five different classification models to find the most effective one. The Gradient Boosting model was chosen for its superior performance, as summarized below.

<img width="1413" height="525" alt="newplot" src="https://github.com/user-attachments/assets/2cb02875-90f1-48af-856f-076b47a1e37a" />

<hr>

### ❤️ Streamlit App UI

<img width="1920" height="874" alt="Screenshot (696)" src="https://github.com/user-attachments/assets/572709d0-55a9-45ed-8a26-0db3fe6306bf" />
<hr>
<img width="1920" height="870" alt="Screenshot (697)" src="https://github.com/user-attachments/assets/78f54e36-bfc3-4942-99c6-b6745f98a500" />

<hr>

### 🙏 Acknowledgements

- 📖 [Adult Dataset from UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- 🤖 [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- 🌐 [Streamlit Documentation](https://docs.streamlit.io/)
