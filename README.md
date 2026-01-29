# 📊 Customer Satisfaction Prediction System

An end-to-end Data Analytics + Machine Learning project that analyzes customer support tickets and predicts customer satisfaction using AI.  
This project combines EDA, visualization, ML modeling, and a professional interactive dashboard built with Streamlit.

---

## 📌 Project Overview
Customer satisfaction directly impacts customer retention and brand reputation.  
This project predicts customer satisfaction levels using Machine Learning and visualizes insights through Power BI and a Python-based dashboard.

The project demonstrates a complete analytics lifecycle:
**Data → ML Model → Predictions → Business Dashboards**

---

## 🎯 Objectives
- Analyze customer support ticket data
- Predict customer satisfaction ratings
- Identify customers at risk of dissatisfaction
- Monitor data quality and model performance
- Present insights using Power BI and Python dashboards

---

## 🗂 Project Structure

Customer-Satisfaction-Prediction/
│
├── app/
│   └── app.py                     # Streamlit Dashboard
│
├── data/
│   ├── customer_support_tickets.csv
│   └── eda_clean.csv
│
├── notebook/
│   ├── 01_EDA.ipynb               # Data analysis & visualization
│   └── 02_ML_Model.ipynb          # Model training
│
├── outputs/
│   ├── figures/                   # EDA graphs
│   ├── ml_figures/                # Model performance plots
│   ├── ml_metrics.csv
│   └── predictions.csv
│
├── powerbi_dashboard/
|   ├── Customer_Satisfaction_Dashboard.pbix
│   ├── Overview.png
│   ├── Tickets.png
│   ├── Voice Of Customer.png
|   ├──Data health.png
│   ├── ML Performance.png
│   └── Prediction.png
│
├── check_model.py                 
├── requirements.txt              
└── README.md


---

## 🧠 Machine Learning Workflow

### 1️⃣ Exploratory Data Analysis (EDA)

- Missing values analysis
- Ticket distribution and trends
- Satisfaction patterns by channel and priority
- Text analysis for customer feedback

### 2️⃣ Model Building

- Feature preprocessing
- Classification model training
- Model evaluation using:
- Accuracy
- F1-score
- Recall
- ROC Curve

### 3️⃣ Prediction Generation

Final predictions saved in outputs/predictions.csv
Best trained model saved as best_model.pkl


---

## 📌 Important Note About Model File

- ⚠️ The trained model file `best_model.pkl` is **not** uploaded to GitHub because the file size is too large for GitHub’s upload limits.  
- 👉 All outputs, graphs, metrics, and prediction files **are** included in the repository.  
- If needed, the model can be recreated by running the ML notebook.

---

## 🖥 How to Run This Project Locally

### 1️ Clone the Repository

git clone https://github.com/aparna190417/Customer-Satisfaction-Prediction.git
cd Customer-Satisfaction-Prediction

### 2 Install Required Libraries

pip install -r requirements.txt

### 3 Run the Streamlit Dashboard

cd app
streamlit run app.py

### Run Model Test Script

python check_model.py

---

## 📊 Dashboard Features

- Executive KPI Snapshot  
- Ticket trends & satisfaction analysis  
- Voice of Customer (WordCloud)  
- Model performance visuals  
- Live AI satisfaction prediction  
- Download prediction results as CSV  

---

## 📈 Power BI Dashboard

The Power BI dashboard converts raw data and ML outputs into business-friendly insights.

Dashboard Pages:
- Overview
- Tickets Analysis
- ML Performance
- Data Health
- Prediction
- Voice of Customer

📌 Note:
The .pbix file must be opened using Power BI Desktop.

---

## 🛠 Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- NLP (Text features)  
- Streamlit  
- Power BI  

---

## 🚀 Key Insights

- Identified customers likely to be dissatisfied
- Highlighted critical ticket channels and priorities
- Improved visibility into data quality and ML performance
- Enabled proactive customer support strategies

## 💡 Business Impact

- Supports data-driven customer experience decisions
- Helps reduce customer churn
- Bridges the gap between Machine Learning and business users

## 👩‍💻 Author

**Aparna Patel**
