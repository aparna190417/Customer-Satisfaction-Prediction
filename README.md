# 📊 Customer Satisfaction Prediction System

An end-to-end Data Analytics + Machine Learning project that analyzes customer support tickets and predicts customer satisfaction using AI.  
This project combines EDA, visualization, ML modeling, and a professional interactive dashboard built with Streamlit.

---

## 🚀 Project Highlights

- Industry-style data cleaning & preprocessing  
- Insightful EDA visualizations  
- Machine Learning model to predict satisfaction rating  
- Interactive AI dashboard for live predictions  
- Business-focused KPIs & analytics  
- Power BI dashboard included for reporting  

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

## 🤖 Machine Learning Model

The model predicts **Customer Satisfaction Rating (1–5)** using:

- Ticket type & priority  
- Channel (Chat, Email, Phone, Social Media)  
- Response & resolution times  
- Customer demographics  
- Ticket subject & description (NLP features)  

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

A business intelligence dashboard is also included inside the `powerbi_dashboard` folder for executive reporting.

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

## 👩‍💻 Author

**Aparna Patel**
