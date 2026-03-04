# Hybrid AI Query System

## About the Project

Hybrid AI Query System is a simple data analytics application built using **Streamlit and Python**.
The goal of this project is to make data analysis easier by allowing users to interact with data using **natural language questions** instead of writing complex SQL queries or Python code.

The application allows users to connect databases, upload datasets, visualize data, and run basic machine learning models from a single interface.

This project was mainly created as a **learning and portfolio project** to explore how AI can assist with data analytics tasks.

---

## Features

### 1. Database Integration

Users can connect their databases directly to the application.

Supported databases:

* MySQL
* Microsoft SQL Server

Once connected, the application reads the database schema and allows users to ask questions about their data. The system then generates the SQL query automatically.

---

### 2. Data Upload

Users can upload datasets directly into the application.

Supported formats:

* CSV
* Excel (XLSX)
* TXT

Uploaded files are loaded into a pandas dataframe so they can be used for analysis and visualization.

---

### 3. AI Analytics Assistant

The AI assistant allows users to ask questions about their data.

For example:

* "Show total sales by product"
* "Find the highest revenue month"
* "List customers with the highest orders"

The assistant converts the question into either:

* SQL queries (for databases)
* Python data analysis code (for CSV files)

---

### 4. Data Visualization

The application provides simple visualization tools using **Plotly**.

Available charts include:

* Line Chart
* Bar Chart
* Scatter Plot
* Box Plot
* Area Chart
* Histogram
* Heatmap

Users can choose the dataset, select columns, and generate charts directly.

---

### 5. Machine Learning Studio

This module allows users to run basic machine learning models on their dataset.

Currently supported:

**Clustering**

* K-Means clustering
* PCA visualization

**Anomaly Detection**

* Isolation Forest model
* PCA-based anomaly visualization

These tools help users explore patterns and unusual behavior in their data.

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* Scikit-learn
* MySQL Connector
* PyODBC
* Google Gemini API

---

## How to Run the Project

Clone the repository:

```
git clone https://github.com/your-username/Hybrid-AI-Query-System.git
cd Hybrid-AI-Query-System
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the Streamlit app:

```
streamlit run app.py
```

---

## Purpose of the Project

This project was created to explore how AI can assist with data analytics and simplify the process of querying and understanding data.

It combines database connectivity, data visualization, and basic machine learning tools into one simple interface.

---

## Author

Mohd Ibrahim Uddin
