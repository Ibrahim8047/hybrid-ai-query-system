import streamlit as st
import mysql.connector
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
try:
    import pyodbc
except ImportError:
    pyodbc = None
client = st.session_state.get("client", None)

if "datasets" not in st.session_state:
    st.session_state.datasets = {}

if "mysql_conn" not in st.session_state:
    st.session_state.mysql_conn = None

if "mysql_error" not in st.session_state:
    st.session_state.mysql_error = None
if "mssql_conn" not in st.session_state:
    st.session_state.mssql_conn = None
if "mssql_error" not in st.session_state:
    st.session_state.mssql_error = None

if "page" not in st.session_state:
    st.session_state.page = "AI Analytics Assistant"
    

page = st.session_state.page


from google import genai
# 🔑 Put your real API key here





# =========================
# GEMINI ANSWER FUNCTION
# =========================
def get_answer_gemini(question, Generic_only):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": "You are a helpful assistant that generates answers based on user questions."}
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {"text": question}
                    ],
                }
            ],
        )

        return response.text

    except Exception as e:
        return f"Error generating answer: {str(e)}"
def get_mssql_schema():
    connection = st.session_state.get("mssql_conn")

    if connection is None:
        st.error("MSSQL not connected.")
        return ""

    try:
        cursor = connection.cursor()
        cursor.execute("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_CATALOG = ?
        """, (st.session_state.current_database,))

        schema_text = ""
        for row in cursor:
            table_name, column_name, data_type = row
            schema_text += f"Table: {table_name} - Column: {column_name} ({data_type})\n"

        return schema_text

    except Exception as e:
        st.error(f"MSSQL schema error: {e}")
        return ""

def get_database_schema():
    if st.session_state.get("mssql_conn"):
        return get_mssql_schema()
    if "mysql_conn" not in st.session_state:
        return ""
    

    connection = st.session_state.mysql_conn
    cursor = connection.cursor()

    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    schema_text = ""

    for table in tables:
        table_name = table[0]
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", connection)
            st.session_state.datasets[f"DB_{table_name}"] = df

        except Exception as e:
            st.warning(f"Could not load table {table_name}: {e}")

        cursor.execute(f"DESCRIBE {table_name}")
        columns = cursor.fetchall()

        schema_text += f"\nTable: {table_name}\n"
        for col in columns:
            schema_text += f" - {col[0]} ({col[1]})\n"

    return schema_text
    


def generate_flat_file_gemini(user_question, datasets):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents= f"""You are a senior Python developer.

STRICT INSTRUCTIONS (MUST FOLLOW):

1. Return ONLY properly formatted Python code.
2. Code MUST be multi-line with correct indentation.
3. Do NOT compress code in one line.
4.  remove try-except blocks.
5. Use proper indentation (4 spaces).
6. The dataset is already loaded in a pandas DataFrame named df.
7. After generating the Python code
8. After each major result, print clear headings.
9. The final output must display:
   - The Python code block
   - Then a separator line
  
RULES:
-use the provided dataset structure to write code that answers the user's question.
-use proper indentation and multi-line code format.
-Do NOT use print().
-Do NOT add headings.
-Do NOT add explanation text.
-Return only st.dataframe(result_df) output.
-Sort values before displaying.
- Use pandas (pd), numpy (np)
- If using Streamlit, use st.write() properly
- No markdown
- No explanation
- No text before or after
- Never display the raw dataset or dictionary.
- Do NOT display sample data
-If visualization is needed, use matplotlib and call plt.show()
User Question:
{user_question}

Dataset Preview:
{datasets}""",
        )

        return response.text

    except Exception as e:
        return f"Error generating insights: {str(e)}"

def generate_sql_with_gemini(user_question):
    schema_info = get_database_schema()

    if st.session_state.get("mssql_conn"):
        db_type = "SQL Server"
    elif st.session_state.get("mysql_conn"):
        db_type = "MySQL"
    else:
        db_type = "Unknown"

    prompt = f"""

You are an expert SQL query generator.

You must convert the user's natural language question into a valid {db_type} SQL query.

STRICT OUTPUT RULES:
- Output ONLY the SQL query.
- Do NOT explain anything.
- Do NOT use markdown.
- Query must be directly executable.
- If impossible, return exactly:
  ERROR: Cannot generate query from given schema.

SCHEMA RULES:
- Use ONLY tables and columns from the provided schema.
- Never invent tables or columns.
- Always verify column names match exactly.
- Use table aliases when needed.

TEXT MATCHING RULES:
- Always use LOWER(column_name) for text comparison.
- Use LIKE with wildcards for entity names.
- Example:
  If user says "netflix",
  generate:
  LOWER(column_name) LIKE '%netflix%'

AGGREGATION RULES:
- If user says "amount", assume SUM(amount) unless "list" is specified.
- Wrap all aggregations with COALESCE(aggregation, 0).
- Use GROUP BY when needed.

FILTER RULES:
- Avoid over-restricting filters.
- Use LIKE instead of strict equality for text fields.
- If multiple filters are given (e.g., pending + low risk),
  apply them using LIKE for flexibility.

SORTING RULES:
- highest / top / maximum → ORDER BY DESC
- lowest / minimum → ORDER BY ASC
- top N → use LIMIT N

DATE RULES:
- today → CURDATE()
- this month → MONTH(date_column) = MONTH(CURDATE())
- this year → YEAR(date_column) = YEAR(CURDATE())


When matching categorical columns (like risk_level, status, type):
Match based on the main keyword only.
Example:
If user says "low risk" and column values are:
Low, Medium, High
Then match using:
LOWER(column) LIKE '%low%'
Do NOT search for the full phrase unless it exists in schema.

Database Schema:
{schema_info}

User Question (may be in any language):
{user_question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0,
            "max_output_tokens": 500  
        }
    )

    sql_query = response.text.strip()

    # Clean any residual markdown formatting
    sql_query = (
        sql_query
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )

    return sql_query
def query_database(sql_query):
    try:
        if st.session_state.get("mysql_conn"):
            connection = st.session_state.mysql_conn
            return pd.read_sql(sql_query, connection)

        elif st.session_state.get("mssql_conn"):
            connection = st.session_state.mssql_conn
            return pd.read_sql(sql_query, connection)

        else:
            return "Error: Database not connected."

    except Exception as e:
        return f"Error executing query: {str(e)}"
with st.sidebar:
    st.title("Hybrid Query system ")
    st.markdown("  ")
    st.markdown("Navigate between different modules:")
    if st.button("Database Integration",use_container_width=True):
        st.session_state.page = "Database_Integration"
    if st.button("Data Upload Center",use_container_width=True):
        st.session_state.page = "Data_Upload_Center"
    if st.button( " AI Analytics Assistant",use_container_width=True):
        st.session_state.page = "AI_Analytics_Assistant"
    if st.button("Data Visualization",use_container_width=True):
        st.session_state.page = "Data_Visualization"
    if st.button("Machine Learning Studio",use_container_width=True):
        st.session_state.page = "Machine_Learning_Studio"
   
    st.markdown("---")
    st.subheader("🔌 Connection status")
    st.markdown("**MySQL:** " + ("🟢 Connected" if st.session_state.mysql_conn else "⚪ Not connected"))
    if st.session_state.mysql_error:
        st.caption(f"MySQL: {st.session_state.mysql_error[:120]}")

    st.markdown("**MSSQL:** " + ("🟢 Connected" if st.session_state.mssql_conn else "⚪ Not connected"))
    if st.session_state.mssql_error:
        st.caption(f"MSSQL: {st.session_state.mssql_error[:120]}")
    api_key = st.text_input(
        "Enter Google GenAI API Key",
        type="password",
        key="api_key_input"
    )

    if st.button("Set API Key", use_container_width=True):

        if api_key:
            st.session_state.client = genai.Client(api_key=api_key)
            st.success("API key set successfully ✓")
        else:
            st.warning("Please enter API key")
def page_Database_Integration():
    st.title("Database Integration")
    st.divider()
    tab1, tab2 = st.tabs(["MySQL","MSSQL"])
    with tab1:
        st.subheader("MySQL Connection")
        st.write("Enter your MySQL database credentials:")
        host = st.text_input("Host", value="localhost")
        user = st.text_input("User", value="root")
        password = st.text_input("Password", type="password")
        database = st.text_input("Database Name", type ="default" , key="mysql_db")
    
        if st.button("Connect to MySQL"):
            try:
                connection = mysql.connector.connect(
                    host=host,
                    user=user,
                    password=password,
                    database=database
                )
                if connection.is_connected():
                    st.success("Successfully connected to MySQL database!")
                    st.session_state.mysql_conn = connection
                    st.session_state.current_database = database
                    get_database_schema()
                    st.success("Database tables loaded into Analytics & ML Lab!")
            except mysql.connector.Error as err:
                st.error(f"Error connecting to MySQL: {err}")
    with tab2:
        st.subheader("MSSQL Connection")
        st.write("Enter your MSSQL database credentials")
        driver = st.text_input("Driver", value ="{SQL SERVER}")
        server = st.text_input("Server", type ="default")
        database = st.text_input("Database Name", type ="default", key="mssql_db")
        trusted = st.checkbox("Trusted Connection", value=True)
        if st.button("Connect to MSSQL"):
            try:
                conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};"

                if trusted:
                    conn_str += "TRUSTED_CONNECTION=YES;"
                else:
                    st.warning("Only trusted connection supported.")
                    return

                mssql_conn = pyodbc.connect(conn_str)
                st.success("Successfully connected to MSSQL database!")

                st.session_state.mssql_conn = mssql_conn
                st.session_state.current_database = database
                st.session_state.datasets = {}
                cursor = mssql_conn.cursor()
                cursor.execute("""
                   SELECT TABLE_NAME
                   FROM INFORMATION_SCHEMA.TABLES
                   WHERE TABLE_TYPE = 'BASE TABLE'
                """)
                tables = cursor.fetchall()

                for table in tables:
                    table_name = table[0]
                    try:
                        df = pd.read_sql(f"SELECT * FROM {table_name}", mssql_conn)

                # 👇 THIS LINE IS IMPORTANT
                        st.session_state.datasets[f"MSSQL_{table_name}"] = df

                    except Exception as e:
                        st.warning(f"Could not load {table_name}: {e}")

                st.success("All MSSQL tables loaded into Analytics & ML Lab!")

            except Exception as e:
                   st.error(f"Error connecting to MSSQL: {e}")

     

def page_Data_Upload_Center():
    st.title("Data Upload Center")
    st.divider()

    st.write("upload CSV/XLSX or TXT Files")
    uploaded_files = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "txt"], accept_multiple_files=False, key="uploader")
    st.divider()
    if uploaded_files is not None:
        name = uploaded_files.name
        ext = name.split(".")[-1].lower()
        if ext in ("csv", "xlsx"):
            try:
                df = pd.read_csv(uploaded_files)
                if not df.empty:
                    st.session_state.datasets[name] = df
                    st.success(f"Loaded dataset: {name} with shape {df.shape}")
                    st.dataframe(df)
            except Exception as e:
                st.error(f"Dataset read error ({name}): {e}")
        elif ext == "txt":
            try:
                if "context_text" not in st.session_state:
                    st.session_state.context_text = ""

                df = uploaded_files.read().decode("utf-8", errors="ignore")
                st.session_state.context_text += "\n" + df
                st.success(f"Loaded text file: {name}")
                st.dataframe(df)
            except Exception as e:
                st.error(f"TXT read error ({name}): {e}")
def page_AI_Analytics_Assistant():
    st.title("AI Analytics Assistant")
    st.divider()

    user_question = st.text_input("Ask your question")

    data_source = st.selectbox(
        "Select data source",
        ["Generic only", "CSV File", "mysql", "mssql"]
    )

    if st.button("Get Answer"):

        if not user_question:
            st.warning("Please enter a question")
            return

        # 🔹 1️⃣ GENERIC
        if data_source == "Generic only":
            answer = get_answer_gemini(user_question, None)
            st.success(answer)

        # 🔹 2️⃣ CSV FILE
        elif data_source == "CSV File":
            # Sirf flat files (DB_ wale nahi)
            flat_files = {
                name: df for name, df in st.session_state.datasets.items()
                if not name.startswith("DB_")
            }

            if not flat_files:
                st.warning("No flat file uploaded.")
                return
            
            # Get the first flat file from the dictionary
            first_file_df = next(iter(flat_files.values()))
            answer = generate_flat_file_gemini(
                user_question,
                f"Columns: {list(first_file_df.columns)}"
            )
            
            answer = answer.replace("```python", "").replace("```", "").strip()
            local_vars = {
                 "df": first_file_df,
                "st": st,
                "pd": pd,
                "np": np
                
            }
            if answer.strip().startswith("Error"):
                st.error(answer)
            else:
                exec(answer, {}, local_vars)

            st.success(answer)

        # 🔹 3️⃣ DATABASE
        elif data_source == "mysql":

            if not st.session_state.get("mysql_conn"):
                st.warning("MySQL not connected.")
                return
            sql_query = generate_sql_with_gemini(user_question)

            st.info(f"Generated SQL:\n{sql_query}")

            result = query_database(sql_query)

            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
            else:
                st.error(result)

        elif data_source == "mssql":

            if not st.session_state.get("mssql_conn"):
                st.warning("MSSQL not connected.")
                return

            sql_query = generate_sql_with_gemini(user_question)

            st.info(f"Generated SQL:\n{sql_query}")

            result = query_database(sql_query)

            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
            else:
                st.error(result)

     
def page_Data_Visualization():
    st.title("Data Visualization")
    st.markdown("---")

    if not st.session_state.datasets:
        st.info("Load a dataset first.")
        return

    ds_name = st.selectbox("Dataset", list(st.session_state.datasets.keys()))
    df = st.session_state.datasets[ds_name]

    st.write(f"Dataset Shape: {df.shape}")

    chart_type = st.selectbox(
        "Chart Type",
        ["Line", "Bar", "Scatter", "Box", "Area",
         "Histogram", "Heatmap"]
    )

    x = st.selectbox("X axis", df.columns)
    y = st.multiselect("Y axis", [c for c in df.columns if c != x])

    if st.button("Generate Chart"):

        try:
            fig = None

            if chart_type == "Line":
                fig = px.line(df, x=x, y=y)

            elif chart_type == "Bar":
                fig = px.bar(df, x=x, y=y)

            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x, y=y[0])

            elif chart_type == "Box":
                fig = px.box(df, x=x, y=y[0])

            elif chart_type == "Area":
                fig = px.area(df, x=x, y=y)

            elif chart_type == "Histogram":
                fig = px.histogram(df, x=y[0])

            elif chart_type == "Heatmap":
                corr = df.select_dtypes(include=[np.number]).corr()
                fig = px.imshow(corr, text_auto=True)

            if fig:
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Chart Error: {e}")
def page_Machine_Learning_Studio():
    st.title("Machine Learning Studio")
    st.markdown("---")

    # -------------------------------
    # Dataset Check
    # -------------------------------
    if "datasets" not in st.session_state or not st.session_state.datasets:
        st.warning("Please load a dataset first from Datasets & Docs.")
        return

    ds_name = st.selectbox(
        "Select Dataset",
        list(st.session_state.datasets.keys())
    )

    df = st.session_state.datasets[ds_name]

    if df is None or df.empty:
        st.error("Selected dataset is empty or invalid.")
        return

    st.write("Dataset Shape:", df.shape)
    st.dataframe(df.head())

    # -------------------------------
    # Problem Type
    # -------------------------------
    problem_type = st.selectbox(
        "Select ML Problem Type",
        ["Clustering", "Anomaly Detection"]
    )

    # -------------------------------
    # Target & Feature Selection
    # -------------------------------
    target_column = None

    
    st.markdown("### 🧩 Select Feature Columns")
    feature_columns = st.multiselect(
        "Feature Columns",
        [col for col in df.columns if col != target_column]
    )

    if not feature_columns:
        st.warning("Please select at least one feature column.")
        return

    # -------------------------------
    # Run Model
    # -------------------------------
    if st.button("Run ML Model"):

        # Clean Data
        df_clean = df.copy().dropna()

        # Encode categorical columns
        for col in df_clean.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))

        st.markdown("### 🔍 Problem Type Detected")
        st.success(problem_type)

        # ===============================
        # CLUSTERING
        # ===============================
        if problem_type == "Clustering":

            x = df_clean[feature_columns]
            df_result = df_clean.copy()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(x)

            model = KMeans(n_clusters=3, random_state=42, n_init=10)
            cluster_labels = model.fit_predict(X_scaled)
            df_result["Cluster"] = cluster_labels

    # ✅ Apply PCA for 2D visualization
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(X_scaled)

            df_result["PCA1"] = pca_components[:, 0]
            df_result["PCA2"] = pca_components[:, 1]
            st.dataframe(df_result)
            fig = px.scatter(
                df_result,
                x="PCA1",
                y="PCA2",
                color="Cluster",
                title="🔍 Cluster Visualization (PCA 2D)",
                template="plotly_white"
            )

            st.plotly_chart(fig)


        # ===============================
        # ANOMALY DETECTION
        # ===============================
        elif problem_type == "Anomaly Detection":
            X = df_clean[feature_columns]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = IsolationForest(
            contamination=0.05,
            random_state=42
            )
            df_result = df_clean.copy()
            df_result["Anomaly"] = model.fit_predict(X_scaled)
            anomaly_count = (df_result["Anomaly"] == -1).sum()
            st.info("Model Used: Isolation Forest")
            st.success(f"Total Anomalies Detected: {anomaly_count}")
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(X_scaled)
            df_result["PCA1"] = pca_components[:, 0]
            df_result["PCA2"] = pca_components[:, 1]
            st.dataframe(df_result)
            fig = px.scatter(
                df_result,
                x="PCA1",
                y="PCA2",
                color="Anomaly",
                title="🔍 Anomaly Detection (PCA 2D)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
           
if st.session_state.page == "Database_Integration":
    page_Database_Integration()
elif st.session_state.page == "Data_Upload_Center":
    page_Data_Upload_Center()
elif st.session_state.page == "AI_Analytics_Assistant":
    page_AI_Analytics_Assistant()
elif st.session_state.page == "Data_Visualization":
    page_Data_Visualization()
elif st.session_state.page == "Machine_Learning_Studio":
    page_Machine_Learning_Studio()

else:
    st.session_state.page = "🧠 AI Analytics Assistant"
    page_AI_Analytics_Assistant()

