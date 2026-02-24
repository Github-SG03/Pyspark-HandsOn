# COMMAND ---------- [markdown]
# Step 1: ⚙️ Initialize Environment & Storage

# COMMAND ----------
from pyspark.sql import SparkSession #type: ignore
from pyspark.sql.functions import col, when, lit, expr, count, avg, min, max, explode, rank, dense_rank, ntile,row_number,countDistinct #type: ignore
from pyspark.sql.window import Window #type: ignore
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType #type: ignore
from logs.logger import logger
import os
import json

# Initialize Spark Session
spark = SparkSession.builder.appName("MyPySparkAutomatic").getOrCreate() #type: ignore

# 1. Read Input Data into log files
logger.info("Reading input data")

# 1. Create Volume for ETL data
CATALOG = "workspace"
SCHEMA = "default"
VOLUME = "my_elt_data"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# Create Volume if it doesn't exist 
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

# 2. Prepare directories
INPUT_VOL = f"{VOLUME_PATH}/input"
OUTPUT_VOL = f"{VOLUME_PATH}/output"
OTHER_VOL = f"{VOLUME_PATH}/other"

# USE DBUTILS for Volumes - this is the "Databricks Way"
for folder in [INPUT_VOL, OUTPUT_VOL, OTHER_VOL]:
    dbutils.fs.mkdirs(folder)  #type: ignore

print("✅ Volume and Folders ready:", VOLUME_PATH)


# COMMAND ---------- [markdown]
# Step 2: 🌉 The Data Bridge (Extract)

# COMMAND ----------
# =====================================================
# STEP 2: Data Bridge (Extract) - SHUTIL FIX
# =====================================================
import os
import shutil

# 1. Correct paths based on your bundle structure
# The error shows your files are at: /Workspace/Users/.../.bundle/default/dev/files/datasets/input/
current_dir = os.getcwd() # e.g., .../files/scripts
parent_dir = os.path.dirname(current_dir) # e.g., .../files

# Your synced data source
local_source_file = os.path.join(parent_dir, "datasets", "input", "2015-summary.csv")

# Your target Volume destination
target_volume_file = f"{INPUT_VOL}/2015-summary.csv"

print(f"📂 Source: {local_source_file}")
print(f"📦 Target: {target_volume_file}")

try:
    # Ensure the target directory in the Volume exists
    os.makedirs(os.path.dirname(target_volume_file), exist_ok=True)
    
    # Use SHUTIL to copy (Avoids dbutils SecurityException)
    shutil.copy(local_source_file, target_volume_file)
    
    print("✅ SUCCESS: File copied to Volume using shutil!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 If it fails, manually verify if the source exists locally using os.path.exists")


# COMMAND ---------- [markdown]
# Step 3: ✨ Read CSV (AUTO SCHEMA) (Cleanse)
# 

# COMMAND ----------
# =====================================================
logger.info("Reading input data")

CSV_PATH= target_volume_file

flight_df_1 = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(CSV_PATH)

flight_df_1.show()
flight_df_1.printSchema()
print("Total Rows:", flight_df_1.count())

# COMMAND ---------- [markdown]
# Step 4: Manual Schema + Bad Records

# COMMAND ----------

my_schema = StructType([
    StructField("COUNTRY_1", StringType()),
    StructField("COUNTRY_2", StringType()),
    StructField("TOTAL_COUNT", IntegerType())
])

flight_df_2 = spark.read \
    .schema(my_schema) \
    .option("mode", "PERMISSIVE") \
    .csv(CSV_PATH)

bad_df = flight_df_2.filter(col("TOTAL_COUNT").isNull())

bad_df.show()


# COMMAND ---------- [markdown]
# STEP 5 — Read JSON (FROM Volume)

# COMMAND ----------

JSON_PATH = f"{INPUT_VOL}/multiline.json"

people_df = spark.read \
    .option("multiline", "true") \
    .json(JSON_PATH)

people_df.show()
people_df.printSchema()

# COMMAND ---------- [markdown]
# STEP 6 — Parquet (Volume-safe)

# COMMAND ----------
PARQUET_PATH = f"{OUTPUT_VOL}/parquet_data"

flight_df_1.write.mode("overwrite").parquet(PARQUET_PATH)

parquet_df = spark.read.parquet(PARQUET_PATH)
parquet_df.show()


# COMMAND ---------- [markdown]
# STEP 7 — CSV Output (Volume)

# COMMAND ----------
CSV_OUT = f"{OUTPUT_VOL}/csv_output"

parquet_df.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(CSV_OUT)

print("✅ CSV written to Volume")


# COMMAND ---------- [markdown]
# STEP 8 — Partitioned CSV (FIXED COLUMN NAME)

# COMMAND ----------
PARTITION_OUT = f"{OUTPUT_VOL}/partitioned_csv"

parquet_df.write \
    .mode("overwrite") \
    .option("header", "true") \
    .partitionBy("COUNTRY_1") \
    .csv(PARTITION_OUT)

print("✅ Partitioned output ready")


# COMMAND ---------- [markdown]
# STEP 9 — Data Engineering Pipeline (READ → TRANSFORM → WRITE)

# COMMAND ----------
#Step 9.0Create a DataFrame using the DataFrame API for pe4rforming transformations in PySpark
ROW_COUNT_MSG = "Total number of rows in the DataFrame:"
data = [(1, 1),(2, 1),(3, 1),(4, 2),(5, 1),(6, 2),(7, 2)]
columns = ["id", "num"]
example_df = spark.createDataFrame(data, columns)
print("✅ Example DataFrame Created")
print(ROW_COUNT_MSG, example_df.count())
example_df.printSchema()

#STEP 9.1:Transformation in PySpark:Using Select Method
emp_data = [
    (1,"Amit",70000,"M","INDIA","Delhi",25),
    (2,"Neha",80000,"F","JAPAN","Tokyo",28),
    (3,"Raj",60000,"M","INDIA","Mumbai",30),
    (4,"Sara",90000,"F","JAPAN","Osaka",26),
    (5,"Tom",75000,"M","USA","NY",35)]
emp_schema = ["id","name","salary","gender","address","city","age"]
employee_df = spark.createDataFrame(emp_data, emp_schema)
# Create a Transformation using the DataFrame API & storing in another DataFrame as variable
employee_df_1 = employee_df.select("id", "salary", (col("id") + 5).alias("id_plus_5"), employee_df.gender, employee_df["address"])
employee_df_2 = employee_df.select(expr("id+5").alias("id_plus_5"), expr("salary*2").alias("salary_times_2"), expr("concat(name, address)").alias("name_address"))
#show the DataFrame
employee_df_1.show(truncate=False)
employee_df_2.show(truncate=False)

#STEP 9.2:Transformation in PySpark:Using Spark SQL(select query)
#Crating a temporary view for the DataFrame
employee_df.createOrReplaceTempView("employee_tbl")
# Create a SQL query to select the desired columns
employee_tbl_1=spark.sql("""select * from employee_tbl where salary > 70000""")
employee_tbl_1.show(truncate=False)

#STEP 9.3:Transformation in PySpark:Using dataframe API Filter,Aliases,Literal,Casting,etc
#Alises the DataFrame
employee_df_4 = employee_df.select("id", "salary", (col("id") + 5).alias("id_plus_5"), employee_df.gender, employee_df["address"])
# Filtering the DataFrame using the DataFrame API
employee_df_5 = employee_df.filter(col("address") == "JAPAN")
employee_df_6 = employee_df.filter((col("address") == "JAPAN") & (col("salary") > 70000)) \
    .select("id", "salary", (col("id") + 5).alias("id_plus_5"))
employee_df_7 = employee_df.select("id", "salary", (col("id") + 5).alias("id_plus_5")).where("address = 'JAPAN' and salary > 70000")
#Literal function used to create a column with a constant value
employee_df_8 = employee_df.select("*", lit("Gupta").alias("last_name"))
employee_df_9= employee_df.withColumn("last_name", lit("Gupta"))
#Renaming the columns
employee_df_10 = (
    employee_df.withColumnRenamed("id", "emp_id")
    .withColumnRenamed("salary", "emp_salary")
    .withColumnRenamed("address", "emp_address")
    .withColumnRenamed("gender", "emp_gender")
    .withColumnRenamed("name", "emp_name")
    .withColumnRenamed("last_name", "emp_last_name")
    .withColumnRenamed("age", "emp_age"))
#Casting the column
employee_df_11 = employee_df.withColumn("id", col("id").cast(StringType())).withColumn("salary", col("salary").cast("long"))
#Dropping the column
employee_df_12 = employee_df.drop("last_name", "age", "address", "gender", "name", "id",)
# Show the DataFrame
employee_df_4.show(truncate=False)
employee_df_5.show(truncate=False)
employee_df_6.show(truncate=False)
employee_df_7.show(truncate=False)
employee_df_8.show(truncate=False)
employee_df_9.show(truncate=False)
employee_df_10.show(truncate=False)
employee_df_11.printSchema()
employee_df_12.show(truncate=False)
employee_df.show(truncate=False)

#STEP 9.4:Transformation in PySpark:Using dataframe API Union & Union All(Same in Datafreme API But Different in Spark SQL)
#Create a Data for manager1
data=[(10 ,'Anil',50000, 18),
(11 ,'Vikas',75000,  16),
(12 ,'Nisha',40000,  18),
(13 ,'Nidhi',60000,  17),
(14 ,'Priya',80000,  18),
(15 ,'Mohit',45000,  18),
(16 ,'Rajesh',90000, 10),
(17 ,'Raman',55000, 16),
(18 ,'Sam',65000,   17),
(18 ,'Sam',65000,   17)]
# Create a schema for the DataFrame
schema=['id', 'name', 'sal', 'mngr_id']
# Create a DataFrame using the DataFrame API
manager_df_1 = spark.createDataFrame(data, schema)
# Show the DataFrame
manager_df_1.show(truncate=False)
#show the schema of the DataFrame
manager_df_1.printSchema()
#show the total number of rows in the DataFram
print(ROW_COUNT_MSG, manager_df_1.count())
#create data for manager2
data1=[(19 ,'Sohan',50000, 18),
(20 ,'Sima',75000,  17)]
# Create a schema for the DataFrame
schema1=['id', 'name', 'sal', 'mngr_id']
# Create a DataFrame using the DataFrame API
manager_df_2 = spark.createDataFrame(data1, schema1)
# Show the DataFrame
manager_df_2.show(truncate=False)
#show the schema of the DataFrame
manager_df_2.printSchema()
#show the total number of rows in the DataFrame
print(ROW_COUNT_MSG, manager_df_2.count())
#Union of two DataFrames
manager_df_union = manager_df_1.union(manager_df_2)
manager_df_unionAll= manager_df_1.unionAll(manager_df_2)
manager_df_unionByName= manager_df_1.unionByName(manager_df_2)
# Show the DataFrame
manager_df_union.show(truncate=False)
print(ROW_COUNT_MSG, manager_df_union.count())
manager_df_unionAll.show(truncate=False)
print(ROW_COUNT_MSG, manager_df_unionAll.count())
manager_df_unionByName.show(truncate=False)
print(ROW_COUNT_MSG, manager_df_unionByName.count())

#STEP 9.5:Transformation in PySpark:Using dataframe API-Case(if-else comaprison using when/otherwise)
# Create data for DataFrame
emp_data = [
(1,'manish',26,20000,'india','IT'),
(2,'rahul',None,40000,'germany','engineering'),
(3,'pawan',12,60000,'india','sales'),
(4,'roshini',44,None,'uk','engineering'),
(5,'raushan',35,70000,'india','sales'),
(6,None,29,200000,'uk','IT'),
(7,'adam',37,65000,'us','IT'),
(8,'chris',16,40000,'us','sales'),
(None,None,None,None,None,None),
(7,'adam',37,65000,'us','IT')]
# Create a schema for the DataFrame
schema = ['id', 'name', 'age', 'salary', 'address', 'department']
# Create a DataFrame using the DataFrame API
emp_df= spark.createDataFrame(emp_data, schema)
# Show the DataFrame
emp_df.show(truncate=False)
#show the schema of the DataFrame
emp_df.printSchema()
#show the total number of rows in the DataFrame
print(ROW_COUNT_MSG, emp_df.count())
#Checking the Age of the employee if they are adult or not(otherwise).Assuming emp_df is your original DataFrame
emp_df_1 = emp_df.withColumn(
    "is_adult",
    when(col("age").isNull(), None)           # If age is null → null
    .when(col("age") > 18, "Yes")             # If age > 18 → "Yes"
    .otherwise("No"))                         # Otherwise → "No")
emp_df_2 = emp_df.withColumn(
    "is_adult",
    when((col("age")>0) &(col("age")<18), "minor")          
    .when((col("age")>18) &(col("age")<30), "medium")                        
    .otherwise("major"))
# Show the DataFrame
emp_df_1.show(truncate=False)
emp_df_2.show(truncate=False)


#STEP 9.6:Transformation in PySpark:Using dataframe API-Case(Unique & Sorted Record in datafarame)
# Create data for DataFrame
data=[(10 ,'Anil',50000, 18),
(11 ,'Vikas',75000,  16),
(12 ,'Nisha',40000,  18),
(13 ,'Nidhi',60000,  17),
(14 ,'Priya',80000,  18),
(15 ,'Mohit',45000,  18),
(16 ,'Rajesh',90000, 10),
(17 ,'Raman',55000, 16),
(18 ,'Sam',65000,   17),
(15 ,'Mohit',45000,  18),
(13 ,'Nidhi',60000,  17),      
(14 ,'Priya',90000,  18),  
(18 ,'Sam',65000,   17)]
# Create a schema for the DataFrame
schema = ['id', 'name', 'sal', 'mngr_id']
# Create a DataFrame using the DataFrame API
mngr_df = spark.createDataFrame(data, schema)
# Show the DataFrame
mngr_df.show(truncate=False)
#show the schema of the DataFrame
mngr_df.printSchema()
#show the total number of rows in the DataFrame
print(ROW_COUNT_MSG, mngr_df.count())
# Finding unique records & deleting/droping duplicates in the DataFrame
mngr_df_1 = mngr_df.distinct()
mngr_df_2 = mngr_df.select("id", "name").distinct() #selecting distinct records from dataframe created using the id & name columns
mngr_df_3 = mngr_df.dropDuplicates(["id", "name", "sal", "mngr_id"]) #droping duplicates from the DataFrame using the id & name columns
#sorting the DataFrame
mngr_df_4 = mngr_df_1.sort(col("sal").desc(),col("name").asc()) #sorting the DataFrame using the sal column
#show the schema of the DataFrame
mngr_df_1.show(truncate=False)
mngr_df_2.show(truncate=False)
mngr_df_3.show(truncate=False)
mngr_df_4.show(truncate=False)


#STEP 9.7:Transformation in PySpark:Using dataframe API-Aggregate function
## Create data for DataFrame
empl_data = [
(1,'manish',26,20000,'india','IT'),
(2,'rahul',None,40000,'germany','engineering'),
(3,'pawan',12,60000,'india','sales'),
(4,'roshini',44,None,'uk','engineering'),
(5,'raushan',35,70000,'india','sales'),
(6,None,29,200000,'uk','IT'),
(7,'adam',37,65000,'us','IT'),
(8,'chris',16,40000,'us','sales'),
(None,None,None,None,None,None),
(7,'adam',37,65000,'us','IT')]
# Create a schema for the DataFrame
schema = ['id', 'name', 'age', 'salary', 'address', 'department']
# Create a DataFrame using the DataFrame API
empl_df= spark.createDataFrame(empl_data, schema)
# Show the DataFrame
empl_df.show(truncate=False)
#show the schema of the DataFrame
empl_df.printSchema()
#show the total number of rows in the DataFrame
print(ROW_COUNT_MSG, empl_df.count())
#count() function is used to count the number of rows in the DataFrame
empl_df_1 = empl_df.select(count("*"))
empl_df_2 = empl_df.select(count("name")) 
empl_df_3 = empl_df.select(countDistinct("address").alias("distinct_address_count")) #counting the distinct records in the DataFrame using the address column
#min().max() and avg() function is used to find the minimum, maximum and average of the column in the DataFrame
empl_df_4 = empl_df.select(min("salary").alias("min_salary"), max("salary").alias("max_salary"), avg("salary").alias("avg_salary")) #finding the min, max and avg of the salary column in the DataFrame
#show the DataFrame
empl_df_1.show(truncate=False)
empl_df_2.show(truncate=False)
empl_df_3.show(truncate=False)
empl_df_4.show(truncate=False)

#STEP 9.8:Transformation in PySpark:Using dataframe API-GroupBy
#data for DataFrame
data=[(1,'manish',50000,"IT"),
(2,'vikash',60000,"sales"),
(3,'raushan',70000,"marketing"),
(4,'mukesh',80000,"IT"),
(5,'pritam',90000,"sales"),
(6,'nikita',45000,"marketing"),
(7,'ragini',55000,"marketing"),
(8,'rakesh',100000,"IT"),
(9,'aditya',65000,"IT"),
(10,'rahul',50000,"marketing")]
# Create a schema for the DataFrame
schema = ['id', 'name', 'salary', 'department']
# Create a DataFrame using the DataFrame API
dept_df = spark.createDataFrame(data, schema)
# Show the DataFrame
dept_df.show(truncate=False)
#show the schema of the DataFrame
dept_df.printSchema()
#show the total number of rows in the DataFrame
print(ROW_COUNT_MSG, dept_df.count())
#groupby() function is used to group the DataFrame by the department column
dept_df_2 = dept_df.groupBy("department").agg(count("*").alias("count"), avg("salary").alias("avg_salary"), min("salary").alias("min_salary"), max("salary").alias("max_salary")) #counting the number of records in the DataFrame using the department column and finding the min, max and avg of the salary column in the DataFrame
# Show the DataFrame
dept_df_2.show(truncate=False)


#STEP 9.9:Transformation in PySpark:Using dataframe API-joins
#Create 'costomer_data' data for dataframe
customer_data = [(1,'manish','patna',"30-05-2022"),
(2,'vikash','kolkata',"12-03-2023"),
(3,'nikita','delhi',"25-06-2023"),
(4,'rahul','ranchi',"24-03-2023"),
(5,'mahesh','jaipur',"22-03-2023"),
(6,'prantosh','kolkata',"18-10-2022"),
(7,'raman','patna',"30-12-2022"),
(8,'prakash','ranchi',"24-02-2023"),
(9,'ragini','kolkata',"03-03-2023"),
(10,'raushan','jaipur',"05-02-2023")]
# Create a schema for the DataFrame
customer_schema=['customer_id','customer_name','address','date_of_joining']
# Create a DataFrame using the DataFrame API
customer_df = spark.createDataFrame(customer_data, customer_schema)
# Show the DataFrame
customer_df.show(truncate=False)
#show the schema of the DataFrame
customer_df.printSchema()
#show the total number of rows in the DataFrame
print(ROW_COUNT_MSG, customer_df.count())
# Create a new DataFrame with the date column converted to a date type
#Create 'sales_data' data for dataframe
sales_data = [(1,22,10,"01-06-2022"),
(1,27,5,"03-02-2023"),
(2,5,3,"01-06-2023"),
(5,22,1,"22-03-2023"),
(7,22,4,"03-02-2023"),
(9,5,6,"03-03-2023"),
(2,1,12,"15-06-2023"),
(1,56,2,"25-06-2023"),
(5,12,5,"15-04-2023"),
(11,12,76,"12-03-2023")]
# Create a schema for the DataFrame
sales_schema=['customer_id','product_id','quantity','date_of_purchase']
# Create a DataFrame using the DataFrame API
sales_df = spark.createDataFrame(sales_data, sales_schema)
# Show the DataFrame
sales_df.show(truncate=False)
#show the schema of the DataFrame
sales_df.printSchema()
#show the total number of rows in the DataFrame
print(ROW_COUNT_MSG, sales_df.count())
#Create 'product_data' data for dataframe
product_data = [(1, 'fanta',20),
(2, 'dew',22),
(5, 'sprite',40),
(7, 'redbull',100),
(12,'mazza',45),
(22,'coke',27),
(25,'limca',21),
(27,'pepsi',14),
(56,'sting',10)]
# Create a schema for the DataFrame
product_schema=['id','name','price']
# Create a DataFrame using the DataFrame API
product_df = spark.createDataFrame(product_data, product_schema)
# Show the DataFrame
product_df.show(truncate=False)
#show the schema of the DataFrame
product_df.printSchema()
#show the total number of rows in the DataFrame
print(ROW_COUNT_MSG, product_df.count())
# Inner Join
customer_sales_inner_df = customer_df.join(sales_df, customer_df.customer_id == sales_df.customer_id, "inner").select(sales_df.product_id).sort(col("product_id").asc())#join on single column
# Left Join
customer_sales_left_df = customer_df.join(sales_df, customer_df.customer_id == sales_df.customer_id, "left").select(sales_df.product_id).sort(col("product_id").asc())#join on single column
# Right Join
customer_sales_right_df = customer_df.join(sales_df, customer_df.customer_id == sales_df.customer_id, "right").select(sales_df.product_id).sort(col("product_id").asc())#join on single column
#Outer Join
customer_sales_outer_df = customer_df.join(sales_df, customer_df.customer_id == sales_df.customer_id, "outer").select(sales_df.product_id).sort(col("product_id").asc())#join on single column
# Cross Join
customer_sales_cross_df = customer_df.crossJoin(sales_df)
# Show the DataFrame
customer_sales_inner_df.show(truncate=False)
customer_sales_left_df.show(truncate=False)
customer_sales_right_df.show(truncate=False)
customer_sales_outer_df.show(truncate=False)
customer_sales_cross_df.show(truncate=False)

#STEP 9.10:Transformation in PySpark:Using dataframe API-Window Function
#data for DataFrame
e_data = [(1,'manish',50000,'IT','m'),
(2,'vikash',60000,'sales','m'),
(3,'raushan',70000,'marketing','m'),
(4,'mukesh',80000,'IT','m'),
(5,'priti',90000,'sales','f'),
(6,'nikita',45000,'marketing','f'),
(7,'ragini',55000,'marketing','f'),
(8,'rashi',100000,'IT','f'),
(9,'aditya',65000,'IT','m'),
(10,'rahul',50000,'marketing','m'),
(11,'rakhi',50000,'IT','f'),
(12,'akhilesh',90000,'sales','m')]
# Create a schema for the DataFrame
schema = ['id', 'name', 'salary', 'department', 'gender']
# Create a DataFrame using the DataFrame API
e_df = spark.createDataFrame(e_data, schema)
# Show the DataFrame
e_df.show(truncate=False)
#show the schema of the DataFrame
e_df.printSchema()
#show the total number of rows in the DataFrame
print(ROW_COUNT_MSG, e_df.count())
# Create product_data for DataFrame API
product_data = [
(1,"iphone","01-01-2023",1500000),
(2,"samsung","01-01-2023",1100000),
(3,"oneplus","01-01-2023",1100000),
(1,"iphone","01-02-2023",1300000),
(2,"samsung","01-02-2023",1120000),
(3,"oneplus","01-02-2023",1120000),
(1,"iphone","01-03-2023",1600000),
(2,"samsung","01-03-2023",1080000),
(3,"oneplus","01-03-2023",1160000),
(1,"iphone","01-04-2023",1700000),
(2,"samsung","01-04-2023",1800000),
(3,"oneplus","01-04-2023",1170000),
(1,"iphone","01-05-2023",1200000),
(2,"samsung","01-05-2023",980000),
(3,"oneplus","01-05-2023",1175000),
(1,"iphone","01-06-2023",1100000),
(2,"samsung","01-06-2023",1100000),
(3,"oneplus","01-06-2023",1200000)]
# Create a schema for the DataFrame
product_schema = ['product_id', 'product_name', 'sales_date', 'sales']
# Create a DataFrame using the DataFrame API
product_df = spark.createDataFrame(product_data, product_schema)
# Show the DataFrame
product_df.show(truncate=False)
#show the schema of the DataFrame
product_df.printSchema()
#show the total number of rows in the DataFrame
print(ROW_COUNT_MSG, product_df.count())
#create data for DataFrame
empls_data = [(1,"manish","11-07-2023","10:20"),
        (1,"manish","11-07-2023","11:20"),
        (2,"rajesh","11-07-2023","11:20"),
        (1,"manish","11-07-2023","11:50"),
        (2,"rajesh","11-07-2023","13:20"),
        (1,"manish","11-07-2023","19:20"),
        (2,"rajesh","11-07-2023","17:20"),
        (1,"manish","12-07-2023","10:32"),
        (1,"manish","12-07-2023","12:20"),
        (3,"vikash","12-07-2023","09:12"),
        (1,"manish","12-07-2023","16:23"),
        (3,"vikash","12-07-2023","18:08")]
# Create a schema for the DataFrame
emp_schema = ["id", "name", "date", "time"]
# Create a DataFrame using the DataFrame API
empls_df = spark.createDataFrame(data=empls_data, schema=emp_schema)
# Show the DataFrame
empls_df.show(truncate=False)
#show the schema of the DataFrame
empls_df.printSchema()
#show the total number of rows in the DataFrame
print(ROW_COUNT_MSG, empls_df.count())
##0.Create a Window specification
window_spec = Window.partitionBy("department").orderBy(col("salary").desc())
# Create a new column with the rank of each row within its department
e_window_df = e_df.withColumn("row_number", row_number().over(window_spec)).withColumn("rank", rank().over(window_spec)).withColumn("dense_rank", dense_rank().over(window_spec)).withColumn("ntile", ntile(3).over(window_spec))
#show the DataFrame
e_window_df.show(truncate=False)

##1.Create a new column with the percentage of sales of each product in each month
# Step 1: Define the lag window (ordered by month per product)
#window_spec_lag = Window.partitionBy("product_id").orderBy("sales_month")
# Step 2: Get previous month's sales
# product_previous_df = product_df.withColumn("previous_month_sales",lag("sales").over(window_spec_lag))
# Step 3: Calculate percentage gain/loss (handle nulls safely)
#per_loss_gain_df = product_previous_df.withColumn("percentage_loss_gain",round(((col("sales") - col("previous_month_sales")) / col("previous_month_sales")) * 100,2)
# Show result
#per_loss_gain_df.select("product_id", "sales_month", "sales", "previous_month_sales", "percentage_loss_gain").show()


##2.Create a new column with the percentage of sales of each product in each month
# Step 1: Truncate date to month level
#product_df = product_df.withColumn("sales_month", trunc("sales_date", "month"))
# Step 2: Define window partitioned by product_id and ordered by month
#window_spec = Window.partitionBy("product_id").orderBy("sales_month")
# Step 3: Calculate monthly sum of sales for each product
#sum_sales_df = product_df.withColumn("sum_sales", sum(col("sales")).over(window_spec))
# Step 4: Calculate percentage
#per_sales_each_month_df = sum_sales_df.withColumn("percentage_sales_each_month",(col("sales") / col("sum_sales") * 100).cast("double"))
# Optional: Round the percentage to 2 decimals
#per_sales_each_month_df = per_sales_each_month_df.withColumn("percentage_sales_each_month", round("percentage_sales_each_month", 2))
# Show final result
#per_sales_each_month_df.select("product_id", "sales_month", "sales", "sum_sales", "percentage_sales_each_month").show()

##3.Create a new column with the first and latest sales & to find the difference between the sales of each product from the first and last month sales
#Ensure sales_date is of DateType
#sales_df = sales_df.withColumn("sales_date", to_date(col("sales_date")))
# Define window partitioned by product and ordered by sales_date
#w = Window.partitionBy("product_id").orderBy("sales_date").rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
# Add first and last sales for each product
#sales_df_with_extremes = sales_df.withColumn("first_sale", first("sales_amount").over(w)).withColumn("last_sale", last("sales_amount").over(w))
# Calculate difference
#sales_df_final = sales_df_with_extremes.withColumn("sales_difference", col("last_sale") - col("first_sale"))
# Show only unique result per product (optional)
#sales_df_final.select("product_id", "first_sale", "last_sale", "sales_difference").distinct().show()

##4.send an email to all employees who have not completed compulsory 8 hour office work when they are in the office

##5.find out the performance of the sales based on the last three months average sales of each product
# Ensure the sales_date column is in DateType
#sales_df = sales_df.withColumn("sales_date", to_date(col("sales_date")))
# Define a window partitioned by product_id and ordered by date
#window_spec = Window.partitionBy("product_id").orderBy(col("sales_date")).rowsBetween(-90, -1)
# Calculate 3-month rolling average (excluding the current row)
#sales_df_with_avg = sales_df.withColumn("avg_sales_last_3_months",avg("sales_amount").over(window_spec))
# Calculate performance compared to average
#sales_df_with_perf = sales_df_with_avg.withColumn("performance_vs_avg",when(col("avg_sales_last_3_months").isNull(), None).when(col("sales_amount") > col("avg_sales_last_3_months"), "Above Average").when(col("sales_amount") < col("avg_sales_last_3_months"), "Below Average").otherwise("Average"))
#.select("product_id", "sales_date", "sales_amount", "avg_sales_last_3_months", "performance_vs_avg").show()
#last_3_months_avg_sales_df = product_df.withColumn("last_3_months_avg_sales", avg(col("sales")).over(window_spec_agg))
#STEP 9.11:Transformation in PySpark:Using dataframe API-Nested Json Flattening

# Read the JSON file into a DataFrame
# 1. Define your JSON string
nested_json_str = """
{
  "restaurants": [
    { "restaurant": { "R": { "res_id": "1001" } } },
    { "restaurant": { "R": { "res_id": "1002" } } }
  ]
}
"""

# Parse JSON string to dict
data_dict = json.loads(nested_json_str)

# Create DataFrame from dict
nested_json_df = spark.createDataFrame([data_dict])

# 4. Show the DataFrame and Schema
print("📋 Initial Data Preview:")
nested_json_df.show(truncate=False)
nested_json_df.printSchema()
print("Total number of rows:", nested_json_df.count())

# 5. Flattening the nested JSON
# Step A: Explode the 'restaurants' array into new rows
nested_json_df_0 = nested_json_df.select("*", explode(col("restaurants")).alias("new_restaurant"))

# Step B: Drop the original array column
nested_json_df_1 = nested_json_df_0.drop("restaurants")

# Step C: Select the deep nested ID (new_restaurant -> restaurant -> R -> res_id)
nested_json_df_2 = nested_json_df_1.select("new_restaurant.restaurant.R.res_id")

# 6. Final Result
print("✨ Flattened Result (res_id only):")
nested_json_df_2.show(truncate=False)

logger.info("Transformations complete")

# COMMAND ---------- [markdown]
# FINAL STEP — Copy Results BACK to datasets/output

# COMMAND ----------
# =====================================================
# STEP 10: OUTBOUND BRIDGE (FOR VS CODE SYNC)
# =====================================================

#
logger.info("Writing output data")

# 1. Define Paths (Absolute paths are safer for shutil)
# current_dir is usually /Workspace/Users/.../dev/files/scripts
parent_dir = os.path.dirname(current_dir)
workspace_output_target = os.path.join(parent_dir, "datasets", "output")
workspace_other_target = os.path.join(parent_dir, "datasets", "other")

def sync_back_to_workspace(volume_src, workspace_dest):
    # Check if Volume folder exists locally on the driver mount
    if os.path.exists(volume_src):
        try:
            # Clear the old workspace destination if it exists to avoid 'File exists' errors
            if os.path.exists(workspace_dest):
                shutil.rmtree(workspace_dest)
            
            # Create the destination folder
            os.makedirs(os.path.dirname(workspace_dest), exist_ok=True)
            
            # SHUTIL COPY: Bridges Volume mount -> Workspace
            shutil.copytree(volume_src, workspace_dest)
            print(f"🎉 Synced back to Workspace: {workspace_dest}")
        except Exception as e:
            print(f"⚠️ Shutil Sync failed for {volume_src}: {e}")
    else:
        print(f"❌ Source Volume folder not found: {volume_src}")

# 2. Run the sync back
sync_back_to_workspace(OUTPUT_VOL, workspace_output_target)
sync_back_to_workspace(OTHER_VOL, workspace_other_target)

print("\n✅ DONE. You can now run 'databricks bundle sync --pull' in VS Code.")

#ETL pipeline finished
logger.info("ETL Pipeline Finished")

# COMMAND ---------- [markdown]
# Step 11: ⚙️ Add AWS S3 Output

import boto3 #type:ignore
import os

def upload_to_s3(local_file, bucket, key, region="eu-north-1"):

    s3 = boto3.client(
        "s3",
        region_name=region
    )

    s3.upload_file(
        local_file,
        bucket,
        key,
        ExtraArgs={
            "ExpectedBucketOwner": os.getenv("AWS_ACCOUNT_ID")
        }
    )

    print(f"✅ Uploaded to s3://{bucket}/{key}")


# Call after ETL
upload_to_s3(
    "datasets/output/result.csv",
    "sos-databricks-bucket",
    "etl/result.csv"
)
