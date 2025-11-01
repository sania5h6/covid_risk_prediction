from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName("COVIDDataHDFS").getOrCreate()

# Read CSV from HDFS
df = spark.read.csv("hdfs://localhost:9000/covid/input/01-01-2022.csv", header=True, inferSchema=True)

# Show the first few rows
df.show(5)

# Example: Count total cases per country
df.groupBy("Country_Region").sum("Confirmed").show()

# Stop Spark session
spark.stop()

