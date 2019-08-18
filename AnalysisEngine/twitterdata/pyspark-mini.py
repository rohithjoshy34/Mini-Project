from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import *

sc =SparkContext()
sqlContext = SQLContext(sc)

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('tweets.csv')
newDf = df.withColumn('Tweets', regexp_replace('Tweets', '[^a-zA-Z0-9]', ' '))
print(newDF)

'''from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd

sc = SparkContext('local','example')  # if using locally
sql_sc = SQLContext(sc)

pandas_df = pd.read_csv('tweets.csv')  # assuming the file contains a header
# pandas_df = pd.read_csv('file.csv', names = ['column 1','column 2']) # if no header
s_df = sql_sc.createDataFrame(pandas_df)
print(s_df)'''

'''from pyspark import SparkContext, SparkConf
import csv

sc =SparkContext()
rdd = sc.textFile("tweets.csv")
rdd = rdd.mapPartitions(lambda x: csv.reader(x))
print(rdd)'''
