import time

from pyspark.sql.functions import col, current_date
import findspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import year


def loadjson():
    # Initialize Spark
    findspark.init()

    # Create SparkSession
    spark = SparkSession.builder.appName("readJSON").getOrCreate()

    # Read the JSON file into a DataFrame
    path = "books.json"
    books = spark.read.json(path, multiLine=True)
    books.show()

    return books

def F_authors(df):
    df.printSchema()
    # Task A: Print table with books by authors starting with F
    print(current_date)
    ans = df.select(['title','author',year(current_date)-'year'])
    ans.show()


def english_pages_amount(spark,df):
    # Task B: Calculate average number of pages written by each author for English books
    df_filtered_english = df.filter(col("language") == "English")
    df_avg_pages = df_filtered_english.groupBy("author").agg({"pages": "avg"})
    df_avg_pages.show()

    # Stop the SparkSession
    spark.stop()


if __name__ == '__main__':
    # ---Part A---
    books = loadjson()
    F_authors(books)

    # ---Part B---
    # english_pages_amount(spark,df)

