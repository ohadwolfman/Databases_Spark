import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, current_date, avg
import numpy as np

def loadjson():
    # Initialize Spark
    findspark.init()

    # Create SparkSession
    spark = SparkSession.builder.appName("readJSON").getOrCreate()

    # Read the JSON file into a DataFrame
    path = "books.json"
    books = spark.read.json(path, multiLine=True)
    books.show()

    return spark, books

def F_authors(df):
    df.printSchema()
    # Task A1: Print table with books by authors starting with F
    current_year = year(current_date())
    f_authors = df.select('title', 'author', (current_year - df.year).alias('years_since_published'))
    f_authors = f_authors.filter(df.author.startswith('F')).orderBy(df.author)
    f_authors.show()


def english_pages_amount(df):
    # Task A2: Calculation of the average number of pages written by each author for books in English
    avg_pages = df.filter(df.language == "English").groupBy("author","language").agg(avg('pages')).orderBy(df.author)
    avg_pages.show()


def loadTextFile(file_path):
    # Initialize SparkSession
    findspark.init()
    spark = SparkSession.builder.appName("loadTxt").getOrCreate()

    # Read the text file into an RDD
    lines_rdd = spark.sparkContext.textFile(file_path)

    # Split each line by comma to create an RDD of lists
    data_rdd = lines_rdd.map(lambda line: line.split(','))

    # Convert numeric values to float or integer and keep string values as strings
    def parse_row(row):
        parsed_row = []
        for value in row:
            try:
                parsed_row.append(float(value))
            except ValueError:
                parsed_row.append(value)
        return parsed_row

    data_rdd = data_rdd.map(parse_row)

    # Create a DataFrame from the RDD with column names
    column_names = ['loaclPrices', 'bathrooms', 'totalArea', 'residentialArea', 'garages', 'rooms',
                    'bedrooms', 'age', 'buildingType', 'architectureType', 'firefightingSite', 'price']
    df = spark.createDataFrame(data_rdd, column_names)
    df.show()

    # Return the DataFrame
    return spark, df


def splitData(df, trainRatio):
    # Split the DataFrame into train and test sets
    train_data, test_data = df.randomSplit([trainRatio, 1 - trainRatio], seed=42)

    x_train = train_data.select(['loaclPrices', 'bathrooms', 'totalArea', 'residentialArea', 'garages',
                    'rooms', 'bedrooms', 'age', 'buildingType', 'architectureType', 'firefightingSite'])
    y_train = train_data.select('price')
    x_test = test_data.select(['loaclPrices', 'bathrooms', 'totalArea', 'residentialArea', 'garages',
                    'rooms', 'bedrooms', 'age', 'buildingType', 'architectureType', 'firefightingSite'])
    y_test = test_data.select('price')

    # Return the train and test sets
    return x_train, x_test, y_train, y_test

def linearRegression(data_x, data_y):
    data_x = np.array(data_x.toPandas())
    data_y = np.array(data_y.toPandas())
    w = np.array(np.zeros((11,1)))
    b = np.array(np.zeros((19,1)))
    alpha = 0.001

    print("data_x:", data_x.shape, "data_y", data_y.shape, "w", w.shape, "b", b.shape)
    for iteration in range(100000):
        deriv_b = np.mean((np.dot(data_x, w)+b) - data_y)
        gradient_w = 1.0/len(data_y) * np.dot(((np.dot(data_x, w)+b) - data_y).T, data_x).T
        b -= alpha * deriv_b
        w -= alpha * gradient_w

        if iteration % 10000 == 0:
            print("W and b is: ", w, b[0])
    return w, b


def evaluateModel(X_test, y_test, weights, bias):
    # Convert the Spark DataFrames to NumPy arrays
    features = np.array(X_test.toPandas())
    labels = np.array(y_test.toPandas())
    bias = bias[:9]
    print("features:", features.shape, "labels", labels.shape, "weights", weights.shape, "bias", bias.shape)

    # Calculate the predicted labels
    predicted_labels = np.dot(features, weights) + bias

    # Calculate the MSE score
    mse = np.mean((predicted_labels - labels)**2)/2

    # Return the MSE score
    return mse


if __name__ == '__main__':
    # --- Part A ---
    spark, books = loadjson()
    F_authors(books)
    english_pages_amount(books)
    spark.stop()

    # ---Part B---
    spark, apts = loadTextFile('prices.txt')
    print(apts.printSchema())
    x_train, x_test, y_train, y_test = splitData(apts, 0.75)
    x_train.show()
    y_train.show()
    x_test.show()
    y_test.show()

    weights, bias = linearRegression(x_train, y_train)
    mse = evaluateModel(x_test, y_test, weights, bias)
    print("Mean Squared Error:", mse)
    spark.stop()



