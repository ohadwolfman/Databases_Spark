import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, current_date, col, avg
import numpy as np
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

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


# def splitData(apts, train_ratio):
#     # Split the data into features and target
#     train_rows = apts.select().orderBy()
#     X = apts.select(['loaclPrices', 'bathrooms', 'totalArea', 'residentialArea', 'garages', 'rooms',
#                     'bedrooms', 'age', 'buildingType', 'architectureType', 'firefightingSite'])
#     y = apts.select('price')
#
#     # Get the number of samples in the dataset
#     num_samples = apts.count()
#     print("num_samples:", num_samples)
#
#     # Calculate the number of samples for training and testing
#     num_train_samples = int(train_ratio * num_samples)
#
#     # Split the data into training and testing sets
#     # Function to get rows at specific places
#     def getrows(df, start, end):
#         return df.rdd.zipWithIndex().filter(lambda x: x[1] in range(start, end).map(lambda x: x[0]))
#
#     x_collection = X.rdd.zipWithIndex().collect()
#     y_collection = y.rdd.zipWithIndex().collect()
#     X_train =
#     X_test = getrows(X, start=num_train_samples+1, end=num_samples)
#     y_train = getrows(y, start=0, end=num_train_samples)
#     y_test = getrows(y, start=num_train_samples+1, end=num_samples)
#     print("The data split")
#     print(x_train)
#     return X_train, y_train, X_test, y_test

def splitData(df, trainRatio):
    # Split the DataFrame into train and test sets
    train_data, test_data = df.randomSplit([trainRatio, 1 - trainRatio], seed=42)

    x_train = train_data.select(['loaclPrices', 'bathrooms', 'totalArea', 'residentialArea', 'garages',
                    'rooms', 'bedrooms', 'age', 'buildingType', 'architectureType', 'firefightingSite'])
    x_test = train_data.select('price')
    y_train = test_data.select(['loaclPrices', 'bathrooms', 'totalArea', 'residentialArea', 'garages',
                    'rooms', 'bedrooms', 'age', 'buildingType', 'architectureType', 'firefightingSite'])
    y_test = test_data.select('price')

    # Return the train and test sets
    return x_train, x_test, y_train, y_test

def linearRegression(X, y, num_iterations=100, learning_rate=0.01):
    # Add a column of ones to the feature matrix for the bias term
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # Initialize the weights
    num_features = len(apts.columns)
    weights = np.zeros(num_features)

    # Perform gradient descent
    for iteration in range(num_iterations):
        # Calculate the predicted values
        y_pred = np.dot(X, weights)

        # Calculate the error
        error = y_pred - y

        # Update the weights
        gradient = np.dot(X.T, error)
        weights -= learning_rate * gradient

    return weights


def evaluateModel(X_test, y_test, weights):
    # Add a column of ones to the feature matrix for the bias term
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

    # Calculate the predicted values
    y_pred = np.dot(X_test, weights)

    # Calculate the mean squared error
    mse = np.mean((y_pred - y_test) ** 2)

    return mse


if __name__ == '__main__':
    # --- Part A ---
    # spark, books = loadjson()
    # F_authors(books)
    # english_pages_amount(books)
    # spark.stop()

    # ---Part B---
    spark, apts = loadTextFile('prices.txt')
    print(apts.printSchema())
    x_train, x_test, y_train, y_test = splitData(apts, 0.75)
    x_train.show()
    x_test.show()
    y_train.show()
    y_test.show()

    weights = linearRegression(x_train, y_train)
    mse = evaluateModel(x_test, y_test, weights)
    print("Mean Squared Error:", mse)
    spark.stop()



