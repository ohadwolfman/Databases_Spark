import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, current_date, col, avg
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


def splitData(apts):
    # Split the data into features and target
    X = apts[:,:-1]
    y = apts[:-1]

    # Split the data to 75& train and 25% test
    train_size = int(0.75 * len(apts))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test


def predictPrices(x_train, x_test, y_train,y_test):
    # Calculate the regression coefficients using the normal equation
    coefficients = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train

    # Predict the target variable for the test set
    y_pred = x_test @ coefficients

    # Calculate the mean squared error
    mse = np.mean((y_pred - y_test) ** 2)
    # Print the mean squared error
    print("Mean Squared Error:", mse)

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Add bias term to X
        X = np.insert(X, 0, 1, axis=1)

        # Compute the weights using the normal equation
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

        # Extract the bias term
        self.bias = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        # Add bias term to X
        X = np.insert(X, 0, 1, axis=1)

        # Predict the target variable
        y_pred = X @ np.concatenate(([self.bias], self.weights))

        return y_pred


if __name__ == '__main__':
    # --- Part A ---
    # spark, books = loadjson()
    # F_authors(books)
    # english_pages_amount(books)
    # spark.stop()

    # ---Part B---
    spark, apts = loadTextFile('prices.txt')
    print(apts.printSchema())
    # print("ok")
    x_train, x_test, y_train,y_test = splitData(apts)
    # print("ok")
    # predictPrices(x_train, x_test, y_train,y_test)
    # spark.stop()
    # Create an instance of the linear regression model

    model = LinearRegression()

    # Train the model
    model.fit(x_train, y_train)

    # Predict on the test set
    y_pred = model.predict(x_test)

    # Calculate mean squared error (MSE)
    mse = np.mean((y_test - y_pred) ** 2)

    # Print the MSE
    print("Mean Squared Error (MSE):", mse)


