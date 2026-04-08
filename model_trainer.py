import pandas
import sqlite3
import os

def load_data_error(message):
    print(message)
    sys.exit(1)


filename = input("Enter the path to training data (.csv or db): ")

if not os.path.exists(filename):
    load_data_error(f"File not found: {filename}")
elif filename.endswith(".csv"):
    try:
        data_frame = pandas.read_csv(filename)
    except Exception as e:
        load_data_error(f"Error while reading data: {e}")
elif filename.endswith(".db"):
    try:
        connection = sqlite3.connect(filename)
        data_frame = pandas.read_sql_query("SELECT * FROM passangers", connection)
        connection.close()
    except Exception as e:
        load_data_error(f"Error while reading data: {e}")
else:
    load_data_error(f"Unsupported file type: {filename}. Expected .csv or .db (SQLite).")

print(f"Succesfully loaded {len(data_frame)} rows from {filename}:")

print(data_frame.head())

