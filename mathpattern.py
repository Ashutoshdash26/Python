# Importing all modules
import math
import re
import json
import datetime
import sqlite3
import os
import random
import statistics
import requests
import http.server
import urllib.parse

# 1. math → calculate square root
num = 25
print("Square root:", math.sqrt(num))

# 2. re → find pattern
text = "My phone number is 9876543210"
match = re.search(r'\d{10}', text)
print("Found number:", match.group())

# 3. json → convert dict to JSON
data = {"name": "Ashutosh", "age": 22}
json_data = json.dumps(data)
print("JSON:", json_data)

# 4. datetime → current date & time
now = datetime.datetime.now()
print("Current date & time:", now)

# 5. sqlite3 → create DB & insert data
conn = sqlite3.connect("test.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS users (name TEXT, age INT)")
cursor.execute("INSERT INTO users VALUES ('Ashutosh', 22)")
conn.commit()
print("Database updated")

# 6. os → current directory
print("Current directory:", os.getcwd())

# 7. random → random number
rand_num = random.randint(1, 100)
print("Random number:", rand_num)

# 8. statistics → mean
nums = [10, 20, 30, 40, 50]
print("Mean:", statistics.mean(nums))

# 9. requests → API call
response = requests.get("https://jsonplaceholder.typicode.com/todos/1")
print("API Response:", response.json())

# 10. urllib → parse URL
url = "https://www.example.com/?name=Ashutosh"
parsed = urllib.parse.urlparse(url)
print("Parsed URL:", parsed)

# 11. http → simple server (runs on localhost)
print("Starting server at http://localhost:8000")
server = http.server.HTTPServer(("localhost", 8000), http.server.SimpleHTTPRequestHandler)

# Run server (Press Ctrl+C to stop)
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("Server stopped")
    server.server_close()

# Close database connection
conn.close()