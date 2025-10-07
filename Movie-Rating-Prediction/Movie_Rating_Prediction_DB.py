# 📌 Movie Rating Prediction with Database (SQLite)
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# =======================
# Step 1: Connect to Database
# =======================
conn = sqlite3.connect("movies.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Genre TEXT,
    Director TEXT,
    Actors TEXT,
    Rating REAL
)
""")

# =======================
# Step 2: Insert Sample Data
# =======================
sample_data = [
    ("Inception", "Sci-Fi", "Christopher Nolan", "Leonardo DiCaprio", 8.8),
    ("Titanic", "Romance", "James Cameron", "Leonardo DiCaprio", 7.8),
    ("The Dark Knight", "Action", "Christopher Nolan", "Christian Bale", 9.0),
    ("Avatar", "Sci-Fi", "James Cameron", "Sam Worthington", 7.9),
    ("Interstellar", "Sci-Fi", "Christopher Nolan", "Matthew McConaughey", 8.6)
]

cursor.executemany("INSERT INTO movies (Title, Genre, Director, Actors, Rating) VALUES (?, ?, ?, ?, ?)", sample_data)
conn.commit()

# =======================
# Step 3: Load Data from Database
# =======================
df = pd.read_sql_query("SELECT * FROM movies", conn)
print("Movies from Database:\n", df, "\n")

# =======================
# Step 4: Preprocessing
# =======================
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[['Genre', 'Director']]).toarray()

encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Genre','Director']))
df = pd.concat([df, encoded_df], axis=1)

# =======================
# Step 5: Train-Test Split
# =======================
X = df.drop(columns=['id', 'Title', 'Rating', 'Genre', 'Director', 'Actors'])
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# =======================
# Step 6: Train Model
# =======================
model = LinearRegression()
model.fit(X_train, y_train)

# =======================
# Step 7: Evaluate Model
# =======================
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print("\nExample Predictions:")
for i in range(len(y_test)):
    print(f"Movie: {df['Title'].iloc[y_test.index[i]]} | Actual: {y_test.iloc[i]} | Predicted: {round(y_pred[i], 2)}")

# =======================
# Close Database
# =======================
conn.close()
