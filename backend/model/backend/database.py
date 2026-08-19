import sqlite3
from pathlib import Path
from datetime import datetime

DB = Path("database.db")


def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_prediction(filename, prediction, confidence):
    with sqlite3.connect(DB) as con:
        con.execute(
            """
            INSERT INTO predictions
            (filename, prediction, confidence, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                filename,
                prediction,
                confidence,
                datetime.now().isoformat(timespec="seconds")
            )
        )


def get_predictions():
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row

        rows = con.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]