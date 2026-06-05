"""
database.py
SQLite database — stores user info and face embeddings.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/biometric.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables on first run."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            student_id  TEXT UNIQUE NOT NULL,
            role        TEXT DEFAULT 'student',
            department  TEXT,
            registered_at TEXT NOT NULL,
            last_seen   TEXT,
            access_count INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            embedding   TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            student_id  TEXT,
            name        TEXT,
            status      TEXT NOT NULL,
            confidence  REAL,
            timestamp   TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized.")


def register_user(name: str, student_id: str, role: str,
                  department: str, embedding: list) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    try:
        cursor.execute("""
            INSERT INTO users (name, student_id, role, department, registered_at)
            VALUES (?, ?, ?, ?, ?)
        """, (name, student_id, role, department, now))
        user_id = cursor.lastrowid

        cursor.execute("""
            INSERT INTO face_embeddings (user_id, embedding, created_at)
            VALUES (?, ?, ?)
        """, (user_id, json.dumps(embedding), now))

        conn.commit()
        return {"success": True, "user_id": user_id, "message": f"{name} registered successfully"}
    except sqlite3.IntegrityError:
        return {"success": False, "message": f"Student ID {student_id} already registered"}
    finally:
        conn.close()


def get_all_embeddings() -> list:
    """Load all embeddings for comparison during authentication."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.id, u.name, u.student_id, u.role, u.department,
               fe.embedding
        FROM users u
        JOIN face_embeddings fe ON u.id = fe.user_id
    """)
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        result.append({
            "user_id":    row["id"],
            "name":       row["name"],
            "student_id": row["student_id"],
            "role":       row["role"],
            "department": row["department"],
            "embedding":  json.loads(row["embedding"]),
        })
    return result


def log_access(user_id, student_id, name, status, confidence):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO access_logs (user_id, student_id, name, status, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, student_id, name, status, confidence, now))
    if user_id and status == "GRANTED":
        cursor.execute("""
            UPDATE users SET last_seen=?, access_count=access_count+1
            WHERE id=?
        """, (now, user_id))
    conn.commit()
    conn.close()


def get_all_users() -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, student_id, role, department,
               registered_at, last_seen, access_count
        FROM users ORDER BY registered_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_access_logs(limit=50) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM access_logs
        ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_user(student_id: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE student_id=?", (student_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {"success": False, "message": "User not found"}
    user_id = row["id"]
    cursor.execute("DELETE FROM face_embeddings WHERE user_id=?", (user_id,))
    cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": f"User {student_id} deleted"}


if __name__ == "__main__":
    init_db()
