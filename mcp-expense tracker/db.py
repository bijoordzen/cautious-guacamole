"""
SQLite Database layer for the Expense Tracker.
Manages database initialization, schema, and CRUD operations.
"""

import os
import sqlite3
from datetime import datetime, date
from typing import List, Dict, Any, Optional

# Ensure database file is placed in this folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "expenses.db")

CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Transportation",
    "Shopping",
    "Entertainment",
    "Bills & Utilities",
    "Healthcare",
    "Education",
    "Travel",
    "Investments",
    "Other"
]

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking", "Other"]


def get_connection() -> sqlite3.Connection:
    """Creates a database connection with dict-like row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(seed_sample_data: bool = True):
    """Initializes tables in the SQLite database and seeds initial demo data if empty."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Expenses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0),
                description TEXT,
                payment_method TEXT DEFAULT 'Cash',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Categories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );
        """)

        # Budgets table (monthly budget per category)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT UNIQUE NOT NULL,
                monthly_limit REAL NOT NULL
            );
        """)

        # Populate default categories
        for cat in CATEGORIES:
            cursor.execute("INSERT OR IGNORE INTO categories (name) VALUES (?)", (cat,))

        # Check if expenses is empty, seed samples if requested
        cursor.execute("SELECT COUNT(*) FROM expenses")
        count = cursor.fetchone()[0]
        if count == 0 and seed_sample_data:
            sample_expenses = [
                ("2026-10-01", "Bills & Utilities", 1200.0, "Electricity bill", "Net Banking"),
                ("2026-10-03", "Groceries", 2450.5, "Supermarket weekly essentials", "Credit Card"),
                ("2026-10-05", "Transportation", 350.0, "Metro card recharge", "UPI"),
                ("2026-10-08", "Food & Dining", 680.0, "Dinner with friends", "UPI"),
                ("2026-10-12", "Entertainment", 499.0, "Streaming subscription", "Credit Card"),
                ("2026-10-15", "Shopping", 1899.0, "Running shoes sale", "Debit Card"),
                ("2026-10-18", "Healthcare", 450.0, "Pharmacy medicines", "UPI"),
                ("2026-10-22", "Food & Dining", 220.0, "Café coffee & snacks", "Cash"),
                ("2026-10-25", "Groceries", 1540.0, "Fresh vegetables and fruits", "UPI"),
                ("2026-10-28", "Transportation", 420.0, "Cab ride", "UPI"),
                ("2026-10-30", "Food & Dining", 850.0, "Weekend family dinner", "Credit Card"),
            ]
            cursor.executemany("""
                INSERT INTO expenses (date, category, amount, description, payment_method)
                VALUES (?, ?, ?, ?, ?)
            """, sample_expenses)

        conn.commit()


def add_expense(
    amount: float,
    category: str,
    description: str = "",
    expense_date: Optional[str] = None,
    payment_method: str = "Cash"
) -> Dict[str, Any]:
    """Adds a new expense record."""
    if not expense_date:
        expense_date = date.today().isoformat()
    if amount <= 0:
        raise ValueError("Amount must be greater than zero.")
    if not category:
        category = "Other"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO expenses (date, category, amount, description, payment_method)
            VALUES (?, ?, ?, ?, ?)
        """, (expense_date, category.strip(), round(float(amount), 2), description.strip(), payment_method.strip()))
        expense_id = cursor.lastrowid
        conn.commit()

        cursor.execute("SELECT * FROM expenses WHERE id = ?", (expense_id,))
        row = cursor.fetchone()
        return dict(row)


def get_expenses(
    limit: Optional[int] = None,
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
    order_by: str = "date DESC, id DESC"
) -> List[Dict[str, Any]]:
    """Retrieves expenses matching filter criteria."""
    query = "SELECT * FROM expenses WHERE 1=1"
    params: List[Any] = []

    if category and category != "All":
        query += " AND category = ?"
        params.append(category)

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    if search:
        query += " AND (description LIKE ? OR category LIKE ?)"
        term = f"%{search.strip()}%"
        params.extend([term, term])

    # Basic sanitization of order_by
    allowed_order = {
        "date DESC, id DESC", "date ASC, id ASC",
        "amount DESC", "amount ASC", "category ASC"
    }
    if order_by not in allowed_order:
        order_by = "date DESC, id DESC"

    query += f" ORDER BY {order_by}"

    if limit and limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_expense_by_id(expense_id: int) -> Optional[Dict[str, Any]]:
    """Fetches a single expense by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM expenses WHERE id = ?", (expense_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_expense(
    expense_id: int,
    amount: Optional[float] = None,
    category: Optional[str] = None,
    description: Optional[str] = None,
    expense_date: Optional[str] = None,
    payment_method: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Updates fields of an existing expense."""
    current = get_expense_by_id(expense_id)
    if not current:
        return None

    new_amount = float(amount) if amount is not None else current["amount"]
    new_category = category if category is not None else current["category"]
    new_desc = description if description is not None else current["description"]
    new_date = expense_date if expense_date is not None else current["date"]
    new_pm = payment_method if payment_method is not None else current["payment_method"]

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE expenses
            SET amount = ?, category = ?, description = ?, date = ?, payment_method = ?
            WHERE id = ?
        """, (round(new_amount, 2), new_category, new_desc, new_date, new_pm, expense_id))
        conn.commit()

        cursor.execute("SELECT * FROM expenses WHERE id = ?", (expense_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_expense(expense_id: int) -> bool:
    """Deletes an expense by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_category_summary(start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns total spent and count grouped by category."""
    query = """
        SELECT 
            category, 
            ROUND(SUM(amount), 2) as total_amount, 
            COUNT(*) as transaction_count,
            ROUND(AVG(amount), 2) as avg_amount
        FROM expenses
        WHERE 1=1
    """
    params = []
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " GROUP BY category ORDER BY total_amount DESC"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_monthly_summary() -> List[Dict[str, Any]]:
    """Returns monthly spending aggregations."""
    query = """
        SELECT 
            strftime('%Y-%m', date) as month,
            ROUND(SUM(amount), 2) as total_amount,
            COUNT(*) as transaction_count
        FROM expenses
        GROUP BY month
        ORDER BY month DESC
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]


def get_overall_stats() -> Dict[str, Any]:
    """Returns high-level statistics for dashboard KPI cards."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_count,
                COALESCE(ROUND(SUM(amount), 2), 0.0) as total_spent,
                COALESCE(ROUND(AVG(amount), 2), 0.0) as avg_spent,
                COALESCE(MAX(amount), 0.0) as max_spent,
                MIN(date) as first_date,
                MAX(date) as last_date
            FROM expenses
        """)
        stats = dict(cursor.fetchone())

        # Current month stats
        current_month = date.today().strftime("%Y-%m")
        cursor.execute("""
            SELECT 
                COALESCE(ROUND(SUM(amount), 2), 0.0) as current_month_spent,
                COUNT(*) as current_month_count
            FROM expenses
            WHERE strftime('%Y-%m', date) = ?
        """, (current_month,))
        month_stats = dict(cursor.fetchone())

        stats.update(month_stats)
        stats["current_month"] = current_month
        return stats


def execute_custom_sql(sql_query: str) -> List[Dict[str, Any]]:
    """
    Executes a SELECT query safely against the expenses SQLite database.
    Prevents destructive operations like DROP, DELETE, ALTER, UPDATE from arbitrary SQL tool calls.
    """
    trimmed = sql_query.strip().strip(";").strip()
    first_word = trimmed.split()[0].upper() if trimmed else ""

    if first_word != "SELECT":
        raise ValueError("Only read-only SELECT queries are permitted via the custom query tool.")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
