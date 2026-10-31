"""
FastMCP Server for Personal Expense Tracker.
Exposes tools and resources to MCP clients (e.g. Claude Desktop, Cursor, Antigravity)
to interact with the SQLite expense database.
"""

import sys
import os
import json
from datetime import date
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP

# Ensure local imports work regardless of execution directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import db

# Initialize database schema if not already initialized
db.init_db()

# Create FastMCP instance
mcp = FastMCP(
    name="ExpenseTracker",
    instructions=(
        "Personal Expense Tracker MCP Server backed by SQLite. "
        "Allows adding, searching, analyzing, and managing personal expenses and financial transactions."
    )
)


@mcp.tool()
def add_expense(
    amount: float,
    category: str,
    description: str = "",
    expense_date: Optional[str] = None,
    payment_method: str = "Cash"
) -> Dict[str, Any]:
    """
    Record a new expense into the SQLite database.

    Args:
        amount: The expense amount (must be positive number, e.g. 45.50).
        category: Expense category (e.g. 'Food & Dining', 'Groceries', 'Transportation', 'Shopping', 'Entertainment', 'Bills & Utilities', 'Healthcare', 'Education', 'Travel', 'Other').
        description: Note or description of the purchase (e.g. 'Coffee and sandwich with client').
        expense_date: Date in YYYY-MM-DD format. Defaults to today's date if omitted.
        payment_method: Payment method used (e.g. 'UPI', 'Credit Card', 'Debit Card', 'Cash', 'Net Banking'). Defaults to 'Cash'.

    Returns:
        The newly created expense record including assigned ID.
    """
    if not expense_date:
        expense_date = date.today().isoformat()
    created = db.add_expense(
        amount=amount,
        category=category,
        description=description,
        expense_date=expense_date,
        payment_method=payment_method
    )
    return {
        "status": "success",
        "message": f"Expense of ${amount:.2f} in '{category}' added successfully.",
        "expense": created
    }


@mcp.tool()
def get_expenses(
    limit: int = 50,
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None
) -> Dict[str, Any]:
    """
    List or search expenses from the SQLite database with optional filters.

    Args:
        limit: Maximum number of records to return (default 50).
        category: Filter by specific category (e.g. 'Groceries').
        start_date: Filter expenses on or after this date (YYYY-MM-DD).
        end_date: Filter expenses on or before this date (YYYY-MM-DD).
        search: Text search inside description or category name.

    Returns:
        List of matching expense records and total count.
    """
    results = db.get_expenses(
        limit=limit,
        category=category,
        start_date=start_date,
        end_date=end_date,
        search=search
    )
    total_amount = sum(row["amount"] for row in results)
    return {
        "count": len(results),
        "total_amount": round(total_amount, 2),
        "expenses": results
    }


@mcp.tool()
def get_expense_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get spending summary and category breakdown for a specific date range or all time.

    Args:
        start_date: Starting date (YYYY-MM-DD, inclusive).
        end_date: Ending date (YYYY-MM-DD, inclusive).

    Returns:
        Aggregated spending metrics including total spent, transaction count, average per transaction,
        and category breakdown sorted by highest spending.
    """
    categories = db.get_category_summary(start_date=start_date, end_date=end_date)
    total_spent = sum(item["total_amount"] for item in categories)
    total_transactions = sum(item["transaction_count"] for item in categories)
    avg_per_transaction = round(total_spent / total_transactions, 2) if total_transactions > 0 else 0.0

    return {
        "period": {
            "start_date": start_date or "earliest",
            "end_date": end_date or "latest"
        },
        "total_spent": round(total_spent, 2),
        "total_transactions": total_transactions,
        "average_per_transaction": avg_per_transaction,
        "by_category": categories
    }


@mcp.tool()
def update_expense(
    expense_id: int,
    amount: Optional[float] = None,
    category: Optional[str] = None,
    description: Optional[str] = None,
    expense_date: Optional[str] = None,
    payment_method: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update an existing expense record by ID.

    Args:
        expense_id: The ID of the expense to update.
        amount: New amount (optional).
        category: New category (optional).
        description: New description (optional).
        expense_date: New date in YYYY-MM-DD (optional).
        payment_method: New payment method (optional).

    Returns:
        Updated record or error status.
    """
    updated = db.update_expense(
        expense_id=expense_id,
        amount=amount,
        category=category,
        description=description,
        expense_date=expense_date,
        payment_method=payment_method
    )
    if not updated:
        return {"status": "error", "message": f"Expense with ID {expense_id} not found."}
    return {
        "status": "success",
        "message": f"Expense {expense_id} updated successfully.",
        "expense": updated
    }


@mcp.tool()
def delete_expense(expense_id: int) -> Dict[str, Any]:
    """
    Delete an expense record from the SQLite database.

    Args:
        expense_id: The ID of the expense record to delete.

    Returns:
        Status indicating whether the expense was deleted.
    """
    success = db.delete_expense(expense_id)
    if success:
        return {"status": "success", "message": f"Expense {expense_id} deleted successfully."}
    return {"status": "error", "message": f"Expense with ID {expense_id} not found."}


@mcp.tool()
def query_database(sql_query: str) -> Dict[str, Any]:
    """
    Execute a read-only SQL query against the expenses SQLite database.
    Useful for custom aggregations, date slicing, or complex financial analytics.

    Available table schema:
    - expenses (id INTEGER, date TEXT, category TEXT, amount REAL, description TEXT, payment_method TEXT, created_at TIMESTAMP)

    Args:
        sql_query: A SQL SELECT statement (e.g. 'SELECT category, SUM(amount) FROM expenses GROUP BY category').

    Returns:
        Rows matching the query.
    """
    try:
        rows = db.execute_custom_sql(sql_query)
        return {
            "status": "success",
            "row_count": len(rows),
            "results": rows
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.resource("expenses://recent")
def recent_expenses_resource() -> str:
    """Provides a quick JSON view of the latest 15 expenses recorded."""
    records = db.get_expenses(limit=15)
    return json.dumps(records, indent=2)


@mcp.resource("expenses://summary")
def monthly_summary_resource() -> str:
    """Provides spending summary by category."""
    summary = db.get_category_summary()
    return json.dumps(summary, indent=2)


@mcp.prompt()
def analyze_finances() -> str:
    """Prompt template for AI agents to analyze expense habits and recommend optimizations."""
    stats = db.get_overall_stats()
    cats = db.get_category_summary()
    return (
        f"You are a personal financial advisor. Review this user's current spending:\n"
        f"Total Spent: ${stats.get('total_spent', 0)}\n"
        f"Transactions: {stats.get('total_count', 0)}\n"
        f"Category Breakdown: {json.dumps(cats)}\n\n"
        "Please provide:\n"
        "1. Key observations on the biggest spending categories.\n"
        "2. Actionable suggestions to optimize monthly expenses.\n"
        "3. Recommended category limits for budgeting."
    )


if __name__ == "__main__":
    # Run the FastMCP server via standard IO
    mcp.run()
