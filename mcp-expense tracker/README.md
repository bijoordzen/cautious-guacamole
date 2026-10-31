# 💰 MCP Expense Tracker

A full-featured personal expense tracker powered by:
- **FastMCP**: Model Context Protocol (MCP) server for seamless AI integration (Claude Desktop, Cursor, Antigravity, etc.).
- **SQLite**: Local, reliable, zero-config relational SQL database.
- **Streamlit**: Modern interactive dashboard with visual analytics, full natural language **AI Chat for CRUD**, expense management, SQL query console, and an MCP tool tester.

All files are self-contained within this directory.

---

## 📁 Project Structure

```
mcp-expense tracker/
├── app.py              # Streamlit Web UI & Dashboards (with AI Chat CRUD)
├── chat_assistant.py   # Natural Language / LLM CRUD Chat Assistant Engine
├── mcp_server.py       # FastMCP Server exposing MCP tools & resources
├── db.py               # SQLite database layer, schema, and queries
├── expenses.db         # SQLite database file (auto-created on first run)
├── requirements.txt    # Python dependencies
└── README.md           # Documentation and setup instructions
```

---

## 🚀 Quick Start

### 1. Launch the Streamlit Web Application

Run Streamlit:

```bash
streamlit run "mcp-expense tracker/app.py"
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 💬 Natural Language AI Chat & CRUD Features

Inside the app's **💬 AI Chat & CRUD** tab, you can perform all operations simply by typing in everyday English:

### 1. ➕ Create (Add Expense)
- `"Spent $45 on groceries today via UPI"`
- `"Add $15 for lunch with team"`
- `"Bought running shoes for $85 via Credit Card"`
- `"Paid 120 electricity bill"`

### 2. 📋 Read (Search & Summarize)
- `"Show my recent expenses"`
- `"How much did I spend on food?"`
- `"What is my spending on groceries?"`
- `"What was my highest expense?"`
- `"Show monthly summary"`

### 3. ✏️ Update
- `"Update expense 1 amount to 1300"`
- `"Change expense #3 category to Entertainment"`
- `"Update expense 2 description to Dinner with team"`

### 4. 🗑️ Delete
- `"Delete expense #4"`
- `"Remove expense 2"`

> **Zero Setup Needed**: By default, the chat runs on the built-in **Smart NLP Engine**, which executes immediately on your local machine with **zero API keys required**. If you want to use Google Gemini or OpenAI, you can optionally toggle the provider in the **Settings** expander.

---

## 🤖 Connecting to Claude Desktop / Cursor / Antigravity

Add the following to your MCP client configuration (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "expense-tracker": {
      "command": "C:\\Users\\Srinivas\\Documents\\projects\\cautious-guacamole\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\Srinivas\\Documents\\projects\\cautious-guacamole\\mcp-expense tracker\\mcp_server.py"
      ]
    }
  }
}
```

---

## 🛠️ Available MCP Tools

| Tool Name | Parameters | Description |
|-----------|------------|-------------|
| `add_expense` | `amount`, `category`, `description`, `expense_date`, `payment_method` | Inserts an expense record into SQLite |
| `get_expenses` | `limit`, `category`, `start_date`, `end_date`, `search` | Queries and filters expense records |
| `get_expense_summary` | `start_date`, `end_date` | Aggregated total, averages, and category breakdown |
| `update_expense` | `expense_id`, `amount`, `category`, `description`, `expense_date`, `payment_method` | Updates existing record fields |
| `delete_expense` | `expense_id` | Deletes an expense by ID |
| `query_database` | `sql_query` | Executes safe read-only SQL `SELECT` queries |
