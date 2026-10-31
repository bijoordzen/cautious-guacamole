"""
Streamlit Web UI for the MCP Expense Tracker.
Connects with SQLite database and provides interactive dashboards,
AI Chat for full CRUD, expense management, MCP tool explorer, and SQL query console.
"""

import sys
import os
import json
from datetime import datetime, date, timedelta
import pandas as pd
import streamlit as st
import altair as alt

# Ensure local imports work regardless of current working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import db
import mcp_server
import chat_assistant
import importlib
importlib.reload(chat_assistant)

# Initialize database
db.init_db()

# Streamlit Page Setup
st.set_page_config(
    page_title="MCP Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        color: #38bdf8;
        font-size: 0.8rem;
        margin-top: 2px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
    .chat-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        background-color: #0284c7;
        color: white;
        font-size: 0.75rem;
        font-weight: bold;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper to load dataframe ---
def load_expenses_df(category=None, start_date=None, end_date=None, search=None) -> pd.DataFrame:
    records = db.get_expenses(
        category=category,
        start_date=start_date,
        end_date=end_date,
        search=search
    )
    if not records:
        return pd.DataFrame(columns=["id", "date", "category", "amount", "description", "payment_method", "created_at"])
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/budget.png", width=64)
    st.title("Expense Tracker")
    st.caption("Powered by **FastMCP** & **SQLite**")
    st.divider()

    st.subheader("Filter Dashboard")
    date_preset = st.selectbox(
        "Date Range Preset",
        ["This Month", "Last 30 Days", "This Year", "All Time", "Custom"]
    )

    today = date.today()
    if date_preset == "This Month":
        start_filter = today.replace(day=1)
        end_filter = today
    elif date_preset == "Last 30 Days":
        start_filter = today - timedelta(days=30)
        end_filter = today
    elif date_preset == "This Year":
        start_filter = today.replace(month=1, day=1)
        end_filter = today
    elif date_preset == "All Time":
        start_filter = None
        end_filter = None
    else:  # Custom
        col_s, col_e = st.columns(2)
        with col_s:
            start_filter = st.date_input("From", today - timedelta(days=30))
        with col_e:
            end_filter = st.date_input("To", today)

    cat_options = ["All"] + db.CATEGORIES
    category_filter = st.selectbox("Category", cat_options)
    search_query = st.text_input("Search description", placeholder="e.g. coffee, uber...")

    st.divider()
    st.info(
        "💡 **Natural Language CRUD**\n\n"
        "Go to the **AI Chat & CRUD** tab to add, view, update, and delete expenses using everyday plain English!"
    )
    st.caption(f"Database: `{os.path.basename(db.DB_PATH)}`")


# Convert date objects to ISO strings for DB query
start_str = start_filter.isoformat() if start_filter else None
end_str = end_filter.isoformat() if end_filter else None
cat_arg = None if category_filter == "All" else category_filter

df_filtered = load_expenses_df(
    category=cat_arg,
    start_date=start_str,
    end_date=end_str,
    search=search_query if search_query.strip() else None
)

# Fetch general statistics
stats = db.get_overall_stats()


# --- TOP KPI METRICS ---
st.markdown("## 💰 Expense Overview")

m1, m2, m3, m4 = st.columns(4)

total_current_month = stats.get("current_month_spent", 0.0)
filtered_total = df_filtered["amount"].sum() if not df_filtered.empty else 0.0
filtered_count = len(df_filtered)
avg_transaction = (filtered_total / filtered_count) if filtered_count > 0 else 0.0
top_cat = (
    df_filtered.groupby("category")["amount"].sum().idxmax()
    if not df_filtered.empty else "N/A"
)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Selected Period Total</div>
        <div class="metric-value">${filtered_total:,.2f}</div>
        <div class="metric-sub">{filtered_count} transactions recorded</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Current Month Spent</div>
        <div class="metric-value">${total_current_month:,.2f}</div>
        <div class="metric-sub">{stats.get('current_month_count', 0)} items in {stats.get('current_month', '')}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Transaction</div>
        <div class="metric-value">${avg_transaction:,.2f}</div>
        <div class="metric-sub">Max: ${df_filtered['amount'].max() if not df_filtered.empty else 0:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Top Category</div>
        <div class="metric-value" style="font-size: 1.4rem;">{top_cat}</div>
        <div class="metric-sub">Lifetime Total: ${stats.get('total_spent', 0):,.2f}</div>
    </div>
    """, unsafe_allow_html=True)


# --- MAIN TABS ---
tab_chat, tab_dash, tab_add, tab_manage, tab_mcp, tab_sql = st.tabs([
    "💬 AI Chat & CRUD",
    "📊 Analytics & Trends",
    "➕ Add Expense",
    "📋 Manage & Export",
    "🤖 FastMCP Server & Assistant",
    "🔍 SQL Console"
])


# ==============================================================================
# TAB 1: AI CHAT & CRUD
# ==============================================================================
with tab_chat:
    st.subheader("💬 Chat & Manage Expenses in Natural Language")
    st.caption("You can Create, Read, Update, and Delete expenses by talking directly to your assistant.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hi there! I'm your **Expense Tracker AI Assistant**.\n\n"
                    "You can talk to me naturally to manage your finances. Here are a few things you can ask:\n"
                    "- ➕ **Create**: `Spent $45 on groceries today via Credit Card`\n"
                    "- ➕ **Create**: `Add $15 for lunch with team via UPI`\n"
                    "- 📋 **Read**: `Show my recent expenses` or `How much on food?`\n"
                    "- ✏️ **Update**: `Update expense 1 amount to 1300`\n"
                    "- 🗑️ **Delete**: `Delete expense #2`\n"
                    "- 📊 **Summary**: `What was my highest expense?`"
                )
            }
        ]

    # Model settings accordion
    with st.expander("⚙️ Assistant & Engine Settings", expanded=False):
        c_mode1, c_mode2 = st.columns([1, 2])
        with c_mode1:
            chat_engine = st.selectbox(
                "Engine Mode",
                ["Smart NLP (No API Key Required)", "Google Gemini (gemini-2.5-flash)", "OpenAI (gpt-4o-mini)"]
            )
        with c_mode2:
            if "Gemini" in chat_engine:
                api_key_input = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
            elif "OpenAI" in chat_engine:
                api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
            else:
                api_key_input = ""
                st.info("Using built-in natural language parser. Fast, private, and works without any external API keys!")

    # Quick prompt buttons
    st.markdown("**Quick Prompts:**")
    q_cols = st.columns(5)
    selected_prompt = None

    if q_cols[0].button("🛒 Spent $42 on groceries", use_container_width=True):
        selected_prompt = "Spent $42 on groceries via UPI"
    if q_cols[1].button("☕ Add $5 for morning coffee", use_container_width=True):
        selected_prompt = "Add $5 for morning coffee"
    if q_cols[2].button("📋 Show recent expenses", use_container_width=True):
        selected_prompt = "Show my recent expenses"
    if q_cols[3].button("📊 How much on dining?", use_container_width=True):
        selected_prompt = "How much did I spend on food?"
    if q_cols[4].button("🏆 Top expenses", use_container_width=True):
        selected_prompt = "What was my highest expense?"

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input handler
    user_input = st.chat_input("Type your expense command or question (e.g. Spent $30 on dinner)...")
    prompt_to_run = selected_prompt or user_input

    if prompt_to_run:
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": prompt_to_run})
        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        # Process with assistant
        with st.chat_message("assistant"):
            with st.spinner("Processing request..."):
                if "Gemini" in chat_engine and api_key_input.strip():
                    result = chat_assistant.handle_llm_chat(prompt_to_run, api_key_input.strip(), provider="Gemini")
                elif "OpenAI" in chat_engine and api_key_input.strip():
                    result = chat_assistant.handle_llm_chat(prompt_to_run, api_key_input.strip(), provider="OpenAI")
                else:
                    result = chat_assistant.handle_smart_nlp_chat(prompt_to_run)

                reply_text = result.get("response", "Done.")
                st.markdown(reply_text)

        # Append assistant reply to session state
        st.session_state.messages.append({"role": "assistant", "content": reply_text})

        # If data was mutated, trigger a rerun so dashboards and tables update
        if result.get("tool_called") in ["add_expense", "update_expense", "delete_expense"] and result.get("success"):
            st.rerun()


# ==============================================================================
# TAB 2: ANALYTICS & TRENDS
# ==============================================================================
with tab_dash:
    if df_filtered.empty:
        st.warning("No expenses found for the selected filters. Try broadening your date range or add an expense!")
    else:
        col_chart1, col_chart2 = st.columns([1, 1])

        with col_chart1:
            st.subheader("Category Breakdown")
            cat_summary = (
                df_filtered.groupby("category", as_index=False)["amount"]
                .sum()
                .sort_values(by="amount", ascending=False)
            )

            donut = alt.Chart(cat_summary).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="amount", type="quantitative"),
                color=alt.Color(field="category", type="nominal", legend=alt.Legend(title="Category")),
                tooltip=[
                    alt.Tooltip("category", title="Category"),
                    alt.Tooltip("amount", title="Total Spent ($)", format=",.2f")
                ]
            ).properties(height=320)

            st.altair_chart(donut, use_container_width=True)

        with col_chart2:
            st.subheader("Payment Method Distribution")
            pm_summary = (
                df_filtered.groupby("payment_method", as_index=False)["amount"]
                .sum()
                .sort_values(by="amount", ascending=False)
            )

            bar = alt.Chart(pm_summary).mark_bar(cornerRadius=6).encode(
                x=alt.X("amount:Q", title="Total Amount ($)"),
                y=alt.Y("payment_method:N", sort="-x", title="Method"),
                color=alt.Color("payment_method:N", legend=None),
                tooltip=[
                    alt.Tooltip("payment_method", title="Method"),
                    alt.Tooltip("amount", title="Total ($)", format=",.2f")
                ]
            ).properties(height=320)

            st.altair_chart(bar, use_container_width=True)

        st.subheader("Daily Spending Timeline")
        daily_df = (
            df_filtered.groupby("date", as_index=False)["amount"]
            .sum()
            .sort_values(by="date")
        )

        timeline = alt.Chart(daily_df).mark_area(
            line={"color": "#38bdf8"},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#38bdf8", offset=0),
                    alt.GradientStop(color="rgba(56, 189, 248, 0.05)", offset=1)
                ],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("amount:Q", title="Total Spent ($)"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("amount:Q", title="Spent ($)", format=",.2f")
            ]
        ).properties(height=260)

        st.altair_chart(timeline, use_container_width=True)


# ==============================================================================
# TAB 3: ADD EXPENSE
# ==============================================================================
with tab_add:
    st.subheader("Record a New Expense")
    st.caption("Transactions are immediately committed to the SQLite database and accessible via FastMCP.")

    col_quick1, col_quick2 = st.columns([2, 1])

    with col_quick1:
        with st.form("add_expense_form", clear_on_submit=True):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                amount_input = st.number_input("Amount ($)", min_value=0.01, step=1.00, format="%.2f")
                category_input = st.selectbox("Category", db.CATEGORIES)
                payment_method_input = st.selectbox("Payment Method", db.PAYMENT_METHODS)

            with f_col2:
                date_input = st.date_input("Date", date.today())
                desc_input = st.text_input("Description / Notes", placeholder="e.g. Groceries at Trader Joe's")
                custom_category = st.text_input("Or enter Custom Category (optional)", placeholder="e.g. Pet Care")

            submitted = st.form_submit_button("💾 Save Expense", use_container_width=True)

            if submitted:
                chosen_category = custom_category.strip() if custom_category.strip() else category_input
                try:
                    new_rec = db.add_expense(
                        amount=amount_input,
                        category=chosen_category,
                        description=desc_input,
                        expense_date=date_input.isoformat(),
                        payment_method=payment_method_input
                    )
                    st.success(f"✅ Added: **${amount_input:.2f}** for '{desc_input or chosen_category}' on {date_input}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add expense: {e}")

    with col_quick2:
        st.markdown("#### ⚡ Quick Presets")
        st.write("Click to pre-fill common everyday purchases:")

        presets = [
            ("☕ Morning Coffee", 4.50, "Food & Dining", "UPI"),
            ("🍔 Lunch combo", 12.00, "Food & Dining", "Debit Card"),
            ("🚕 Cab ride", 18.50, "Transportation", "UPI"),
            ("🛒 Grocery essentials", 45.00, "Groceries", "Credit Card"),
            ("🎬 Movie ticket", 15.00, "Entertainment", "Credit Card"),
        ]

        for label, p_amt, p_cat, p_pm in presets:
            if st.button(f"{label} - ${p_amt:.2f}", key=f"quick_{label}", use_container_width=True):
                try:
                    db.add_expense(
                        amount=p_amt,
                        category=p_cat,
                        description=label.split(" ", 1)[1],
                        expense_date=date.today().isoformat(),
                        payment_method=p_pm
                    )
                    st.success(f"Added preset: {label} (${p_amt:.2f})")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")


# ==============================================================================
# TAB 4: MANAGE & EXPORT
# ==============================================================================
with tab_manage:
    st.subheader("Manage Expenses")
    st.caption("Review, edit, or delete existing transaction records.")

    all_expenses = db.get_expenses(limit=200)
    if not all_expenses:
        st.info("No records in database.")
    else:
        df_all = pd.DataFrame(all_expenses)

        # Export to CSV
        csv_data = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export All Expenses as CSV",
            data=csv_data,
            file_name=f"expenses_export_{date.today().isoformat()}.csv",
            mime="text/csv"
        )

        st.dataframe(
            df_all[["id", "date", "category", "amount", "description", "payment_method", "created_at"]],
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("Modify / Delete an Expense")
        del_col1, del_col2 = st.columns([2, 1])

        with del_col1:
            expense_ids = [row["id"] for row in all_expenses]
            selected_id = st.selectbox(
                "Select Expense to Manage",
                expense_ids,
                format_func=lambda x: f"ID #{x} - {next((r['date'] + ' | $' + str(r['amount']) + ' | ' + r['category'] + ' (' + str(r['description'] or '') + ')') for r in all_expenses if r['id'] == x)}"
            )

            current_item = db.get_expense_by_id(selected_id)
            if current_item:
                with st.expander("✏️ Edit Selected Expense", expanded=False):
                    e_amount = st.number_input("Amount", value=float(current_item["amount"]), step=1.0)
                    e_category = st.selectbox("Category", db.CATEGORIES, index=db.CATEGORIES.index(current_item["category"]) if current_item["category"] in db.CATEGORIES else 0)
                    e_desc = st.text_input("Description", value=current_item["description"] or "")
                    e_date = st.date_input("Date", datetime.strptime(current_item["date"], "%Y-%m-%d").date())
                    e_pm = st.selectbox("Payment Method", db.PAYMENT_METHODS, index=db.PAYMENT_METHODS.index(current_item["payment_method"]) if current_item["payment_method"] in db.PAYMENT_METHODS else 0)

                    if st.button("Update Record", key="btn_update"):
                        db.update_expense(
                            expense_id=selected_id,
                            amount=e_amount,
                            category=e_category,
                            description=e_desc,
                            expense_date=e_date.isoformat(),
                            payment_method=e_pm
                        )
                        st.success("Expense updated successfully!")
                        st.rerun()

        with del_col2:
            st.write("#### 🗑️ Delete Action")
            st.warning(f"Delete Expense #{selected_id}?")
            if st.button("Delete Permanently", type="primary", key="btn_delete"):
                if db.delete_expense(selected_id):
                    st.success(f"Expense #{selected_id} removed.")
                    st.rerun()
                else:
                    st.error("Failed to delete record.")


# ==============================================================================
# TAB 5: MCP SERVER & ASSISTANT
# ==============================================================================
with tab_mcp:
    st.subheader("🤖 FastMCP Model Context Protocol Integration")
    st.markdown("""
    The file `mcp_server.py` implements an MCP server using **FastMCP**. Any MCP-compatible LLM client 
    (such as Claude Desktop, Cursor, Antigravity, or custom agents) can read and write to this SQLite database.
    """)

    col_mcp_cfg, col_mcp_test = st.columns([1, 1])

    with col_mcp_cfg:
        st.markdown("### 🔌 Connecting an MCP Client")
        python_executable = sys.executable
        server_path = os.path.join(CURRENT_DIR, "mcp_server.py")

        config_example = {
            "mcpServers": {
                "expense-tracker": {
                    "command": python_executable,
                    "args": [server_path]
                }
            }
        }

        st.caption("Paste this into your Claude Desktop or MCP Client configuration:")
        st.code(json.dumps(config_example, indent=2), language="json")

        st.markdown("### 📦 Available MCP Tools")
        st.markdown("""
        - `add_expense(amount, category, description, expense_date, payment_method)`
        - `get_expenses(limit, category, start_date, end_date, search)`
        - `get_expense_summary(start_date, end_date)`
        - `update_expense(expense_id, amount, category, description, expense_date, payment_method)`
        - `delete_expense(expense_id)`
        - `query_database(sql_query)`
        """)

    with col_mcp_test:
        st.markdown("### 🧪 Live MCP Tool Simulator")
        st.caption("Directly test the FastMCP tool handlers against the SQLite database:")

        test_tool = st.selectbox(
            "Select MCP Tool to Invoke",
            ["get_expense_summary", "get_expenses", "query_database", "add_expense"]
        )

        if test_tool == "get_expense_summary":
            if st.button("Execute get_expense_summary()"):
                res = mcp_server.get_expense_summary()
                st.json(res)

        elif test_tool == "get_expenses":
            t_limit = st.slider("Limit", 1, 50, 5)
            t_cat = st.selectbox("Filter Category", ["All"] + db.CATEGORIES, key="mcp_test_cat")
            cat_param = None if t_cat == "All" else t_cat
            if st.button("Execute get_expenses()"):
                res = mcp_server.get_expenses(limit=t_limit, category=cat_param)
                st.json(res)

        elif test_tool == "query_database":
            sample_query = st.text_area(
                "Read-only SQL Query",
                "SELECT category, COUNT(*) as count, ROUND(SUM(amount), 2) as total FROM expenses GROUP BY category"
            )
            if st.button("Execute query_database()"):
                res = mcp_server.query_database(sample_query)
                st.json(res)

        elif test_tool == "add_expense":
            with st.form("mcp_test_add"):
                m_amt = st.number_input("Amount", value=19.99)
                m_cat = st.selectbox("Category", db.CATEGORIES, key="mcp_add_cat")
                m_desc = st.text_input("Description", value="MCP automated test expense")
                m_sub = st.form_submit_button("Simulate Tool Call")
                if m_sub:
                    res = mcp_server.add_expense(amount=m_amt, category=m_cat, description=m_desc)
                    st.json(res)
                    st.rerun()


# ==============================================================================
# TAB 6: SQL CONSOLE
# ==============================================================================
with tab_sql:
    st.subheader("🔍 SQLite Query Console")
    st.caption("Execute custom read-only SQL queries directly against `expenses.db`.")

    col_q1, col_q2 = st.columns([3, 1])

    with col_q2:
        st.markdown("#### Sample Queries")
        preset_q = st.radio(
            "Templates",
            [
                "Category Totals",
                "Top 5 Highest Expenses",
                "Payment Method Totals",
                "Monthly Aggregations"
            ]
        )

        templates = {
            "Category Totals": "SELECT category, COUNT(*) AS count, ROUND(SUM(amount), 2) AS total_spent FROM expenses GROUP BY category ORDER BY total_spent DESC;",
            "Top 5 Highest Expenses": "SELECT id, date, category, amount, description, payment_method FROM expenses ORDER BY amount DESC LIMIT 5;",
            "Payment Method Totals": "SELECT payment_method, COUNT(*) AS transactions, ROUND(SUM(amount), 2) AS total_amount FROM expenses GROUP BY payment_method ORDER BY total_amount DESC;",
            "Monthly Aggregations": "SELECT strftime('%Y-%m', date) AS month, COUNT(*) AS count, ROUND(SUM(amount), 2) AS total FROM expenses GROUP BY month ORDER BY month DESC;"
        }

    with col_q1:
        query_text = st.text_area("SQL Statement", value=templates[preset_q], height=120)
        run_query_btn = st.button("▶️ Run SQL Query", type="primary")

        if run_query_btn:
            try:
                results = db.execute_custom_sql(query_text)
                if results:
                    st.success(f"Query returned {len(results)} rows.")
                    res_df = pd.DataFrame(results)
                    st.dataframe(res_df, use_container_width=True)
                else:
                    st.info("Query executed successfully. 0 rows returned.")
            except Exception as e:
                st.error(f"SQL Error: {e}")
