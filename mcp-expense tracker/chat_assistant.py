"""
Chat Assistant Engine for Expense Tracker.
Supports:
1. Smart Rule-based / NLP Agent: Zero API key needed, handles natural language CRUD.
2. LLM Provider (Google Gemini / OpenAI): Uses function calling / MCP tool dispatch.
"""

import re
import json
import os
import sys
from datetime import date, timedelta
from typing import Dict, Any, Optional, Tuple, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import db
import mcp_server


CATEGORY_KEYWORDS = {
    "Food & Dining": [
        "food", "dining", "lunch", "dinner", "breakfast", "cafe", "coffee", "restaurant",
        "pizza", "burger", "starbucks", "snack", "eat", "eating", "meal", "swiggy", "zomato",
        "takeout", "brunch", "drinks", "tea"
    ],
    "Groceries": [
        "groceries", "grocery", "supermarket", "walmart", "vegetables", "fruits", "milk",
        "bread", "ration", "provisions", "market", "veggies", "blinkit", "zepto", "instamart"
    ],
    "Transportation": [
        "uber", "ola", "cab", "taxi", "metro", "bus", "train", "fuel", "petrol", "diesel",
        "gas", "parking", "flight", "auto", "commute", "toll"
    ],
    "Shopping": [
        "shopping", "clothes", "shoes", "amazon", "flipkart", "dress", "shirt", "pants",
        "electronics", "gadget", "myntra", "t-shirt", "watch"
    ],
    "Entertainment": [
        "movie", "cinema", "netflix", "spotify", "hotstar", "game", "gaming", "concert",
        "subscription", "theatre", "steam", "youtube"
    ],
    "Bills & Utilities": [
        "bill", "electricity", "water", "wifi", "internet", "recharge", "rent", "broadband",
        "gas bill", "utility", "phone bill", "maintenance"
    ],
    "Healthcare": [
        "doctor", "medicine", "pharmacy", "hospital", "clinic", "dental", "checkup", "pills",
        "meds", "health"
    ],
    "Education": [
        "course", "udemy", "book", "tuition", "exam", "college", "school", "training", "books", "class"
    ],
    "Travel": [
        "hotel", "trip", "vacation", "resort", "airbnb", "flight ticket", "tour", "travel"
    ],
    "Investments": [
        "stock", "mutual fund", "sip", "crypto", "shares", "gold", "deposit", "investment"
    ],
}

PAYMENT_KEYWORDS = {
    "UPI": ["upi", "gpay", "google pay", "phonepe", "paytm"],
    "Credit Card": ["credit card", "cc"],
    "Debit Card": ["debit card", "dc"],
    "Cash": ["cash"],
    "Net Banking": ["net banking", "netbanking", "bank transfer", "neft", "imps"],
}


def infer_category(text: str) -> str:
    """Infers the most appropriate category based on keywords."""
    lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', lower):
                return cat
    return "Other"


def infer_payment_method(text: str) -> str:
    """Infers payment method from message text."""
    lower = text.lower()
    for pm, keywords in PAYMENT_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', lower):
                return pm
    return "Cash"


def handle_smart_nlp_chat(user_msg: str) -> Dict[str, Any]:
    """
    Parses and handles CRUD natural language intents.
    """
    msg = user_msg.strip()
    lower = msg.lower()

    # -------------------------------------------------------------
    # 0. GREETINGS & CASUAL CONVERSATION
    # -------------------------------------------------------------
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "namaste", "sup", "yo"]
    if lower in greetings or lower.startswith("hi ") or lower.startswith("hello ") or lower.startswith("hey "):
        return {
            "response": (
                "👋 **Hello!** I'm your Expense Tracker Assistant.\n\n"
                "I'm ready to help you manage your money! You can:\n"
                "- ➕ **Add an expense**: *'Spent 2$ eating food'*, *'45 dinner via UPI'*, or *'Grocery 50'*\n"
                "- 📋 **View expenses**: *'Show my recent expenses'* or *'How much on food?'*\n"
                "- ✏️ **Edit an expense**: *'Update expense 2 to $35'*\n"
                "- 🗑️ **Delete an expense**: *'Delete expense #3'*\n\n"
                "What would you like to record?"
            ),
            "tool_called": "greeting",
            "success": True
        }

    # "Help" or "What can you do?"
    if lower in ["help", "what can you do", "features", "commands", "options"]:
        return {
            "response": (
                "🤖 **Here is what I can do for you:**\n\n"
                "1. **Add Expenses**: `Spent 2$ eating food`, `Spent $50 on groceries`, `Add 15 coffee`, `Cab ride 18.50 via UPI`\n"
                "2. **View & Search**: `Show expenses`, `How much on dining?`, `What is my total spent?`\n"
                "3. **Update Records**: `Update expense 1 amount to 1200`, `Change expense 2 category to Shopping`\n"
                "4. **Delete Records**: `Delete expense 3`, `Remove #4`\n"
                "5. **Analytics**: `What was my biggest expense?`, `Show monthly summary`"
            ),
            "tool_called": "help",
            "success": True
        }

    # Prompts without details
    if lower in ["add an expense", "add expense", "i want to add an expense", "record expense", "new expense"]:
        return {
            "response": (
                "Sure! Please tell me what you spent and on what.\n\n"
                "For example: *'Spent 2$ eating food'*, *'Spent $30 on dinner'*, or *'Add 50 for groceries via UPI'*."
            ),
            "tool_called": "prompt_add",
            "success": True
        }

    if lower in ["delete expense", "delete an expense", "remove an expense"]:
        return {
            "response": (
                "Which expense would you like to remove?\n\n"
                "Please specify the ID, for example: *'Delete expense #3'* or *'Delete 5'*."
            ),
            "tool_called": "prompt_delete",
            "success": True
        }

    if lower in ["update expense", "edit expense", "change expense"]:
        return {
            "response": (
                "Which expense would you like to update?\n\n"
                "For example: *'Update expense 2 amount to $45'* or *'Change expense 1 category to Groceries'*."
            ),
            "tool_called": "prompt_update",
            "success": True
        }

    # -------------------------------------------------------------
    # 1. DELETE EXPENSE
    # e.g., "delete expense 4", "remove expense #3", "delete id 5", "delete 5", "del 2"
    # -------------------------------------------------------------
    del_match = re.search(r'\b(?:delete|remove|drop|del|cancel)\s+(?:expense\s*)?(?:id\s*|#\s*)?(\d+)\b', lower)
    if del_match:
        exp_id = int(del_match.group(1))
        item = db.get_expense_by_id(exp_id)
        if not item:
            return {
                "response": f"⚠️ Expense with ID **#{exp_id}** was not found in your database.",
                "tool_called": "delete_expense",
                "success": False
            }
        db.delete_expense(exp_id)
        return {
            "response": f"🗑️ **Deleted Expense #{exp_id}**: ${item['amount']:.2f} for '{item['description'] or item['category']}' on {item['date']}.",
            "tool_called": "delete_expense",
            "data": item,
            "success": True
        }

    # -------------------------------------------------------------
    # 2. UPDATE EXPENSE
    # e.g. "update expense 2 to $35", "change expense #3 description to Dinner", "update expense 4 category to Food"
    # -------------------------------------------------------------
    update_match = re.search(r'\b(?:update|change|modify|edit|set)\s+(?:expense\s*)?(?:id\s*|#\s*)?(\d+)\b', lower)
    if update_match:
        exp_id = int(update_match.group(1))
        current = db.get_expense_by_id(exp_id)
        if not current:
            return {
                "response": f"⚠️ Expense with ID **#{exp_id}** was not found in your records.",
                "tool_called": "update_expense",
                "success": False
            }

        amt_match = re.search(r'(?:amount\s*(?:to|=)?|to\s*[\$₹]?)\s*[\$₹]?\s*(\d+(?:\.\d+)?)', lower)
        new_amount = float(amt_match.group(1)) if amt_match else None

        new_cat = None
        for cat in db.CATEGORIES:
            if cat.lower() in lower:
                new_cat = cat
                break

        desc_match = re.search(r'(?:desc|description|note)\s*(?:to|=)?\s*[\'"]?([^\'",]+)[\'"]?', lower)
        new_desc = desc_match.group(1).strip() if desc_match else None

        if new_amount is None and new_cat is None and new_desc is None:
            fallback_amt = re.search(r'to\s+[\$₹]?(\d+(?:\.\d+)?)', lower)
            if fallback_amt:
                new_amount = float(fallback_amt.group(1))

        if new_amount is None and new_cat is None and new_desc is None:
            return {
                "response": (
                    f"Found expense **#{exp_id}** (${current['amount']:.2f} - {current['category']}).\n"
                    "What would you like to update? For example: `update expense 2 amount to 45` or `update expense 2 category to Groceries`."
                ),
                "tool_called": "update_expense",
                "success": False
            }

        updated = db.update_expense(
            expense_id=exp_id,
            amount=new_amount,
            category=new_cat,
            description=new_desc
        )
        return {
            "response": (
                f"✅ **Updated Expense #{exp_id}** successfully!\n\n"
                f"- **Amount**: ${updated['amount']:.2f}\n"
                f"- **Category**: {updated['category']}\n"
                f"- **Description**: {updated['description']}\n"
                f"- **Date**: {updated['date']}"
            ),
            "tool_called": "update_expense",
            "data": updated,
            "success": True
        }

    # -------------------------------------------------------------
    # 3. CREATE / ADD EXPENSE (Checked before read queries if an amount is present!)
    # Handles:
    # "spent 2$ eating food. add it as an expense for today"
    # "spent $45 on groceries today via Credit Card"
    # "grocery 50", "2$ coffee", "pizza 20", "paid 120 bill"
    # -------------------------------------------------------------
    # Check if a number with optional currency prefix or suffix exists
    amt_match = re.search(r'[\$₹]?\s*(\d+(?:\.\d+)?)\s*(?:[\$₹]|dollars?|bucks?|rupees?|rs)?\b', lower)
    has_action_word = any(w in lower for w in [
        "spent", "spend", "paid", "pay", "bought", "buy", "add", "eating", "ate", "cost", "for", "on", "record"
    ])

    # If it has an amount, and either an action word or clearly isn't an explicit question:
    is_explicit_question = any(q in lower for q in ["how much", "what did i spend", "show", "list", "view", "what was", "highest", "summary"])

    if amt_match and (has_action_word or not is_explicit_question):
        try:
            amount = float(amt_match.group(1))
        except ValueError:
            amount = 0.0

        if amount > 0:
            # Clean description
            cleaned = re.sub(r'[\$₹]?\s*' + re.escape(amt_match.group(1)) + r'\s*(?:[\$₹]|dollars?|bucks?|rupees?|rs)?', '', lower)
            cleaned = re.sub(r'\b(?:add\s+it\s+as\s+an\s+expense\s*(?:for\s+today)?|as\s+an\s+expense|for\s+today|today|yesterday)\b', '', cleaned)
            cleaned = re.sub(r'^\s*(?:i\s+)?(?:spent|spend|paid|pay|bought|add|record)\s*(?:on|for)?\s*', '', cleaned)
            cleaned = re.sub(r'\b(?:via|by|with|using|in)\s+(?:upi|gpay|google pay|phonepe|paytm|credit card|debit card|cash|net banking|netbanking)\b', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'[\$₹]', '', cleaned)
            cleaned = cleaned.strip('. ,!')

            # Infer payment method
            pm = infer_payment_method(lower)

            # Infer category
            category = infer_category(cleaned if cleaned else lower)
            final_desc = cleaned.capitalize() if cleaned else f"{category} purchase"

            # Date calculation
            exp_date = date.today()
            if "yesterday" in lower:
                exp_date = date.today() - timedelta(days=1)

            new_rec = db.add_expense(
                amount=amount,
                category=category,
                description=final_desc,
                expense_date=exp_date.isoformat(),
                payment_method=pm
            )

            return {
                "response": (
                    f"✅ **Added New Expense #{new_rec['id']}**\n\n"
                    f"- **Amount**: ${amount:.2f}\n"
                    f"- **Category**: {category}\n"
                    f"- **Description**: {final_desc}\n"
                    f"- **Date**: {exp_date.isoformat()}\n"
                    f"- **Payment**: {pm}"
                ),
                "tool_called": "add_expense",
                "data": new_rec,
                "success": True
            }

    # -------------------------------------------------------------
    # 4. READ / SEARCH / SUMMARY QUERIES
    # -------------------------------------------------------------
    # Check category queries
    for cat in db.CATEGORIES:
        cat_words = [cat.lower()] + CATEGORY_KEYWORDS.get(cat, [])
        if any(re.search(r'\b' + re.escape(w) + r'\b', lower) for w in cat_words):
            if any(w in lower for w in ["how much", "spent", "spending", "total", "show", "list", "cost"]):
                expenses = db.get_expenses(category=cat, limit=10)
                cat_summary = db.get_category_summary()
                cat_data = next((c for c in cat_summary if c["category"] == cat), None)

                if cat_data:
                    items_list = "\n".join([f"- **${e['amount']:.2f}** on {e['date']}: {e['description'] or 'No description'}" for e in expenses[:5]])
                    return {
                        "response": (
                            f"📊 **Spending on {cat}**:\n\n"
                            f"- **Total Spent**: ${cat_data['total_amount']:,.2f}\n"
                            f"- **Transactions**: {cat_data['transaction_count']}\n"
                            f"- **Average**: ${cat_data['avg_amount']:,.2f}\n\n"
                            f"**Recent Transactions**:\n{items_list}"
                        ),
                        "tool_called": "get_expenses",
                        "data": expenses,
                        "success": True
                    }
                else:
                    return {
                        "response": f"📊 You currently have **$0.00** recorded under **{cat}**.",
                        "tool_called": "get_expenses",
                        "success": True
                    }

    # Highest expense
    if any(w in lower for w in ["highest", "biggest", "largest", "max", "top expense"]):
        top = db.execute_custom_sql("SELECT * FROM expenses ORDER BY amount DESC LIMIT 5")
        if top:
            lines = [f"- **#{e['id']}**: ${e['amount']:.2f} on {e['date']} ({e['category']}) - {e['description'] or 'No note'}" for e in top]
            return {
                "response": "🏆 **Your Highest Expenses**:\n\n" + "\n".join(lines),
                "tool_called": "query_database",
                "data": top,
                "success": True
            }

    # Overall Summary
    if any(w in lower for w in ["summary", "total", "balance", "stats", "overview", "overall", "how much have i spent", "how much did i spend"]):
        stats = db.get_overall_stats()
        cats = db.get_category_summary()
        top_cats = "\n".join([f"- **{c['category']}**: ${c['total_amount']:,.2f} ({c['transaction_count']} items)" for c in cats[:4]])
        return {
            "response": (
                f"📈 **Financial Summary Overview**\n\n"
                f"- **Current Month Spent**: ${stats.get('current_month_spent', 0):,.2f}\n"
                f"- **All-Time Total**: ${stats.get('total_spent', 0):,.2f}\n"
                f"- **Total Transactions**: {stats.get('total_count', 0)}\n"
                f"- **Average Ticket**: ${stats.get('avg_spent', 0):,.2f}\n\n"
                f"**Top Categories**:\n{top_cats}"
            ),
            "tool_called": "get_expense_summary",
            "data": stats,
            "success": True
        }

    # List expenses
    if any(w in lower for w in ["show", "list", "recent", "history", "records", "expenses", "transactions"]):
        expenses = db.get_expenses(limit=8)
        if not expenses:
            return {"response": "You don't have any expenses recorded yet.", "tool_called": "get_expenses", "success": True}

        lines = [f"- **#{e['id']}** | **${e['amount']:.2f}** | *{e['category']}* | {e['date']} - {e['description'] or 'No notes'} ({e['payment_method']})" for e in expenses]
        return {
            "response": "📋 **Recent Expenses**:\n\n" + "\n".join(lines),
            "tool_called": "get_expenses",
            "data": expenses,
            "success": True
        }

    # -------------------------------------------------------------
    # 5. FRIENDLY FALLBACK IF UNMATCHED
    # -------------------------------------------------------------
    return {
        "response": (
            "I didn't quite catch that. You can:\n\n"
            "- ➕ **Add an expense**: *'Spent 2$ eating food'*, *'Spent $25 on groceries'*, or *'40 lunch via UPI'*\n"
            "- 📋 **View expenses**: *'Show recent expenses'* or *'How much on food?'*\n"
            "- ✏️ **Update**: *'Update expense 2 to $35'*\n"
            "- 🗑️ **Delete**: *'Delete expense #3'*"
        ),
        "tool_called": "unknown",
        "success": False
    }


def handle_llm_chat(user_msg: str, api_key: str, provider: str = "Gemini") -> Dict[str, Any]:
    """
    Handles chat using an LLM with function calling / tools.
    Falls back to Smart NLP if an error occurs.
    """
    if provider == "Gemini":
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            tool_declarations = [
                mcp_server.add_expense,
                mcp_server.get_expenses,
                mcp_server.get_expense_summary,
                mcp_server.update_expense,
                mcp_server.delete_expense,
                mcp_server.query_database
            ]

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_msg,
                config=types.GenerateContentConfig(
                    tools=tool_declarations,
                    system_instruction=(
                        "You are an intelligent personal finance assistant for an Expense Tracker app. "
                        "You can manage expenses (add, list, update, delete, summary) using the provided tools. "
                        "Be helpful, concise, and confirm any actions taken."
                    ),
                    temperature=0.2
                )
            )

            function_calls = response.function_calls
            if function_calls:
                results_text = []
                for call in function_calls:
                    fname = call.name
                    fargs = dict(call.args) if call.args else {}
                    tool_func = getattr(mcp_server, fname, None)
                    if tool_func:
                        res = tool_func(**fargs)
                        results_text.append(f"Tool `{fname}` executed: {json.dumps(res)}")

                return {
                    "response": f"{response.text or ''}\n\n" + "\n".join(results_text),
                    "tool_called": function_calls[0].name,
                    "success": True
                }

            return {
                "response": response.text or "I processed your request.",
                "tool_called": "llm_chat",
                "success": True
            }

        except Exception as e:
            res = handle_smart_nlp_chat(user_msg)
            res["response"] = f"(Note: LLM error: {str(e)[:80]}... Falling back to local NLP)\n\n" + res["response"]
            return res

    elif provider == "OpenAI":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "add_expense",
                        "description": "Add an expense",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "amount": {"type": "number"},
                                "category": {"type": "string"},
                                "description": {"type": "string"},
                                "payment_method": {"type": "string"}
                            },
                            "required": ["amount", "category"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "delete_expense",
                        "description": "Delete an expense by ID",
                        "parameters": {
                            "type": "object",
                            "properties": {"expense_id": {"type": "integer"}},
                            "required": ["expense_id"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_expenses",
                        "description": "Get or search expenses",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "limit": {"type": "integer"}
                            }
                        }
                    }
                }
            ]
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expense tracker assistant. Use tools to perform CRUD."},
                    {"role": "user", "content": user_msg}
                ],
                tools=tools
            )
            msg_obj = resp.choices[0].message
            if msg_obj.tool_calls:
                call = msg_obj.tool_calls[0]
                fname = call.function.name
                fargs = json.loads(call.function.arguments)
                tool_func = getattr(mcp_server, fname, None)
                if tool_func:
                    res = tool_func(**fargs)
                    return {
                        "response": f"Tool `{fname}` executed: {json.dumps(res, indent=2)}",
                        "tool_called": fname,
                        "success": True
                    }
            return {
                "response": msg_obj.content or "Done.",
                "tool_called": "llm_chat",
                "success": True
            }
        except Exception as e:
            res = handle_smart_nlp_chat(user_msg)
            res["response"] = f"(Note: LLM error: {str(e)[:80]}... Falling back to local NLP)\n\n" + res["response"]
            return res

    return handle_smart_nlp_chat(user_msg)
