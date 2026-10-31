"""Pydantic schemas for the AI Financial Copilot orchestration layer.

Defines typed intent enums, tool argument schemas, and planner decisions
to ensure strict validation between LLM output and backend service execution.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class ToolName(str, Enum):
    # STORE
    GET_STORE_PROFILE = "get_store_profile"
    GET_STORE_BUSINESS_TYPE = "get_store_business_type"
    GET_STORE_OPERATIONAL_STATUS = "get_store_operational_status"

    # CASHIERS
    COUNT_CASHIERS = "count_cashiers"
    LIST_CASHIERS = "list_cashiers"
    GET_CASHIER_ACTIVITY = "get_cashier_activity"
    GET_CASHIER_TRANSACTION_SUMMARY = "get_cashier_transaction_summary"

    # CATEGORIES
    LIST_INCOME_CATEGORIES = "list_income_categories"
    LIST_EXPENSE_CATEGORIES = "list_expense_categories"
    LIST_ALL_CATEGORIES = "list_all_categories"
    RANK_EXPENSE_CATEGORIES = "rank_expense_categories"
    RANK_INCOME_CATEGORIES = "rank_income_categories"

    # TRANSACTIONS
    GET_TRANSACTION_DETAILS = "get_transaction_details"
    LIST_TRANSACTIONS = "list_transactions"
    GET_CASHIER_TRANSACTIONS = "get_cashier_transactions"
    GET_PENDING_APPROVALS = "get_pending_approvals"
    COUNT_PENDING_APPROVALS = "count_pending_approvals"
    GET_INCOME_SUMMARY = "get_income_summary"
    GET_EXPENSE_SUMMARY = "get_expense_summary"
    GET_CASH_FLOW_SUMMARY = "get_cash_flow_summary"
    COMPARE_FINANCIAL_PERIODS = "compare_financial_periods"
    GET_LARGEST_EXPENSES = "get_largest_expenses"
    GET_TRANSACTION_COUNT = "get_transaction_count"

    # ANALYTICS
    GET_CATEGORY_BREAKDOWN = "get_category_breakdown"
    GET_SPENDING_TRENDS = "get_spending_trends"
    GET_FINANCIAL_INSIGHTS = "get_financial_insights"
    GET_AVAILABLE_FORECAST = "get_available_forecast"
    GET_PROFIT_ANALYSIS = "get_profit_analysis"

    # APPLICATION
    EXPLAIN_FEATURE = "explain_feature"
    EXPLAIN_NAVIGATION = "explain_navigation"
    EXPLAIN_OWNER_APPROVAL_WORKFLOW = "explain_owner_approval_workflow"
    EXPLAIN_CSV_IMPORT = "explain_csv_import"
    EXPLAIN_AI_CLASSIFICATION = "explain_ai_classification"

    # MUTATIONS & SPECIAL
    CREATE_TRANSACTION_PROPOSAL = "create_transaction_proposal"
    UNSUPPORTED_MUTATION = "unsupported_mutation"
    GENERAL_HELP = "general_help"
    CLARIFICATION_NEEDED = "clarification_needed"
    UNKNOWN = "unknown"


class DateRangeParams(BaseModel):
    start_date: Optional[str] = Field(None, description="ISO format YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="ISO format YYYY-MM-DD")
    period_label: Optional[str] = Field("Semua Periode", description="Descriptive label e.g. 'Bulan Ini', 'Hari Ini'")


class FilterParams(BaseModel):
    transaction_type: Optional[Literal["pemasukan", "pengeluaran", "all"]] = "all"
    approval_status: Optional[Literal["disetujui", "pending", "ditolak", "all"]] = "all"
    cashier_id: Optional[str] = None
    cashier_name: Optional[str] = None
    category_id: Optional[str] = None
    limit: Optional[int] = Field(10, ge=1, le=50)


class PlannerDecision(BaseModel):
    tool_name: ToolName
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    date_params: DateRangeParams = Field(default_factory=DateRangeParams)
    filter_params: FilterParams = Field(default_factory=FilterParams)
    raw_query: Optional[str] = None
    feature_topic: Optional[str] = None
    clarification_question: Optional[str] = None
    follow_up_resolved: bool = False


class ConversationContext(BaseModel):
    last_tool: Optional[str] = None
    last_metric: Optional[str] = None
    last_period_label: Optional[str] = None
    last_start_date: Optional[str] = None
    last_end_date: Optional[str] = None
    last_transaction_type: Optional[str] = None
    user_id: Optional[str] = None
    store_id: Optional[str] = None
