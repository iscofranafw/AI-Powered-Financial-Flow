# AI-Powered Financial Recording & Cash Flow Assistant

**AI-Based Financial Recording, Analysis, and Cash Flow Automation for MSMEs**

## 1. Overview

**AI-Powered Financial Recording & Cash Flow Assistant** is a Python and Streamlit web application developed to help micro, small, and medium enterprises (MSMEs) record, manage, and analyze business cash flow.

The application integrates financial transaction management, a transaction approval system, report visualization, and local artificial intelligence using **Llama 3 8B through Ollama**.

Application development focuses on common challenges in MSME financial management, such as:

- Inconsistent recording of income and expenses.
- Difficulty grouping transactions by business category.
- Mixing additional capital with operating revenue.
- Limited information for evaluating cash flow conditions.
- The need to oversee transactions recorded by cashiers.

The system is designed to support various business types, including bakeries, wholesalers, repair shops, and small eateries.

### Development Status

**Development Status: Finalization for Dicoding Capstone Project Submission**

This repository documents the complete development of the application. All core system features have been implemented, including:
- User authentication and role-based access separation (Superadmin, Store Owner, Cashier).
- Recording and management of financial transactions (income, expenses, and separation of capital from operating revenue).
- Transaction approval by the Store Owner (*owner approval workflow*).
- Bulk transaction imports via CSV with integrity validation, anomaly detection, and atomic write idempotency protection.
- Artificial intelligence-based transaction classification (*AI classification*) with deterministic amount extraction and text normalization.
- Semantic embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions) + FAISS `IndexFlatIP` + a *similarity gate* (`score >= 0.80`, `margin >= 0.04`) for fast transaction classification, followed by Llama 3 8B restricted to the store's category list (`allowed_categories`) without *few-shot* RAG in the prompt.
- Confirmation of initial store categories during store creation / CSV upload before classification runs (strict `allowed_categories`, with inputs outside the list routed to `perlu_tinjauan`).
- Business Guidance RAG isolated by `jenis_usaha` (`ai/knowledge_rag.py`, `data/knowledge/`, `knowledge_store/`) for store owners' operational questions.
- Adaptive multi-model business cash expense forecasting (`Prophet`, `Seasonal Naive`, `MA7`, `ETS`, `ARIMA`, `LinearRegression`) with automatic selection of the model with the lowest MAPE and a 3–90 day horizon.
- An AI-based conversational financial assistant (*AI financial chat assistant*) with a floating agent and confirmation of data changes.

The project is currently in finalization and comprehensive testing for Dicoding capstone project submission. The application is designed as an operational cash flow assistant for MSMEs and is not a replacement for formal accounting systems for taxation or payroll.

---

## 2. Application Features

### 2.1. Financial Transaction Management

- Recording business income and expenses.
- Grouping transactions by category.
- Managing transaction history.
- Filtering data by period.
- Separating additional capital from operating income.
- Managing transactions according to user access rights.

### 2.2. AI-Based Transaction Classification

The application integrates Llama 3 8B through Ollama to help interpret transaction descriptions in natural language.

The capabilities developed include:

- Normalizing transaction text and handling spelling variations.
- Extracting amounts using deterministic rules.
- Identifying transaction types based on business context.
- Recommending categories from the store's category catalog.
- Handling model response errors and providing safe fallbacks.

For example, the description:

`Beli telur 3 tray 250rb`

can be processed to identify the amount **Rp250.000** and automatically determine the relevant expense category.

Classification is supported by **SentenceTransformers and FAISS** as a fast *similarity gate* before falling back to Llama 3 8B, which is restricted to the store's initial category catalog.

### Model Adaptation to Data

The language model is Llama 3 8B (through Ollama) without fine-tuning. For transaction classification: (1) the first path computes 384-dimensional embeddings + a FAISS *similarity gate* (`score >= 0.80` and a rank-2 margin `>= 0.04`) using the store's category and confirmed transaction indexes (`vector_store/<toko_id>/`), (2) if the *similarity gate* is not passed, Llama 3 8B is called **without** *few-shot* RAG examples and selects only from the store's official category list established at the start (`allowed_categories`), (3) results outside the list or with confidence `< 70` receive the `perlu_tinjauan` status. Separately, the business guidance question-and-answer feature (`Panduan Usaha`) uses Markdown RAG + FAISS isolated by `jenis_usaha` in `knowledge_store/<jenis_usaha>/`. Note: every synthetic development dataset in the repository is explicitly labeled as **synthetic data**, not actual store data.

### 2.3. Dashboard and Cash Flow Analysis

The dashboard provides a summary of the business's financial condition, including:

- Total income.
- Total expenses.
- Net cash flow.
- Transaction trends by period.
- Expense distribution by category.
- Interactive data visualizations using Plotly.

The system distinguishes operating cash flow from cash flow that includes additional capital so users can better understand the sources of balance changes.

### 2.4. AI-Based Financial Report Narratives

Llama 3 integration helps compose financial report explanations in accessible language. Numerical values are calculated deterministically by application code or the database, while the language model writes narratives based on those calculations to prevent numerical hallucinations.

### 2.5. Multi-Tenant and Role-Based Access Control

The application is designed to support multiple stores with role-based access separation:

**Superadmin**
- Review and approve new store owner registrations.
- Review store transaction deletion requests.
- Manage application administration across stores.

**Store Owner (Owner)**
- Manage store transactions and the category catalog.
- Review and approve transactions recorded by cashiers.
- Issue cashier registration tickets.
- Access the dashboard, cash flow forecasts, and financial reports.

**Cashier (Cashier)**
- Record daily cashier transactions.
- Submit transactions to the store owner's approval queue.
- View the history of transactions recorded by that cashier.

Supabase PostgreSQL is used for data storage and applies **Row Level Security (RLS)** policies to ensure data isolation between stores.

### 2.6. CSV Transaction Imports

The import feature supports:

- Reading transactions from CSV files.
- Checking and validating data structures before storage.
- Reviewing errors row by row.
- Duplicate detection and atomic write idempotency through the import identity ledger (`import_transaction_identities`).
- Grouping bulk transactions into a single database transaction.

### 2.7. Business Category Management

The system provides category settings based on business needs. Examples of business types supported through reference data:

- Bakery (Bakery Store)
- Motorcycle Repair Shop
- Wholesaler / Grocery Store
- Small Eatery

---

## 3. System Architecture

The application uses a modular approach that separates the interface, business logic, AI integration, and database access.

```text
                     User
                      |
                      v
              Streamlit Web App
                      |
          +-----------+-----------+
          |                       |
          v                       v
    Business Services        AI Services
          |                       |
          v                       v
   Supabase PostgreSQL      Ollama / Llama 3
          |                       |
          v                       v
   Auth, Transactions       Text Processing
   Categories, Reports      Classification
   Role-Based Access        Financial Narrative
```

### Main Components

| Component | Technology | Function |
|---|---|---|
| Programming language | Python 3.11+ | Backend and frontend application logic |
| Interface | Streamlit | Dashboard and user interaction |
| Database | Supabase PostgreSQL | Storage, RLS, and atomic mutation RPCs |
| Authentication | Supabase Auth | Session and user identity management |
| Language model | Llama 3 8B | Natural language processing and classification reasoning |
| Local inference | Ollama | Local language model inference server |
| Vector Engine | FAISS & SentenceTransformers | Classification *similarity gate* (`vector_store/`) and Business Guidance RAG (`knowledge_store/`) |
| Cash Forecasting | Prophet, Statsmodels (ETS/ARIMA), Scikit-Learn (LinearRegression), MA7, Seasonal Naive | Adaptive multi-model expense forecasting (3–90 day horizon) |
| Data processing | Pandas, NumPy | Data transformation and financial aggregation |
| Visualization | Plotly | Interactive report charts and diagrams |
| Report Export | OpenPyXL | Excel spreadsheet report generation |
| Testing | Pytest | Automated unit and integration test suites |

---

## 4. Directory Structure

```text
AI-Powered-Financial-Flow/
├── .streamlit/
│   └── config.toml
├── ai/
│   ├── category_discovery.py
│   ├── category_evidence.py
│   ├── category_suggester.py
│   ├── classifier.py
│   ├── copilot_knowledge.py
│   ├── copilot_planner.py
│   ├── copilot_schema.py
│   ├── copilot_tools.py
│   ├── csv_cluster.py
│   ├── embedding_pipeline.py
│   ├── financial_chat.py
│   ├── guidance_prompts.py
│   ├── hybrid_classifier.py
│   ├── ollama_client.py
│   ├── rag_engine.py          # Embedding + FAISS similarity gate (vector_store/)
│   ├── knowledge_rag.py       # Business Guidance RAG isolated by jenis_usaha (knowledge_store/)
│   ├── store_context.py
│   └── text_normalizer.py
├── config/
│   └── settings.py
├── data/
│   ├── knowledge/
│   │   ├── bakery/
│   │   ├── bengkel_motor/
│   │   ├── category_keyword_profiles.json
│   │   ├── lainnya/
│   │   ├── retail/
│   │   └── warung_makan/
│   └── samples/
│       ├── bakery.csv
│       ├── bengkel_motor.csv
│       ├── grosir.csv
│       └── warung_makan.csv
├── database/
│   ├── client.py
│   ├── migrations/
│   │   ├── 001_initial.sql
│   │   ├── 002_bootstrap_superadmin.sql
│   │   ├── 003_stage2_finance.sql
│   │   ├── 004_fix_cashier_provisioning.sql
│   │   ├── 005_ai_forecast.sql
│   │   ├── 006_transaction_deletion_requests.sql
│   │   ├── 007_import_idempotency.sql
│   │   ├── 008_chat_proposal_idempotency.sql
│   │   ├── 009_audit_log.sql
│   │   ├── 010_atomic_audit_mutations.sql
│   │   ├── 011_nullable_kategori_for_ai_pending.sql
│   │   └── README.md
│   └── schema.sql
├── pages/
│   ├── ai_tools.py
│   ├── cashier.py
│   ├── chat_view.py
│   ├── finance.py
│   ├── guidance_view.py
│   ├── import_view.py
│   ├── login.py
│   ├── owner.py
│   ├── report_view.py
│   └── superadmin.py
├── repositories/
│   ├── deletion_repository.py
│   ├── finance_repository.py
│   └── profile_repository.py
├── scripts/
│   ├── build_dist_zip.py
│   ├── build_dist_zip.sh
│   ├── build_knowledge_index.py
│   ├── measure_accuracy_and_mape.py
│   ├── measure_bakery_keyword_fix.py
│   ├── measure_category_correctness_round2.py
│   ├── measure_new_user_generalization.py
│   ├── measure_recorded_retail_baseline_gate.py
│   ├── reproduce_import_session_before.py
│   ├── verify_display_forecast.py
│   ├── verify_import_session_db.py
│   └── verify_post_push_guidance.py
├── services/
│   ├── admin_service.py
│   ├── ai_service.py
│   ├── auth_service.py
│   ├── cashier_service.py
│   ├── chat_service.py
│   ├── dataset_service.py
│   ├── deletion_service.py
│   ├── finance_service.py
│   ├── forecast_service.py
│   ├── import_category_policy.py
│   ├── import_service.py
│   ├── import_session.py
│   ├── report_service.py
│   ├── smart_import_engine.py
│   └── validation_service.py
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── bakery_16_descriptions.json
│   │   ├── new_user_categories.json
│   │   └── retail_22_descriptions.json
│   ├── test_accuracy_evaluation.py
│   ├── test_admin.py
│   ├── test_ai_rules.py
│   ├── test_ai_service.py
│   ├── test_audit_remediation.py
│   ├── test_auth.py
│   ├── test_bulk_import_benchmark.py
│   ├── test_cashier.py
│   ├── test_cashier_and_admin.py
│   ├── test_category_discovery.py
│   ├── test_category_evidence.py
│   ├── test_chat_assistant.py
│   ├── test_chat_service_protections.py
│   ├── test_classification.py
│   ├── test_copilot_forecasting.py
│   ├── test_copilot_validation.py
│   ├── test_deletion_approval.py
│   ├── test_display_simplification.py
│   ├── test_finance_service.py
│   ├── test_floating_agent.py
│   ├── test_import_security.py
│   ├── test_import_session.py
│   ├── test_knowledge_rag.py
│   ├── test_multi_domain_classification.py
│   ├── test_performance_optimization.py
│   ├── test_post_approval_classification_and_floating_agent.py
│   ├── test_post_push_improvements.py
│   ├── test_preprocessing.py
│   ├── test_rag.py
│   ├── test_recurring_calendar.py
│   ├── test_recurring_calendar_ui.py
│   ├── test_recurring_reminder_ui.py
│   ├── test_reports.py
│   ├── test_smart_import.py
│   ├── test_stage2_rules.py
│   └── test_store_aware_classification.py
├── utils/
│   ├── floating_agent.py
│   ├── forecast_details.py
│   ├── recurring_calendar.py
│   └── ui_components.py
├── .env.example
├── .gitignore
├── app.py
├── LICENSE
├── measure_retail_before.py
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## 5. Installation and Configuration

Follow the steps below in order to set up the application from scratch and run it on a local device.

### 5.1. System Prerequisites

Before starting installation, make sure your local device meets the following prerequisites:

1. **Python**: Version `3.11.x` (Python 3.11.9 recommended).
2. **Git**: For cloning the source code repository.
3. **Ollama**: Installed and running in the background for local Llama 3 8B language model inference.
4. **Supabase Account**: An active project on [Supabase](https://supabase.com) (PostgreSQL 15+).
5. **Internet Connection**: Required to download Python dependencies and language models, and to access the Supabase API.

### 5.2. Clone the Repository

Open a terminal or PowerShell, then run the following commands to clone the repository:

```bash
git clone https://github.com/iscofranafw/AI-Powered-Financial-Flow.git
cd AI-Powered-Financial-Flow
```

### 5.3. Create and Activate a Virtual Environment

Use a clean Python virtual environment to avoid dependency conflicts.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS (Bash)**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.4. Install Python Dependencies

Upgrade `pip` to the latest version, then install all dependency packages listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5.5. Set Up the Supabase Database

The application uses a consolidated PostgreSQL database schema that combines all tables, indexes, atomic mutation RPC procedures, and row-level security (RLS) policies.

1. Sign in to the [Supabase Dashboard](https://supabase.com/dashboard) and create a new project (*New Project*). Set the project name, database password, and nearest region.
2. Wait for project creation to finish and the project status to become active.
3. Navigate to **SQL Editor** in the left sidebar of the Supabase dashboard.
4. Open `database/schema.sql` from this repository in your text editor, copy its entire contents, and paste them into the Supabase **SQL Editor**.
5. Click **Run** to execute the database schema. This schema is designed to be run **only once (*run ONCE*)** on an empty database project and will create all of the following database objects atomically:
   - The `pgcrypto` extension.
   - 9 main tables in a *foreign-key safe order*: `toko`, `profiles`, `kategori`, `transaksi`, `cashier_provisioning`, `rag_index_version`, `transaction_deletion_requests`, `import_transaction_identities`, and `audit_log`.
   - 12 performance and data integrity indexes (including partial unique indexes for chat idempotency and transaction deletion).
   - 12 safe mutation stored procedure / RPC functions marked `SECURITY DEFINER` (such as `mutate_finance_audited`, `import_transactions_atomic`, and `approve_transactions_atomic`).
   - 3 database triggers (`on_auth_user_created`, `kategori_rag_bump`, `transaksi_rag_bump`).
   - Row Level Security (RLS) policies on all tables, along with special `service_role` access rights.
   - An idempotent Superadmin bootstrap block.

> **Note**
> If you have already applied migrations `001` through `011` in sequence to an existing database project, you **DO NOT NEED** to run `database/schema.sql`, as it produces the same final state. The `database/migrations/` directory remains in the repository as a historical record of database schema evolution.

### 5.6. Configure Environment Variables (`.env`)

Copy the `.env.example` template to a file named `.env` in the repository root directory:

**Windows (PowerShell)**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS (Bash)**

```bash
cp .env.example .env
```

Open `.env` in a text editor, then fill in the configuration variables with your Supabase project credentials:

```ini
# ==============================================================================
# AI-POWERED FINANCIAL FLOW - ENVIRONMENT CONFIGURATION
# ==============================================================================

# Supabase Credentials (Get these from Project Settings -> API)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy_anon_public_key
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy_service_role_key

# Local Ollama Inference Service Configuration
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3:8b

# Semantic Embedding Model (Optional, automatically falls back to MiniLM if empty)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**Environment Variable Descriptions:**

| Variable Name | Status | Description |
|---|:---:|---|
| `SUPABASE_URL` | **Required** | Your Supabase project's REST API endpoint URL (found under *Project Settings* to *API* to *Project URL*). |
| `SUPABASE_ANON_KEY` | **Required** | The anonymous public API key for client session authentication (*Project Settings* to *API* to *Project API keys* to `anon / public`). |
| `SUPABASE_SERVICE_ROLE_KEY` | **Required** | The secret `service_role` API key for privileged atomic mutations in the server backend layer (*Project Settings* to *API* to *Project API keys* to `service_role`). |
| `OLLAMA_HOST` | Optional | The local Ollama service URL (default: `http://127.0.0.1:11434`). |
| `OLLAMA_MODEL` | Optional | The language model name used through Ollama (default: `llama3:8b`). |
| `EMBEDDING_MODEL` | Optional | The sentence-transformers model name for similarity gate embeddings and Business Guidance (default: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`). |

> **Caution**
> Never share `.env` or include `SUPABASE_SERVICE_ROLE_KEY` in a public repository. The `.env` file is listed in `.gitignore` and must remain confidential.

### 5.7. Set Up the Local AI Service (Ollama)

1. Make sure Ollama is installed on your computer. If it is not, download it from [ollama.com](https://ollama.com).
2. Run the following command in a terminal to download the **Llama 3 8B** model:

```bash
ollama pull llama3:8b
```

3. Verify that the model is installed and the Ollama service is responding:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

Make sure the `llama3:8b` tag appears in the returned model list.

### 5.8. Run the Application

Run the Streamlit application server from a terminal in the repository root directory:

```bash
streamlit run app.py
```

Once the server has initialized successfully, open your web browser and navigate to:

```text
http://localhost:8501
```

### 5.9. First Login and Setup for the Three User Roles

The system applies access control based on three user roles, with the following initial setup workflow:

#### 1. Register a Store Owner Account (Owner)
- Open the application in a browser on the **Login** page.
- Select the **Register New Account** tab.
- Enter all required details: Username, Email, Password, Store Name, and Business Type (choose one: *Bakery*, *Wholesaler*, *Motorcycle Repair Shop*, or *Small Eatery*).
- Click **Register**. A new store owner account will be created automatically with the initial status `pending` (awaiting Superadmin approval before it can manage store transactions).

#### 2. Activate a Superadmin Account
- To activate the first Superadmin account in a new database:
  1. Register an account through the application's registration form using the email address you want to use for the main administrator.
  2. Open **SQL Editor** in the Supabase dashboard.
  3. Run the following SQL code, replacing `'REPLACE_WITH_YOUR_ADMIN_EMAIL'` with the email address of the account you just registered:

```sql
WITH candidate AS (
    SELECT id, toko_id
    FROM public.profiles
    WHERE email = 'REPLACE_WITH_YOUR_ADMIN_EMAIL' AND role = 'pemilik_toko'
    LIMIT 1
), promoted AS (
    UPDATE public.profiles AS p
    SET role = 'superadmin', toko_id = NULL, status = 'aktif'
    FROM candidate AS c
    WHERE p.id = c.id
    RETURNING p.id, c.toko_id
), removed_shop AS (
    DELETE FROM public.toko AS t
    USING promoted AS p
    WHERE t.id = p.toko_id
    RETURNING t.id
), confirmed_user AS (
    UPDATE auth.users AS u
    SET email_confirmed_at = coalesce(u.email_confirmed_at, now())
    FROM promoted AS p
    WHERE u.id = p.id
      AND (SELECT count(*) FROM removed_shop) >= 0
    RETURNING u.id
)
SELECT 'Superadmin candidate not found — pastikan akun sudah terdaftar terlebih dahulu.' AS notice
WHERE NOT EXISTS (SELECT 1 FROM promoted)
```

  4. Once the script has executed successfully, return to the application and log in with that email address. You will be taken directly to the **Superadmin** navigation page.
  5. On the Superadmin page, you can view store owners with the `pending` status and click **Approve** to activate a store.

#### 3. Register a Cashier Account (Cashier)
- Cashier accounts **cannot register independently** without restriction, to prevent unauthorized transaction recording.
- A Store Owner with an active account opens **Cashier Management** in the sidebar and creates a cashier registration ticket (specifying the cashier's name and email).
- The cashier registers with that ticket token through the cashier registration form, so the account is automatically linked to the authorized store with an immediate `aktif` status.

### 5.10. Use Sample Data (*Sample Datasets*)

To try cash flow analysis, multi-model expense forecasting, and AI classification without manually recording transactions one by one, the repository includes **synthetic** datasets (not actual store data) in `data/samples/`:

- `data/samples/bakery.csv`: Bakery and cake shop transaction dataset (purchases of flour, eggs, and butter, daily sales, initial capital).
- `data/samples/bengkel_motor.csv`: Motorcycle repair shop transaction dataset (spare parts, engine oil, repair services, mechanic tools).
- `data/samples/grosir.csv`: Wholesale and staple goods store transaction dataset (merchandise purchasing, logistics operations, cash receipts).
- `data/samples/warung_makan.csv`: Food business / small eatery transaction dataset (market purchases, spices, LPG gas, daily revenue).

**How to Import Sample Data:**
1. Log in as a **Store Owner** with an active account.
2. Select **CSV Transaction Import** in the left sidebar.
3. Upload one of the sample files from `data/samples/`.
4. The system will display a data preview, verify column compatibility, detect date anomalies, and match categories.
5. Click **Save Transactions** to insert the data into the database in bulk.

---

## 6. Testing

The project includes automated test suites using the **Pytest** framework to verify business logic, transaction validation, role authorization, idempotency handling, and AI algorithms.

Run the entire test suite from the root directory:

```bash
pytest tests/ -v
```

Or use the Python module:

```bash
python -m pytest -v
```

### Test Coverage

The test modules available in `tests/` include:

- `test_auth.py`: Supabase authentication and user session management.
- `test_cashier.py` & `test_cashier_and_admin.py`: Cashier access rights, cashier registration tickets, and owner approval.
- `test_finance_service.py` & `test_stage2_rules.py`: Transaction recording logic, date filtering, and capital separation.
- `test_import_security.py` & `test_smart_import.py`: CSV structure validation, row limits, payload sanitization, and import idempotency.
- `test_ai_rules.py`, `test_ai_service.py`, & `test_classification.py`: Amount extraction, text normalization, and the Llama 3 pipeline.
- `test_rag.py` & `test_knowledge_rag.py`: Classification FAISS *similarity gate* indexes (`vector_store/`) and Business Guidance RAG isolated by `jenis_usaha` (`knowledge_store/`).
- `test_copilot_forecasting.py` & `test_copilot_validation.py`: The multi-model forecasting engine (Prophet, Seasonal Naive, MA7, ETS, ARIMA, LinearRegression, 3–90 day horizon) and financial narrative input validation.
- `test_deletion_approval.py`: The workflow for requesting superadmin approval of store transaction deletions.
- `test_audit_remediation.py`: Atomic audit trail verification and concurrency protection.
- `test_floating_agent.py` & `test_chat_assistant.py`: AI financial assistant agent interactions and execution of conversational instructions.

### Testing Status

All unit and mock tests can be executed locally without relying on an internet connection. For live integration tests that interact directly with the Supabase database or local Ollama, make sure the `.env` variables and Ollama daemon are configured as described in section 5.

---

## 7. Development Roadmap

Development proceeds in stages using a **feature-based development** approach.

Each feature is implemented, tested, corrected if needed, and then documented through Git commits.

| Stage | Development Focus | Status |
|---|---|---|
| Baseline | Application foundation, authentication, transactions, dashboard, initial AI integration | Completed |
| 1 | Reference datasets and preprocessing | Completed |
| 2 | SentenceTransformers and FAISS Vector Indexing | Completed |
| 3 | Embedding classification + FAISS similarity gate + restricted Llama 3 | Completed |
| 4 | Initial CSV category confirmation & business type isolation | Completed |
| 5 | Bulk transaction classification restricted to `allowed_categories` | Completed |
| 6 | Business Guidance RAG isolated by `jenis_usaha` (`knowledge_store/`) | Completed |
| 7 | AI Financial Chat Assistant (Floating Agent + Business Guidance) | Completed |
| 8 | Adaptive multi-model expense forecasting (Prophet, ETS, ARIMA, etc.) | Completed |
| 9 | AI model performance evaluation and numerical validation | Completed |
| 10 | Final integration testing, schema consolidation, and documentation | Completed |

### AI Evaluation Targets

In line with the capstone project objectives, artificial intelligence performance is evaluated against the following quantitative criteria:

- **Transaction classification accuracy of at least 85%.**
- **Cash expense forecasting MAPE of at most 20%.**
- Evaluation of *embedding + FAISS similarity gate* classification + restricted Llama 3 **without** *few-shot* RAG in the prompt.
- Testing transactions outside the category list (`perlu_tinjauan` / `Others`) and preventing unauthorized category creation.
- Measurement of MAE, RMSE, WAPE, and MAPE for automatic multi-model forecasting selection (3–90 day horizon) on actual store data and clearly labeled synthetic datasets.

### AI Financial Chat Assistant Features

The intelligent financial assistant's chat feature allows users to:

- Record income interactively in natural language.
- Record expenses interactively in natural language.
- Submit multiple transactions in one conversation.
- Review and confirm transactions before they are stored in the database.
- Ask for a summary of store cash flow.
- Browse transaction history by period or category.
- Request explanations of financial conditions based on actual store data.

The chat feature operates under the same access rules and validation layers as the main application interface.

---

## 8. Security and Privacy

Application development follows *defense-in-depth* principles:

**Environment Variables**
Credentials are stored through environment configuration (`.env`) and must not be included in source code or public repositories.

**Role-Based Access Control**
User access is strictly restricted according to the Superadmin, Store Owner, and Cashier roles.

**Row Level Security (RLS)**
Supabase PostgreSQL applies RLS rules to all tables to ensure secure data isolation between stores. Critical mutations run through atomic RPCs with managed access rights.

**Local AI Inference**
Llama 3 language model processing runs locally through the Ollama daemon on the user's machine, keeping business transaction descriptions private without sending conversation text to external third-party APIs.

**Data Validation and Audit Trails**
All transaction inputs are validated before storage. Important transaction changes are recorded in an encapsulated audit ledger (`audit_log`).

**Archive Distribution & Data Sanitization (Packaging Instructions)**

To prevent sensitive data leaks when distributing the application or packaging project archives (such as `.zip` files), make sure the following assets **are excluded**:
1. **Local Vector Indexes (`vector_store/`):** All `.faiss` files and `_meta.json` metadata contain embeddings of actual store transaction text. These files are ignored by `.gitignore` and will be regenerated automatically when the application runs in a new environment.
2. **Actual Transaction Data (`data/Daftar_Transaksi_*.csv`):** Users' actual transaction files must not be included in distribution packages. The `data/samples/` folder (such as `bakery.csv`, `bengkel_motor.csv`, etc.) is retained because it contains synthetic datasets for testing and demonstration.
3. **Credentials and Secrets:** The `.env`, `.env.*`, and `.streamlit/secrets.toml` configuration files.

Commands for creating a safe distribution archive:

```bash
# Use the automated packaging script (recommended, cross-platform):
python scripts/build_dist_zip.py financial-flow-dist.zip

# Or use zip (Linux / macOS / Git Bash):
zip -r financial-flow-dist.zip . -x "vector_store/*" "data/Daftar_Transaksi_*.csv" ".env*" ".git/*" "__pycache__/*" ".pytest_cache/*"

# Or use PowerShell (Windows):
$excludePatterns = @('[\\/]vector_store', 'Daftar_Transaksi_.*\.csv$', '^\.env', '[\\/]\.git', '[\\/]__pycache__', '[\\/]\.pytest_cache')
Get-ChildItem -Path . -Recurse | Where-Object {
    $item = $_
    -not ($excludePatterns | Where-Object { $item.FullName -match $_ -or $item.Name -match $_ })
} | Compress-Archive -DestinationPath financial-flow-dist.zip
```

---

## 9. Contributing

Contributions to project development are open to the community.

The recommended contribution workflow:

1. Fork the repository.
2. Create a branch for a feature or fix (`git checkout -b feature/nama-fitur`).
3. Implement the code changes.
4. Run automated tests (`pytest tests/ -v`).
5. Create a commit with a clear, descriptive message.
6. Submit a Pull Request with an explanation of the changes.

Contributions are expected to maintain compatibility with the project architecture and exclude credentials, private transaction data, and other sensitive information.

---

## 10. License

**AI-Powered Financial Recording & Cash Flow Assistant** is distributed under the **MIT License**.

The source code may be used, copied, modified, merged, published, and distributed, including for commercial use, under the terms of the MIT license.

See [LICENSE](LICENSE) for the full license terms.

---

## 11. Development Notes

This repository is part of the **Dicoding Bootcamp — DB15-I023** capstone project development.

Documentation and implementation are updated in line with Dicoding curriculum standards, integration test results, and AI model performance evaluations. The project is developed as open-source software that the Indonesian MSME community can study, use, and develop further.

---

## 12. Sources & Citations

- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2` (SentenceTransformers) — https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Vector database: FAISS (Meta AI Research) — https://github.com/facebookresearch/faiss
- Forecasting: Prophet (Meta) — https://github.com/meta/prophet, Statsmodels (`ETS`, `ARIMA`) — https://www.statsmodels.org, and Scikit-Learn (`LinearRegression`) — https://scikit-learn.org
- Language model: Llama 3 8B (Meta), run locally through Ollama — https://ollama.com and https://llama.meta.com/llama3/
- Embedding library: SentenceTransformers — https://www.sbert.net
- Database: Supabase (PostgreSQL) — https://supabase.com
- Interface: Streamlit — https://streamlit.io
- Dataset: sample data in `data/samples/` (bakeries, wholesalers, motorcycle repair shops, small eateries) is synthetic and was created for this project because actual store transaction data was not available.
