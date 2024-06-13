# %%
import duckdb
import numpy as np
import pandas as pd
from itables import show
from tqdm.auto import tqdm

# %%
con = duckdb.connect(database="data/duckdb.db")
cursor = con.cursor()

# %% [markdown]
# ## Extract

# %%
df = pd.read_parquet("data/output.parquet")

# %% [markdown]
# ## Transform and Load

# %%
df.rename(
    {
        "LoanNr_ChkDgt": "loan_number",
        "Name": "name",
        "City": "city",
        "State": "state",
        "Zip": "zip_code",
        "Bank": "bank_name",
        "BankState": "bank_state",
        "NAICS": "naics_code",
        "ApprovalDate": "approval_date",
        "ApprovalFY": "approval_year",
        "Term": "term",
        "NoEmp": "n_employees",
        "NewExist": "new_or_existing",
        "CreateJob": "n_jobs_created",
        "RetainedJob": "n_jobs_retained",
        "FranchiseCode": "franchise_code",
        "UrbanRural": "urban_or_rural",
        "RevLineCr": "rev_line_of_credit",
        "LowDoc": "low_doc",
        "ChgOffDate": "charged_off_date",
        "DisbursementDate": "disbursement_date",
        "DisbursementGross": "disbursement_gross",
        "BalanceGross": "balance_gross",
        "MIS_Status": "status",
        "ChgOffPrinGr": "charged_off_amount",
        "GrAppv": "approved_gross",
        "SBA_Appv": "sba_guaranteed_amount",
        "DisbursementFY": "disbursement_year",
        "IsFranchised": "has_franchise",
        "IsCreatedJob": "has_jobs_created",
        "IsRetainedJob": "has_jobs_retained",
        "Industry": "industry",
        "RealEstate": "real_estate_backed",
        "DaysTerm": "term_in_days",
        "Active": "maturity_date",
        "Recession": "recession",
        "DaysToDisbursement": "days_to_disbursement",
        "StateSame": "in_bank_state",
        "SBA_AppvPct": "sba_guaranteed_ratio",
        "AppvDisbursed": "full_amount_disbursed",
        "IsExisting": "is_existing",
    },
    axis=1,
    inplace=True,
)

# %%
df.info()


# %%
def create_states():
    query = """
        CREATE SEQUENCE IF NOT EXISTS states_serial;

        CREATE TABLE IF NOT EXISTS states (
            state_key INTEGER PRIMARY KEY DEFAULT nextval('states_serial'),
            name TEXT NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS states_name_idx ON states (name);
    """

    cursor.execute(query)


def create_borrowers():
    query = """
        CREATE SEQUENCE IF NOT EXISTS borrowers_serial;

        CREATE TABLE IF NOT EXISTS borrowers (
            borrower_key INTEGER PRIMARY KEY DEFAULT nextval('borrowers_serial'),
            state_key INTEGER NOT NULL REFERENCES states (state_key),
            name TEXT NOT NULL,
            city TEXT NOT NULL,
            zip_code INTEGER NOT NULL,
            n_employees INTEGER NOT NULL,
            n_jobs_created INTEGER NOT NULL,
            n_jobs_retained INTEGER NOT NULL,
            franchise_code INTEGER NOT NULL,
            urban_or_rural BOOLEAN NOT NULL,
            has_franchise BOOLEAN NOT NULL,
            has_jobs_created BOOLEAN NOT NULL,
            has_jobs_retained BOOLEAN NOT NULL,
            naics_code TEXT NOT NULL,
            industry TEXT NOT NULL,
            snapshot_date DATE NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS borrowers_idx ON borrowers (
            state_key,
            name,
            city,
            zip_code,
            snapshot_date
        );
    """

    cursor.execute(query)


def create_banks():
    query = """
        CREATE SEQUENCE IF NOT EXISTS banks_serial;

        CREATE TABLE IF NOT EXISTS banks (
            bank_key INTEGER PRIMARY KEY DEFAULT nextval('banks_serial'),
            state_key INTEGER NOT NULL REFERENCES states (state_key),
            name TEXT NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS banks_name_idx ON banks (name);
    """

    cursor.execute(query)


def create_dates():
    query = """
        CREATE SEQUENCE IF NOT EXISTS dates_serial;

        CREATE TABLE IF NOT EXISTS dates (
            date_key INTEGER PRIMARY KEY DEFAULT nextval('dates_serial'),
            date DATE NOT NULL,
            year SMALLINT NOT NULL,
            quarter TINYINT NOT NULL,
            month TINYINT NOT NULL,
            day TINYINT NOT NULL,
            day_of_week TEXT NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS dates_date_idx ON dates (date);
    """

    cursor.execute(query)


def create_loan_profiles():
    query = """
        CREATE SEQUENCE IF NOT EXISTS loan_profiles_serial;

        CREATE TABLE IF NOT EXISTS loan_profiles (
            loan_profile_key INTEGER PRIMARY KEY DEFAULT nextval('loan_profiles_serial'),
            rev_line_of_credit BOOLEAN NOT NULL,
            low_doc BOOLEAN NOT NULL,
            status BOOLEAN NOT NULL,
            real_estate_backed BOOLEAN NOT NULL,
            recession BOOLEAN NOT NULL,
            in_bank_state BOOLEAN NOT NULL,
            full_amount_disbursed BOOLEAN NOT NULL,
            is_existing BOOLEAN NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS loan_profiles_idx ON loan_profiles (
            rev_line_of_credit,
            low_doc,
            status,
            real_estate_backed,
            recession,
            in_bank_state,
            full_amount_disbursed,
            is_existing
        );
    """

    cursor.execute(query)


def create_loans():
    query = """
        CREATE TABLE IF NOT EXISTS loans (
            borrower_key INTEGER NOT NULL REFERENCES borrowers (borrower_key),
            bank_key INTEGER NOT NULL REFERENCES banks (bank_key),
            approval_date_key INTEGER NOT NULL REFERENCES dates (date_key),
            charged_off_date_key INTEGER REFERENCES dates (date_key),
            disbursement_date_key INTEGER NOT NULL REFERENCES dates (date_key),
            maturity_date_key INTEGER NOT NULL REFERENCES dates (date_key),
            loan_profile_key INTEGER NOT NULL REFERENCES loan_profiles (loan_profile_key),
            loan_number BIGINT NOT NULL,
            term INTEGER NOT NULL,
            term_in_days INTEGER NOT NULL,
            disbursement_gross DOUBLE NOT NULL,
            balance_gross DOUBLE NOT NULL,
            charged_off_amount DOUBLE NOT NULL,
            approved_gross DOUBLE NOT NULL,
            sba_guaranteed_amount DOUBLE NOT NULL,
            days_to_disbursement INTEGER NOT NULL,
            sba_guaranteed_ratio DOUBLE NOT NULL
        );
    """

    cursor.execute(query)


# %%
def drop_all_tables():
    query = """
        DROP TABLE IF EXISTS loans;
        DROP TABLE IF EXISTS loan_profiles;
        DROP TABLE IF EXISTS dates;
        DROP TABLE IF EXISTS banks;
        DROP TABLE IF EXISTS borrowers;
        DROP TABLE IF EXISTS states;
    """

    cursor.execute(query)


# %%
# drop_all_tables()

# %%
create_states()
create_borrowers()
create_banks()
create_dates()
create_loan_profiles()
create_loans()

# %%
cursor.sql("SHOW TABLES")


# %%
def truncate_all_tables():
    query = """
        TRUNCATE loans;
        TRUNCATE loan_profiles;
        TRUNCATE dates;
        TRUNCATE banks;
        TRUNCATE borrowers;
        TRUNCATE states;
    """


# %%
# truncate_all_tables()

# %%
borrowers_cols = [
    "name",
    "city",
    "zip_code",
    "n_employees",
    "n_jobs_created",
    "n_jobs_retained",
    "franchise_code",
    "urban_or_rural",
    "has_franchise",
    "has_jobs_created",
    "has_jobs_retained",
    "naics_code",
    "industry",
]

loan_profiles_cols = [
    "rev_line_of_credit",
    "low_doc",
    "status",
    "real_estate_backed",
    "recession",
    "in_bank_state",
    "full_amount_disbursed",
    "is_existing",
]

loans_cols = [
    "borrower_key",
    "bank_key",
    "approval_date_key",
    "charged_off_date_key",
    "disbursement_date_key",
    "maturity_date_key",
    "loan_profile_key",
    "loan_number",
    "term",
    "term_in_days",
    "disbursement_gross",
    "balance_gross",
    "charged_off_amount",
    "approved_gross",
    "sba_guaranteed_amount",
    "days_to_disbursement",
    "sba_guaranteed_ratio",
]


def prepare_dates(batch: pd.DataFrame, col_name: str):
    return (
        pd.concat(
            objs=(
                batch[col_name],
                batch[col_name].dt.year,
                batch[col_name].dt.quarter,
                batch[col_name].dt.month,
                batch[col_name].dt.day,
                batch[col_name].dt.day_name(),
            ),
            axis=1,
        )
        .dropna()
        .values.tolist()
    )


# %%
def load(batch_size=1000):
    n_rows = df.shape[0]
    n_batches = (n_rows // batch_size) + 1

    for i in tqdm(range(n_batches)):
        batch = df[i * batch_size : (i + 1) * batch_size]
        cursor.begin()

        # States
        states = batch["state"].tolist()
        states_2d = batch[["state"]].values.tolist()

        bank_states = batch["bank_state"].tolist()
        bank_states_2d = batch[["bank_state"]].values.tolist()

        insert_states = """
            INSERT INTO states (name)
            VALUES ($1)
            ON CONFLICT DO NOTHING
        """

        cursor.executemany(insert_states, states_2d)
        cursor.executemany(insert_states, bank_states_2d)

        select_state_keys = """
            SELECT state_key, name
            FROM states
            WHERE name = ANY($1)
        """

        state_rs = cursor.execute(select_state_keys, [states]).df()
        state_kv = state_rs.set_index("name")["state_key"].to_dict()
        state_keys = batch["state"].map(state_kv)

        bank_state_rs = cursor.execute(select_state_keys, [bank_states]).df()
        bank_state_kv = bank_state_rs.set_index("name")["state_key"].to_dict()
        bank_state_keys = batch["bank_state"].map(bank_state_kv)

        # Borrowers
        borrowers = [
            batch["loan_number"].tolist(),
            state_keys.tolist(),
            batch["name"].tolist(),
            batch["city"].tolist(),
            batch["zip_code"].tolist(),
            batch["approval_date"].tolist(),
        ]

        borrowers_2d = pd.concat(
            objs=(state_keys, batch[[*borrowers_cols, "approval_date"]]), axis=1
        ).values.tolist()

        insert_borrowers = f"""
            INSERT INTO borrowers (state_key, {", ".join(borrowers_cols)}, snapshot_date)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            ON CONFLICT DO NOTHING
        """

        cursor.executemany(insert_borrowers, borrowers_2d)

        select_borrower_keys = """
            WITH t AS (
                SELECT unnest($1) AS loan_number, unnest($2) AS state_key,
                    unnest($3) AS name, unnest($4) AS city, unnest($5) AS zip_code,
                    unnest($6) AS snapshot_date
            )
            SELECT borrower_key, loan_number
            FROM t
            JOIN borrowers b ON (
                b.state_key = t.state_key
                AND b.name = t.name
                AND b.city = t.city
                AND b.zip_code = t.zip_code
                AND b.snapshot_date = t.snapshot_date
            )
        """

        borrower_rs = cursor.execute(select_borrower_keys, borrowers).df()
        borrower_kv = borrower_rs.set_index("loan_number")["borrower_key"].to_dict()
        borrower_keys = batch["loan_number"].map(borrower_kv)

        # Banks
        banks = batch["bank_name"].tolist()
        banks_2d = pd.concat(
            (bank_state_keys, batch["bank_name"]), axis=1
        ).values.tolist()

        insert_banks = """
            INSERT INTO banks (state_key, name)
            VALUES ($1, $2)
            ON CONFLICT DO NOTHING
        """

        cursor.executemany(insert_banks, banks_2d)

        select_bank_keys = """
            SELECT bank_key, name
            FROM banks
            WHERE name = ANY($1)
        """

        bank_rs = cursor.execute(select_bank_keys, [banks]).df()
        bank_kv = bank_rs.set_index("name")["bank_key"].to_dict()
        bank_keys = batch["bank_name"].map(bank_kv)

        # Dates
        approval_dates = batch["approval_date"].tolist()
        charged_dates = batch["charged_off_date"].tolist()
        dis_dates = batch["disbursement_date"].tolist()
        maturity_dates = batch["maturity_date"].tolist()

        approval_dates_2d = prepare_dates(batch, col_name="approval_date")
        charged_off_dates_2d = prepare_dates(batch, col_name="charged_off_date")
        disbursement_dates_2d = prepare_dates(batch, col_name="disbursement_date")
        maturity_dates_2d = prepare_dates(batch, col_name="maturity_date")

        insert_dates = """
            INSERT INTO dates (date, year, quarter, month, day, day_of_week)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT DO NOTHING
        """

        cursor.executemany(insert_dates, approval_dates_2d)
        cursor.executemany(insert_dates, charged_off_dates_2d)
        cursor.executemany(insert_dates, disbursement_dates_2d)
        cursor.executemany(insert_dates, maturity_dates_2d)

        select_date_keys = """
            SELECT date_key, date
            FROM dates
            WHERE date = ANY($1)
        """

        approval_date_rs = cursor.execute(select_date_keys, [approval_dates]).df()
        approval_date_kv = approval_date_rs.set_index("date")["date_key"].to_dict()
        approval_date_keys = batch["approval_date"].map(approval_date_kv)

        charged_date_rs = cursor.execute(select_date_keys, [charged_dates]).df()
        charged_date_kv = charged_date_rs.set_index("date")["date_key"].to_dict()
        charged_date_keys = batch["charged_off_date"].map(charged_date_kv)

        dis_date_rs = cursor.execute(select_date_keys, [dis_dates]).df()
        dis_date_kv = dis_date_rs.set_index("date")["date_key"].to_dict()
        dis_date_keys = batch["disbursement_date"].map(dis_date_kv)

        maturity_date_rs = cursor.execute(select_date_keys, [maturity_dates]).df()
        maturity_date_kv = maturity_date_rs.set_index("date")["date_key"].to_dict()
        maturity_date_keys = batch["maturity_date"].map(maturity_date_kv)

        # Loan Profiles
        loan_profiles = [
            batch["loan_number"].tolist(),
            batch["rev_line_of_credit"].tolist(),
            batch["low_doc"].tolist(),
            batch["status"].tolist(),
            batch["real_estate_backed"].tolist(),
            batch["recession"].tolist(),
            batch["in_bank_state"].tolist(),
            batch["full_amount_disbursed"].tolist(),
            batch["is_existing"].tolist(),
        ]
        loan_profiles_2d = batch[loan_profiles_cols].values.tolist()

        insert_loan_profiles = f"""
            INSERT INTO loan_profiles ({", ".join(loan_profiles_cols)})
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT DO NOTHING
        """

        cursor.executemany(insert_loan_profiles, loan_profiles_2d)

        select_loan_profile_keys = """
            WITH t AS (
                SELECT unnest($1) AS loan_number, unnest($2) AS rev_line_of_credit,
                    unnest($3) AS low_doc, unnest($4) AS status,
                    unnest($5) AS real_estate_backed, unnest($6) AS recession,
                    unnest($7) AS in_bank_state, unnest($8) AS full_amount_disbursed,
                    unnest($9) AS is_existing
            )
            SELECT loan_profile_key, loan_number
            FROM t
            JOIN loan_profiles lp ON (
                lp.rev_line_of_credit = t.rev_line_of_credit
                AND lp.low_doc = t.low_doc
                AND lp.status = t.status
                AND lp.real_estate_backed = t.real_estate_backed
                AND lp.recession = t.recession
                AND lp.in_bank_state = t.in_bank_state
                AND lp.full_amount_disbursed = t.full_amount_disbursed
                AND lp.is_existing = t.is_existing
            )
        """

        lp_rs = cursor.execute(select_loan_profile_keys, loan_profiles).df()
        lp_kv = lp_rs.set_index("loan_number")["loan_profile_key"].to_dict()
        lp_keys = batch["loan_number"].map(lp_kv)

        # Loans
        loans_2d = pd.concat(
            (
                borrower_keys,
                bank_keys,
                approval_date_keys,
                charged_date_keys.replace({np.nan: None}),
                dis_date_keys,
                maturity_date_keys,
                lp_keys,
                batch[["loan_number", "term", "term_in_days", "disbursement_gross"]],
                batch[["balance_gross", "charged_off_amount", "approved_gross"]],
                batch[["sba_guaranteed_amount", "days_to_disbursement"]],
                batch[["sba_guaranteed_ratio"]],
            ),
            axis=1,
        ).values.tolist()

        insert_loans = f"""
            INSERT INTO loans ({", ".join(loans_cols)})
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                $16, $17)
        """

        cursor.executemany(insert_loans, loans_2d)
        cursor.commit()


# %%
# load()

# %% [markdown]
# ## Usage

# %% [markdown]
# ### The Number of Rows in All Tables

# %%
# %%time

query = """
    WITH dimension_1 AS (
        SELECT 'dimension' AS types, 'states' AS tables, COUNT(*) AS rows
        FROM states
    ),
    dimension_2 AS (
        SELECT 'dimension' AS types, 'borrowers' AS tables, COUNT(*) AS rows
        FROM borrowers
    ),
    dimension_3 AS (
        SELECT 'dimension' AS types, 'banks' AS tables, COUNT(*) AS rows
        FROM banks
    ),
    dimension_4 AS (
        SELECT 'dimension' AS types, 'dates' AS tables, COUNT(*) AS rows
        FROM dates
    ),
    dimension_5 AS (
        SELECT 'dimension' AS types, 'loan_profiles' AS tables, COUNT(*) AS rows
        FROM loan_profiles
    ),
    fact_table AS (
        SELECT 'fact_table' AS types, 'loans' AS tables, COUNT(*) AS rows
        FROM loans
    )
    SELECT * FROM dimension_1
    UNION
    SELECT * FROM dimension_2
    UNION
    SELECT * FROM dimension_3
    UNION
    SELECT * FROM dimension_4
    UNION
    SELECT * FROM dimension_5
    UNION
    SELECT * FROM fact_table
    ORDER BY rows ASC
"""

cursor.execute(query).df()

# %% [markdown]
# ### Overview of Loan Applications and Approved Amounts

# %%
# # %%time

query = """
    SELECT
        industry,
        COUNT(l.*) AS n_applications,
        SUM(CASE WHEN status THEN 1 ELSE 0 END) AS n_charged_offs,
        AVG(term),
        AVG(term_in_days),
        AVG(days_to_disbursement),
        AVG(charged_off_amount),
        AVG(approved_gross),
        AVG(sba_guaranteed_amount),
        AVG(sba_guaranteed_ratio)
    FROM loans l
    JOIN borrowers b ON b.borrower_key = l.borrower_key
    JOIN loan_profiles lp ON lp.loan_profile_key = l.loan_profile_key
    GROUP BY industry
    ORDER BY industry
"""

show(cursor.execute(query).df(), scrollX=True)

# %% [markdown]
# ### A Brief Comparative Analysis of Bank Loan Metrics

# %%
# # %%time

query = """
    SELECT
        b.name,
        bs.name AS state,
        MIN(term_in_days),
        MAX(term_in_days),
        AVG(term_in_days),
        MIN(approved_gross),
        MAX(approved_gross),
        AVG(approved_gross),
        AVG(days_to_disbursement),
        AVG(sba_guaranteed_amount),
        AVG(sba_guaranteed_ratio)
    FROM loans l
    JOIN banks b ON b.bank_key = l.bank_key
    JOIN states bs ON bs.state_key = b.state_key
    GROUP BY b.name, bs.name
    ORDER BY (0.5 * AVG(term_in_days) + 0.5 * AVG(approved_gross)) DESC
    LIMIT 500
"""

show(cursor.execute(query).df(), scrollX=True)

# %%
