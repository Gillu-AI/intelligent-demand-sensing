

import pandas as pd

import re
 
# -----------------------------

# Helpers for schema normalization

# -----------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

    """

    Make all column names lower-case snake_case.

    Examples:

      'Date' -> 'date'

      'Festival Name' -> 'festival_name'

      'weekday/weekend' -> 'weekday_weekend'

    """

    def to_snake(s: str) -> str:

        s = s.strip()

        s = s.lower()

        s = s.replace('/', '_').replace('\\', '_')

        s = re.sub(r'[^0-9a-zA-Z_]+', '_', s)     # replace non-alnum with _

        s = re.sub(r'_+', '_', s).strip('_')      # collapse repeats

        return s

    df = df.copy()

    df.columns = [to_snake(c) for c in df.columns]

    return df
 
def safe_read(path: str) -> pd.DataFrame:

    """Read CSV or Excel based on extension."""

    if path.lower().endswith(('.xls', '.xlsx')):

        return pd.read_excel(path)

    return pd.read_csv(path)
 
# -----------------------------

# SALES

# -----------------------------

def read_sales(path: str) -> pd.DataFrame:

    """

    Read daily sales from CSV or Excel, normalize columns,

    and validate the required schema.

    Expected (after mapping):

        date, sales_open, sales_closed, total_sales

    """

    df = safe_read(path)

    df = normalize_columns(df)
 
    # Allow common real-world header variations via a mapping

    # (add entries here if your file uses different names)

    rename_map = {

        'salesopen': 'sales_open',

        'sales_open': 'sales_open',

        'open_sales': 'sales_open',
 
        'salesclosed': 'sales_closed',

        'sales_closed': 'sales_closed',

        'closed_sales': 'sales_closed',
 
        'totalsales': 'total_sales',

        'total_sales': 'total_sales',
 
        'date': 'date',

        'transaction_date': 'date'

    }

    # Apply mapping only for keys that exist

    present_keys = {k: v for k, v in rename_map.items() if k in df.columns}

    df = df.rename(columns=present_keys)
 
    expected = {'date', 'sales_open', 'sales_closed', 'total_sales'}

    missing = expected - set(df.columns)

    if missing:

        # Helpfully report what we saw so the user can extend the mapping

        raise ValueError(

            "[read_sales] Missing columns after normalization/rename: "

            f"{missing}. Columns seen: {sorted(df.columns)}. "

            "If your file uses different header names, add them to rename_map in read_sales()."

        )
 
    # Parse and validate dates

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if df['date'].isna().any():

        bad = df[df['date'].isna()]

        raise ValueError(f"[read_sales] Found unparsable dates:\n{bad.head()}")
 
    df = df.sort_values('date').reset_index(drop=True)

    return df
 
# -----------------------------

# CALENDAR

# -----------------------------

def read_calendar(path: str) -> pd.DataFrame:

    """

    Read festival/holiday calendar, normalize headers, map to expected names,

    then validate. Expected (after mapping):

        date, day, is_festival, festival_name, is_holiday, weekday_weekend

    """

    cal = safe_read(path)

    cal = normalize_columns(cal)
 
    # Map your real-world headers to our expected schema.

    # Add to this map if your file has different names.

    rename_map = {

        'date': 'date',
 
        # 'day' present as 'weekday' or 'day_of_week'

        'day': 'day',

        'weekday': 'day',

        'day_of_week': 'day',
 
        'is_festival': 'is_festival',

        'festival': 'is_festival',      # e.g., 1/0 or Yes/No

        'festival_flag': 'is_festival',
 
        'festival_name': 'festival_name',

        'festivalname': 'festival_name',

        'fest_name': 'festival_name',
 
        # Your example: "Holiday" -> we want 'is_holiday'

        'holiday': 'is_holiday',

        'is_holiday': 'is_holiday',

        'holiday_flag': 'is_holiday',
 
        'weekday_weekend': 'weekday_weekend',

        'weekend_weekday': 'weekday_weekend',

        'is_weekend': 'weekday_weekend'   # we can derive later, but accept it here

    }

    present_keys = {k: v for k, v in rename_map.items() if k in cal.columns}

    cal = cal.rename(columns=present_keys)
 
    expected = {'date', 'day', 'is_festival', 'festival_name', 'is_holiday', 'weekday_weekend'}

    missing = expected - set(cal.columns)

    if missing:

        raise ValueError(

            "[read_calendar] Missing columns after normalization/rename: "

            f"{missing}. Columns seen: {sorted(cal.columns)}. "

            "Update rename_map in read_calendar() to include your header names "

            "(e.g., 'Holiday' -> 'is_holiday')."

        )
 
    # Parse and validate dates

    cal['date'] = pd.to_datetime(cal['date'], errors='coerce')

    if cal['date'].isna().any():

        bad = cal[cal['date'].isna()]

        raise ValueError(f"[read_calendar] Found unparsable dates:\n{bad.head()}")
 
    # Enforce uniqueness on date (one row per day)

    dups = cal['date'].duplicated().sum()

    if dups > 0:

        raise ValueError(f"[read_calendar] Calendar has {dups} duplicate date rows. Ensure one row per date.")
 
    cal = cal.sort_values('date').reset_index(drop=True)

    return cal
 
# -----------------------------

# JOIN

# -----------------------------

def join_sales_calendar(sales: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:

    """

    Left join sales with calendar on 'date'. Validate one_to_one to avoid duplication.

    """

    df = sales.merge(cal, on='date', how='left', validate='one_to_one')

    missing_calendar = df['day'].isna().sum()

    if missing_calendar > 0:

        print(f"[join_sales_calendar] WARNING: {missing_calendar} rows have no calendar info.")

    return df

 