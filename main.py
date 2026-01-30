

# --- path safety (only if you ever see ModuleNotFoundError) ---

# import sys, os

# ROOT = os.path.dirname(os.path.abspath(__file__))

# if ROOT not in sys.path:

#     sys.path.append(ROOT)

# ---------------------------------------------------------------
 
from src.ingestion_01.io_data import read_sales, read_calendar, join_sales_calendar
 
def main():

    # Raw File datapath

    sales_path = r"C:\Users\karthigeyan.sivasamy\source\repos\dailyforcast_whatif\data\raw\Sales_data.xlsx"

    cal_path   = r"C:\Users\karthigeyan.sivasamy\source\repos\dailyforcast_whatif\data\raw\india_calendar_festivals_2020_2030.xlsx"
 
    # Print function names.

    print("If files exist, we will load and join them...")

    try:

        sales = read_sales(sales_path)

        cal   = read_calendar(cal_path)

        df    = join_sales_calendar(sales, cal)

        print(df.head())

        print("Rows:", len(df))

    except FileNotFoundError as e:

        print("File not found. Place files under data/raw and re-run.")

        print(e)

    except Exception as e:

        print("Error during ingestion:")

        print(e)
 
if __name__ == "__main__":

    main()

 