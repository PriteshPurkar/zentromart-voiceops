"""
Date and business-day logic for the ZentroMart dataset.
TODAY is fixed at 20 October 2026.
"""
from datetime import date, datetime, timedelta
from src.config import TODAY, BUSINESS_WEEKDAYS
from src.data.loader import get_holidays

def parse_date(date_str: str | None) -> date | None:
    """Convert 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM' string to a date object."""
    if not date_str:
        return None
    # Handle timestamps
    if "T" in str(date_str):
        date_str = str(date_str).split("T")[0]
    return datetime.strptime(str(date_str), "%Y-%m-%d").date()

def _get_holiday_dates() -> set[date]:
    """Fetch the set of holiday dates from the CSV."""
    df = get_holidays()
    return {parse_date(d) for d in df["date"].tolist() if d}

def is_business_day(d: date, holidays: set[date]) -> bool:
    """A business day is Mon-Sat, excluding holidays."""
    return d.weekday() in BUSINESS_WEEKDAYS and d not in holidays

def add_business_days(start_date: date, days: int) -> date:
    """Add 'days' business days to 'start_date'."""
    holidays = _get_holiday_dates()
    current = start_date
    added = 0
    
    while added < days:
        current += timedelta(days=1)
        if is_business_day(current, holidays):
            added += 1
            
    return current

def count_business_days(start_date: date, end_date: date) -> int:
    """Count how many business days are between two dates (inclusive of end, exclusive of start)."""
    if start_date >= end_date:
        return 0
    
    holidays = _get_holiday_dates()
    current = start_date
    count = 0
    
    while current < end_date:
        current += timedelta(days=1)
        if is_business_day(current, holidays):
            count += 1
            
    return count