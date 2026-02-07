"""
Timezone helpers with validation.
Store everything internally as UTC ISO8601, display in a selected TZ.

Uses standard library zoneinfo on Python 3.11+.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


# Known valid timezones (fast lookup, matches _common_timezones() in app.py)
_COMMON_TIMEZONES = frozenset([
    "UTC", "Europe/London", "Europe/Dublin", "Europe/Paris", "Europe/Rome",
    "Europe/Berlin", "America/New_York", "America/Chicago", "America/Denver",
    "America/Los_Angeles", "America/Toronto", "America/Sao_Paulo",
    "Asia/Dubai", "Asia/Kolkata", "Asia/Singapore", "Asia/Tokyo", "Australia/Sydney",
])


# Timezone abbreviation to IANA mapping
# Note: Some abbreviations are ambiguous (CST = US Central, China, Cuba; IST = India, Ireland, Israel)
# Defaults are based on typical usage patterns in interview scheduling contexts
_TZ_ABBREVIATION_MAP: Dict[str, str] = {
    # US Standard Time
    "PST": "America/Los_Angeles",
    "MST": "America/Denver",
    "CST": "America/Chicago",
    "EST": "America/New_York",
    # US Daylight Time
    "PDT": "America/Los_Angeles",
    "MDT": "America/Denver",
    "CDT": "America/Chicago",
    "EDT": "America/New_York",
    # UTC variants
    "UTC": "UTC",
    "GMT": "UTC",
    "Z": "UTC",
    # European
    "WET": "Europe/London",
    "WEST": "Europe/London",
    "BST": "Europe/London",  # British Summer Time
    "CET": "Europe/Paris",
    "CEST": "Europe/Paris",
    "EET": "Europe/Helsinki",
    "EEST": "Europe/Helsinki",
    "IST": "Europe/Dublin",  # Irish Standard Time (ambiguous - default to Ireland)
    # Asia/Pacific
    "JST": "Asia/Tokyo",
    "KST": "Asia/Seoul",
    "SGT": "Asia/Singapore",
    "HKT": "Asia/Hong_Kong",
    "IST_INDIA": "Asia/Kolkata",  # Explicit India variant
    "AEST": "Australia/Sydney",
    "AEDT": "Australia/Sydney",
    "AWST": "Australia/Perth",
    # South America
    "BRT": "America/Sao_Paulo",
    "BRST": "America/Sao_Paulo",
}


def infer_timezone_from_abbreviation(
    abbrev: str,
    default: str = "UTC"
) -> Tuple[str, bool, Optional[str]]:
    """
    Map timezone abbreviation to IANA timezone.

    Args:
        abbrev: Timezone abbreviation (e.g., "PST", "EST", "GMT")
        default: Default timezone if abbreviation is unknown

    Returns:
        Tuple of (iana_timezone, was_matched, ambiguity_note)
        - iana_timezone: The IANA timezone name
        - was_matched: True if abbreviation was found in mapping
        - ambiguity_note: Optional note about ambiguous abbreviations
    """
    if not abbrev:
        return default, False, None

    normalized = abbrev.strip().upper().replace(" ", "")

    # Direct lookup
    if normalized in _TZ_ABBREVIATION_MAP:
        ambiguity = None
        if normalized == "CST":
            ambiguity = "'CST' is ambiguous; defaulting to US Central Time"
        elif normalized == "IST":
            ambiguity = "'IST' is ambiguous; defaulting to Irish Standard Time. Use 'IST_INDIA' for India."
        return _TZ_ABBREVIATION_MAP[normalized], True, ambiguity

    # Try to match partial keywords (e.g., "Pacific" -> PST)
    partial_map = {
        "PACIFIC": "America/Los_Angeles",
        "MOUNTAIN": "America/Denver",
        "CENTRAL": "America/Chicago",
        "EASTERN": "America/New_York",
    }
    for key, tz in partial_map.items():
        if key in normalized:
            return tz, True, f"Inferred '{tz}' from partial match '{abbrev}'"

    return default, False, f"Unknown timezone abbreviation: {abbrev}"


def is_valid_timezone(tz_name: str) -> bool:
    """
    Check if timezone name is valid without throwing.
    Returns False for None, empty string, or invalid timezone names.
    """
    if not tz_name or not isinstance(tz_name, str):
        return False
    # Fast path for common timezones
    if tz_name in _COMMON_TIMEZONES:
        return True
    # Try to construct it for less common timezones
    try:
        ZoneInfo(tz_name)
        return True
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        return False


def safe_zoneinfo(tz_name: str, fallback: str = "UTC") -> Tuple[ZoneInfo, bool]:
    """
    Get ZoneInfo safely with fallback.
    Returns (ZoneInfo, was_valid) tuple.

    If tz_name is invalid, returns (ZoneInfo(fallback), False).
    """
    if is_valid_timezone(tz_name):
        return ZoneInfo(tz_name), True
    return ZoneInfo(fallback), False


def to_utc(dt_local: datetime) -> datetime:
    """
    Convert timezone-aware datetime to UTC.
    Raises ValueError if datetime is naive (no timezone info).
    """
    if dt_local.tzinfo is None:
        raise ValueError("dt_local must be timezone-aware")
    return dt_local.astimezone(timezone.utc)


def from_utc(dt_utc: datetime, tz_name: str) -> datetime:
    """
    Convert UTC datetime to local timezone.
    Raises ValueError if timezone is invalid.
    """
    if not is_valid_timezone(tz_name):
        raise ValueError(f"Invalid timezone: {tz_name}")
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(ZoneInfo(tz_name))


def iso_utc(dt_utc: datetime) -> str:
    """
    Format datetime as ISO8601 UTC string.
    Assumes naive datetime is UTC (for backwards compatibility).
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso(dt_str: str) -> datetime:
    """
    Parse ISO datetime string.
    Raises ValueError if format is invalid.
    """
    # Handle 'Z' suffix (ISO standard for UTC) which fromisoformat doesn't support
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1] + '+00:00'
    return datetime.fromisoformat(dt_str)


# ----------------------------
# Display Formatting
# ----------------------------
def format_time_for_display(
    dt_utc: datetime,
    tz_name: str,
    include_tz_abbrev: bool = True
) -> str:
    """
    Format UTC datetime for human-readable display in specified timezone.

    Args:
        dt_utc: UTC datetime
        tz_name: IANA timezone name
        include_tz_abbrev: Whether to include timezone abbreviation

    Returns:
        Formatted string like "2:00 PM PST" or "2:00 PM"
    """
    local_dt = from_utc(dt_utc, tz_name)
    time_str = local_dt.strftime("%I:%M %p").lstrip("0")  # "2:00 PM"

    if include_tz_abbrev:
        # Get timezone abbreviation (handles DST automatically)
        tz_abbrev = local_dt.strftime("%Z")
        if tz_abbrev:
            return f"{time_str} {tz_abbrev}"

    return time_str


def format_datetime_for_display(
    dt_utc: datetime,
    tz_name: str,
    include_tz_abbrev: bool = True
) -> str:
    """
    Format UTC datetime with full date for human-readable display.

    Args:
        dt_utc: UTC datetime
        tz_name: IANA timezone name
        include_tz_abbrev: Whether to include timezone abbreviation

    Returns:
        Formatted string like "Monday, January 15, 2025 at 2:00 PM PST"
    """
    local_dt = from_utc(dt_utc, tz_name)
    date_str = local_dt.strftime("%A, %B %d, %Y")
    time_str = format_time_for_display(dt_utc, tz_name, include_tz_abbrev)
    return f"{date_str} at {time_str}"


# ----------------------------
# DST Detection
# ----------------------------
def is_dst_active(dt: datetime, tz_name: str) -> bool:
    """
    Check if DST is active for a given datetime in a timezone.

    Args:
        dt: The datetime to check (can be naive or aware)
        tz_name: IANA timezone name

    Returns:
        True if DST is active, False otherwise
    """
    if not is_valid_timezone(tz_name):
        return False

    zi = ZoneInfo(tz_name)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=zi)
    else:
        dt = dt.astimezone(zi)

    # dst() returns timedelta if DST is active, timedelta(0) if not
    dst_offset = dt.dst()
    return dst_offset is not None and dst_offset.total_seconds() > 0


def is_dst_transition_day(check_date: date, tz_name: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a date is a DST transition day for a timezone.

    Args:
        check_date: The date to check
        tz_name: IANA timezone name

    Returns:
        Tuple of (is_transition_day, transition_type)
        - is_transition_day: True if DST changes on this day
        - transition_type: "spring_forward" or "fall_back" or None
    """
    if not is_valid_timezone(tz_name):
        return False, None

    zi = ZoneInfo(tz_name)

    # Check DST status at start and end of day
    try:
        start_of_day = datetime(check_date.year, check_date.month, check_date.day, 0, 30, tzinfo=zi)
        end_of_day = datetime(check_date.year, check_date.month, check_date.day, 23, 30, tzinfo=zi)

        dst_start = start_of_day.dst()
        dst_end = end_of_day.dst()

        if dst_start is None or dst_end is None:
            return False, None

        if dst_start != dst_end:
            if dst_end > dst_start:
                return True, "spring_forward"
            else:
                return True, "fall_back"
    except (ValueError, OverflowError):
        # Handle edge cases with invalid dates
        return False, None

    return False, None


def get_dst_transition_dates(year: int, tz_name: str) -> List[date]:
    """
    Get DST transition dates for a timezone in a given year.

    Returns list of dates when DST changes (typically 0 or 2 per year).
    """
    if not is_valid_timezone(tz_name):
        return []

    transitions = []
    zi = ZoneInfo(tz_name)

    # Check each day of the year
    current = date(year, 1, 1)
    try:
        prev_dst = datetime(year, 1, 1, 12, 0, tzinfo=zi).dst()
    except (ValueError, OverflowError):
        return []

    while current.year == year:
        try:
            current_dst = datetime(current.year, current.month, current.day, 12, 0, tzinfo=zi).dst()
            if current_dst != prev_dst:
                transitions.append(current)
                prev_dst = current_dst
        except (ValueError, OverflowError):
            pass
        current += timedelta(days=1)

    return transitions


def is_near_dst_transition(
    dt: datetime,
    tz_name: str,
    days_threshold: int = 7
) -> Tuple[bool, Optional[date], Optional[str]]:
    """
    Check if a datetime is within threshold days of a DST transition.

    Args:
        dt: Datetime to check
        tz_name: IANA timezone name
        days_threshold: Number of days to consider "near"

    Returns:
        Tuple of (is_near, transition_date, transition_type)
    """
    if not is_valid_timezone(tz_name):
        return False, None, None

    check_date = dt.date() if isinstance(dt, datetime) else dt
    year = check_date.year

    # Get transitions for this year and next (in case we're near year end)
    transitions = get_dst_transition_dates(year, tz_name)
    if check_date.month >= 11:
        transitions.extend(get_dst_transition_dates(year + 1, tz_name))

    for trans_date in transitions:
        days_diff = (trans_date - check_date).days
        if 0 <= days_diff <= days_threshold:
            _, trans_type = is_dst_transition_day(trans_date, tz_name)
            return True, trans_date, trans_type

    return False, None, None


def format_time_with_dst_info(
    dt_utc: datetime,
    tz_name: str
) -> Tuple[str, Optional[str]]:
    """
    Format time with DST indicator if relevant.

    Returns:
        Tuple of (formatted_time, dst_warning)
        - formatted_time: Human-readable time string
        - dst_warning: Warning message if near DST transition, else None
    """
    formatted = format_datetime_for_display(dt_utc, tz_name)

    is_near, trans_date, trans_type = is_near_dst_transition(dt_utc, tz_name)

    if is_near and trans_date:
        if trans_type == "spring_forward":
            warning = f"Note: Clocks spring forward on {trans_date.strftime('%B %d')} in this timezone"
        elif trans_type == "fall_back":
            warning = f"Note: Clocks fall back on {trans_date.strftime('%B %d')} in this timezone"
        else:
            warning = f"Note: DST transition occurs on {trans_date.strftime('%B %d')} in this timezone"
        return formatted, warning

    return formatted, None
