"""
Availability intersection algorithm for panel interviews.

Handles:
- Single interviewer pass-through
- Multi-interviewer intersection
- Adjacent slot merging
- Partial overlap tracking
- Minimum duration filtering
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TimePoint:
    """Represents a point in time with interviewer availability change."""

    dt: datetime
    interviewer_id: int
    is_start: bool  # True = start of availability, False = end


def normalize_slots_to_utc(
    slots: List[Dict[str, str]], source_timezone: str
) -> List[Tuple[datetime, datetime]]:
    """
    Convert slots to UTC datetime tuples for comparison.

    Args:
        slots: List of slot dicts with 'date', 'start', 'end' keys
        source_timezone: IANA timezone name for the slots

    Returns:
        List of (start_utc, end_utc) datetime tuples
    """
    from timezone_utils import safe_zoneinfo, to_utc

    normalized = []
    zi, _ = safe_zoneinfo(source_timezone, fallback="UTC")

    for slot in slots:
        try:
            start_naive = datetime.strptime(
                f"{slot['date']}T{slot['start']}", "%Y-%m-%dT%H:%M"
            )
            end_naive = datetime.strptime(
                f"{slot['date']}T{slot['end']}", "%Y-%m-%dT%H:%M"
            )
            start_local = start_naive.replace(tzinfo=zi)
            end_local = end_naive.replace(tzinfo=zi)
            normalized.append((to_utc(start_local), to_utc(end_local)))
        except (ValueError, KeyError):
            continue

    return normalized


def merge_adjacent_slots(
    slots: List[Tuple[datetime, datetime]], gap_tolerance_minutes: int = 0
) -> List[Tuple[datetime, datetime]]:
    """
    Merge adjacent or overlapping time slots.

    Args:
        slots: List of (start, end) datetime tuples
        gap_tolerance_minutes: Maximum gap between slots to still merge them

    Returns:
        List of merged (start, end) datetime tuples
    """
    if not slots:
        return []

    sorted_slots = sorted(slots, key=lambda x: x[0])
    merged = [sorted_slots[0]]

    for start, end in sorted_slots[1:]:
        last_start, last_end = merged[-1]
        gap = (start - last_end).total_seconds() / 60

        if gap <= gap_tolerance_minutes:
            # Merge with previous
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def compute_intersection(
    interviewer_slots: Dict[int, List[Tuple[datetime, datetime]]],
    min_duration_minutes: int = 30,
    display_timezone: str = "UTC",
    interviewer_names: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute availability intersection across all interviewers.

    Uses a sweep line algorithm:
    1. Create time points for all slot starts and ends
    2. Sweep through time, tracking which interviewers are available
    3. Record intervals where availability changes

    Args:
        interviewer_slots: Dict mapping interviewer_id to list of (start, end) UTC datetimes
        min_duration_minutes: Minimum slot duration to include
        display_timezone: Timezone for output formatting
        interviewer_names: Optional dict mapping interviewer_id to display name

    Returns:
        List of intersection slot dicts with availability metadata
    """
    from timezone_utils import from_utc

    if not interviewer_slots:
        return []

    total_interviewers = len(interviewer_slots)
    interviewer_names = interviewer_names or {}

    # Single interviewer: pass-through with full availability markers
    if total_interviewers == 1:
        interviewer_id = list(interviewer_slots.keys())[0]
        slots = interviewer_slots[interviewer_id]
        return [
            _format_intersection_slot(
                start,
                end,
                [interviewer_id],
                1,
                1,
                display_timezone,
                interviewer_names,
            )
            for start, end in slots
            if (end - start).total_seconds() >= min_duration_minutes * 60
        ]

    # Build time points for sweep line
    time_points: List[TimePoint] = []
    for interviewer_id, slots in interviewer_slots.items():
        for start, end in slots:
            time_points.append(TimePoint(start, interviewer_id, True))
            time_points.append(TimePoint(end, interviewer_id, False))

    # Sort: by time, then ends before starts at same time (is_start=False < is_start=True)
    time_points.sort(key=lambda p: (p.dt, p.is_start))

    # Sweep through time points
    result_slots = []
    current_available: Set[int] = set()
    prev_time: Optional[datetime] = None

    for point in time_points:
        if prev_time is not None and current_available and point.dt > prev_time:
            duration = (point.dt - prev_time).total_seconds() / 60
            if duration >= min_duration_minutes:
                result_slots.append(
                    _format_intersection_slot(
                        prev_time,
                        point.dt,
                        list(current_available),
                        len(current_available),
                        total_interviewers,
                        display_timezone,
                        interviewer_names,
                    )
                )

        # Update availability
        if point.is_start:
            current_available.add(point.interviewer_id)
        else:
            current_available.discard(point.interviewer_id)

        prev_time = point.dt

    return result_slots


def _format_intersection_slot(
    start_utc: datetime,
    end_utc: datetime,
    available_ids: List[int],
    available_count: int,
    total_count: int,
    display_timezone: str,
    interviewer_names: Dict[int, str],
) -> Dict[str, Any]:
    """Format an intersection slot for display."""
    from timezone_utils import from_utc

    start_local = from_utc(start_utc, display_timezone)
    end_local = from_utc(end_utc, display_timezone)

    # Get names of available interviewers
    available_names = [
        interviewer_names.get(id_, f"Interviewer {id_}") for id_ in available_ids
    ]

    return {
        "date": start_local.strftime("%Y-%m-%d"),
        "start": start_local.strftime("%H:%M"),
        "end": end_local.strftime("%H:%M"),
        "inferred_tz": None,
        "available_interviewers": available_ids,
        "available_names": available_names,
        "available_count": available_count,
        "total_interviewers": total_count,
        "is_full_overlap": available_count == total_count,
    }


def filter_slots_by_availability(
    slots: List[Dict[str, Any]],
    mode: str,
    min_n: int = 1,
    interviewer_count: int = 1,
) -> List[Dict[str, Any]]:
    """
    Filter intersection slots by availability mode.

    Args:
        slots: List of intersection slot dicts
        mode: "all_available" | "any_n" | "show_all"
        min_n: Minimum interviewers for "any_n" mode
        interviewer_count: Total number of interviewers

    Returns:
        Filtered list of slots
    """
    if mode == "show_all":
        return slots
    elif mode == "all_available":
        return [s for s in slots if s.get("is_full_overlap", True)]
    elif mode == "any_n":
        return [s for s in slots if s.get("available_count", 1) >= min_n]
    return slots


def format_slot_label_with_availability(
    slot: Dict[str, Any], total_interviewers: int
) -> str:
    """
    Format a slot label with availability information.

    Args:
        slot: Intersection slot dict
        total_interviewers: Total number of interviewers

    Returns:
        Formatted label string like "Dec 3 - 2:00 PM to 3:00 PM (3/3 available)"
    """
    # Parse date for friendly display
    try:
        date_obj = datetime.strptime(slot["date"], "%Y-%m-%d")
        date_str = date_obj.strftime("%b %d")
    except ValueError:
        date_str = slot["date"]

    # Format times
    try:
        start_obj = datetime.strptime(slot["start"], "%H:%M")
        end_obj = datetime.strptime(slot["end"], "%H:%M")
        start_str = start_obj.strftime("%I:%M %p").lstrip("0")
        end_str = end_obj.strftime("%I:%M %p").lstrip("0")
    except ValueError:
        start_str = slot["start"]
        end_str = slot["end"]

    base_label = f"{date_str} - {start_str} to {end_str}"

    # Add availability info for panel interviews
    if total_interviewers > 1:
        avail = slot.get("available_count", total_interviewers)
        total = slot.get("total_interviewers", total_interviewers)
        if avail == total:
            return f"{base_label} (All {total} available)"
        else:
            names = slot.get("available_names", [])
            if names:
                names_str = ", ".join(names[:2])
                if len(names) > 2:
                    names_str += f" +{len(names) - 2}"
                return f"{base_label} ({avail}/{total}: {names_str})"
            return f"{base_label} ({avail}/{total} available)"

    return base_label
