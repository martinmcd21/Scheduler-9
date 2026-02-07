"""
Export utilities for PowerDash Interview Scheduler.

Provides CSV export functionality for interviews and audit logs,
plus human-readable formatting for audit entries.
"""

import csv
import io
import json
from datetime import datetime, date, timedelta, timezone as tz
from typing import List, Dict, Any, Optional

from timezone_utils import from_utc


# ---------------------------------------------------------------------------
# Human-readable action descriptions for audit log
# ---------------------------------------------------------------------------

AUDIT_ACTION_DESCRIPTIONS: Dict[str, str] = {
    # Interview lifecycle
    "graph_create_event": "Interview scheduled",
    "graph_create_failed": "Failed to schedule interview",
    "graph_create_group_event": "Group interview scheduled",
    "graph_create_group_failed": "Failed to schedule group interview",
    "graph_reschedule_event": "Interview rescheduled",
    "graph_reschedule_failed": "Failed to reschedule interview",
    "graph_cancel_event": "Interview cancelled",
    "interview_cancelled": "Interview cancelled",
    "graph_cancel_failed": "Failed to cancel interview",
    "interview_rescheduled": "Interview rescheduled",
    "interview_reschedule_failed": "Failed to reschedule interview",
    # Email actions
    "email_sent": "Email sent",
    "email_send_failed": "Failed to send email",
    "candidate_notification_sent": "Candidate notified",
    "cancellation_email_sent": "Cancellation notice sent",
    "reschedule_email_sent": "Reschedule notice sent",
    "graph_sent_scheduling_email": "Scheduling email sent",
    "graph_send_failed": "Email send failed",
    "graph_sent_ics": "Calendar invite sent",
    # Parsing actions
    "parse_slots_openai": "Parsed availability from upload",
    "parse_slots_text_openai": "Parsed availability from text",
    # Status changes
    "status_updated": "Status updated",
    "interview_confirmed": "Interview confirmed",
    # ICS actions
    "ics_generated": "Calendar invite generated",
    "ics_downloaded": "Calendar file downloaded",
    # System actions
    "graph_token_refresh": "Authentication refreshed",
    "graph_token_ok": "Authentication successful",
    "graph_token_failed": "Authentication failed",
    "graph_calendar_read_ok": "Calendar read successful",
    "graph_calendar_read_failed": "Calendar read failed",
    "graph_dummy_event_ok": "Test event created",
    "graph_dummy_event_failed": "Test event failed",
    "db_migration": "Database updated",
}


# ---------------------------------------------------------------------------
# Interview CSV Export
# ---------------------------------------------------------------------------

def export_interviews_csv(
    interviews: List[Dict[str, Any]],
    display_timezone: str = "UTC",
    include_all_fields: bool = False,
) -> bytes:
    """
    Export interviews to CSV format.

    Args:
        interviews: List of interview records from database
        display_timezone: Timezone for date/time formatting
        include_all_fields: If True, include all fields; if False, only core fields

    Returns:
        CSV file content as bytes (UTF-8 with BOM for Excel compatibility)
    """
    output = io.StringIO()

    # Define field mappings (db_field -> csv_header)
    core_fields = [
        ("role_title", "Role"),
        ("candidate_display", "Candidate"),
        ("interviewer_display", "Interviewer(s)"),
        ("interview_date", "Date"),
        ("interview_time", "Time"),
        ("duration_minutes", "Duration (min)"),
        ("last_status", "Status"),
        ("interview_type", "Type"),
    ]

    extended_fields = [
        ("hiring_manager_email", "Hiring Manager"),
        ("recruiter_email", "Recruiter"),
        ("teams_join_url", "Teams Link"),
        ("created_utc", "Created"),
        ("graph_event_id", "Event ID"),
    ]

    fields = core_fields + (extended_fields if include_all_fields else [])
    headers = [f[1] for f in fields]

    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()

    for interview in interviews:
        row = _format_interview_for_csv(interview, display_timezone)
        # Map to headers
        csv_row = {header: row.get(db_field, "") for db_field, header in fields}
        writer.writerow(csv_row)

    # Return with UTF-8 BOM for Excel compatibility
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def _format_interview_for_csv(
    interview: Dict[str, Any],
    display_timezone: str,
) -> Dict[str, Any]:
    """Format interview record for CSV export."""
    formatted = dict(interview)

    # Format candidate(s) display
    formatted["candidate_display"] = _format_candidates_for_export(interview)

    # Format interviewer(s) display
    formatted["interviewer_display"] = _format_interviewers_for_export(interview)

    # Format date and time separately
    start_utc = interview.get("start_utc", "")
    if start_utc:
        try:
            dt = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
            local_dt = from_utc(dt, display_timezone)
            formatted["interview_date"] = local_dt.strftime("%Y-%m-%d")
            formatted["interview_time"] = local_dt.strftime("%H:%M")
        except (ValueError, TypeError):
            formatted["interview_date"] = start_utc[:10] if len(start_utc) >= 10 else ""
            formatted["interview_time"] = start_utc[11:16] if len(start_utc) >= 16 else ""

    # Format interview type
    if interview.get("is_group_interview"):
        formatted["interview_type"] = "Group"
    elif interview.get("is_panel_interview"):
        formatted["interview_type"] = "Panel"
    else:
        formatted["interview_type"] = "Individual"

    # Format status for readability
    status = interview.get("last_status", "pending")
    formatted["last_status"] = (status or "pending").capitalize()

    return formatted


def _format_candidates_for_export(interview: Dict[str, Any]) -> str:
    """Format candidate(s) for CSV cell."""
    # Check for multi-candidate JSON
    candidates_json = interview.get("candidates_json")
    if candidates_json:
        try:
            candidates = json.loads(candidates_json)
            if candidates:
                return "; ".join(
                    f"{c.get('name', '')} <{c.get('email', '')}>".strip()
                    if c.get("name")
                    else c.get("email", "")
                    for c in candidates
                )
        except (json.JSONDecodeError, TypeError):
            pass

    # Fall back to single candidate
    email = interview.get("candidate_email", "")
    name = interview.get("candidate_name", "")
    if name and email:
        return f"{name} <{email}>"
    return email or name or ""


def _format_interviewers_for_export(interview: Dict[str, Any]) -> str:
    """Format interviewer(s) for CSV cell."""
    # Check for panel interviewers JSON
    panel_json = interview.get("panel_interviewers_json")
    if panel_json:
        try:
            interviewers = json.loads(panel_json)
            if interviewers:
                return "; ".join(
                    f"{i.get('name', '')} <{i.get('email', '')}>".strip()
                    if i.get("name")
                    else i.get("email", "")
                    for i in interviewers
                )
        except (json.JSONDecodeError, TypeError):
            pass

    # Fall back to hiring manager as interviewer
    hm_email = interview.get("hiring_manager_email", "")
    hm_name = interview.get("hiring_manager_name", "")
    if hm_name and hm_email:
        return f"{hm_name} <{hm_email}>"
    return hm_email or ""


# ---------------------------------------------------------------------------
# Audit Log Human-Readable Formatting
# ---------------------------------------------------------------------------

def format_audit_entry_human(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw audit entry to human-readable format.

    Returns dict with:
    - timestamp: Formatted datetime
    - action_display: Human-readable action description
    - summary: One-line summary of what happened
    - actor_display: Who performed the action
    - status_badge: Success/Failed indicator
    - details: Formatted details if relevant
    - raw: Original entry for expandable view
    """
    action = entry.get("action", "unknown")
    status = entry.get("status", "")
    timestamp = entry.get("timestamp_utc", "")

    # Format timestamp
    formatted_time = ""
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            formatted_time = dt.strftime("%b %d, %Y at %I:%M %p UTC")
        except ValueError:
            formatted_time = timestamp

    # Get action description
    action_display = AUDIT_ACTION_DESCRIPTIONS.get(action, _humanize_action(action))

    # Build summary
    summary = _build_audit_summary(entry, action)

    # Format actor
    actor = entry.get("actor", "")
    actor_display = actor if actor else "System"

    # Status badge
    if status == "success":
        status_badge = "Success"
    elif status == "failed":
        status_badge = "Failed"
    else:
        status_badge = status.capitalize() if status else "Info"

    # Format details from payload
    details = _format_audit_details(entry)

    return {
        "timestamp": formatted_time,
        "action_display": action_display,
        "action_code": action,
        "summary": summary,
        "actor_display": actor_display,
        "status_badge": status_badge,
        "status": status,
        "details": details,
        "raw": entry,
    }


def _humanize_action(action: str) -> str:
    """Convert snake_case action to Title Case."""
    if not action:
        return "Unknown Action"
    return action.replace("_", " ").replace("graph ", "").title()


def _build_audit_summary(entry: Dict[str, Any], action: str) -> str:
    """Build one-line summary for audit entry."""
    candidate = entry.get("candidate_email", "")
    role = entry.get("role_title", "")
    actor = entry.get("actor", "System")

    # Custom summaries per action type
    if action == "graph_create_event":
        if candidate and role:
            return f"Scheduled interview for {candidate} ({role})"
        elif candidate:
            return f"Scheduled interview for {candidate}"
        return "Scheduled interview"

    elif action == "graph_create_group_event":
        if role:
            return f"Scheduled group interview for {role}"
        return "Scheduled group interview"

    elif action in ("graph_reschedule_event", "interview_rescheduled"):
        payload = _safe_json_loads(entry.get("payload_json", "{}"))
        new_time = payload.get("start", {}).get("dateTime", "")
        if new_time and candidate:
            return f"Rescheduled {candidate}'s interview to {new_time[:16]}"
        elif candidate:
            return f"Rescheduled interview for {candidate}"
        return "Rescheduled interview"

    elif action in ("graph_cancel_event", "interview_cancelled"):
        payload = _safe_json_loads(entry.get("payload_json", "{}"))
        reason = payload.get("reason", "")
        if reason and candidate:
            return f"Cancelled {candidate}'s interview: {reason}"
        elif candidate:
            return f"Cancelled interview for {candidate}"
        return "Cancelled interview"

    elif action == "email_sent":
        return f"Sent email to {candidate}" if candidate else "Sent email"

    elif action == "parse_slots_openai":
        payload = _safe_json_loads(entry.get("payload_json", "{}"))
        count = payload.get("slot_count", 0)
        return f"Extracted {count} time slot(s) from uploaded calendar"

    elif action == "parse_slots_text_openai":
        payload = _safe_json_loads(entry.get("payload_json", "{}"))
        count = payload.get("slot_count", 0)
        return f"Extracted {count} time slot(s) from text input"

    elif action == "graph_create_failed":
        error = entry.get("error_message", "Unknown error")
        if candidate:
            return f"Failed to schedule for {candidate}: {error[:50]}"
        return f"Failed to schedule: {error[:50]}"

    elif action == "ics_downloaded":
        return "Calendar invite file downloaded"

    # Default summary
    action_label = AUDIT_ACTION_DESCRIPTIONS.get(action, _humanize_action(action))
    if candidate and role:
        return f"{action_label} for {candidate} ({role})"
    elif candidate:
        return f"{action_label} for {candidate}"
    else:
        return action_label


def _format_audit_details(entry: Dict[str, Any]) -> Optional[str]:
    """Format relevant details from audit payload."""
    payload = _safe_json_loads(entry.get("payload_json", "{}"))
    if not payload:
        return None

    details = []

    # Extract relevant fields based on content
    if "start" in payload:
        start = payload["start"]
        if isinstance(start, dict):
            dt_str = start.get("dateTime", "")
            tz_str = start.get("timeZone", "")
            if dt_str:
                details.append(f"Time: {dt_str[:16]} ({tz_str})")

    if "reason" in payload:
        details.append(f"Reason: {payload['reason']}")

    if "notification_sent" in payload:
        details.append(f"Notification sent: {'Yes' if payload['notification_sent'] else 'No'}")

    if "slot_count" in payload:
        details.append(f"Slots extracted: {payload['slot_count']}")

    if "teams_join_url" in payload:
        details.append("Teams meeting created")

    return " | ".join(details) if details else None


def _safe_json_loads(json_str: str) -> Dict:
    """Safely parse JSON string."""
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Audit Log CSV Export
# ---------------------------------------------------------------------------

def export_audit_log_csv(entries: List[Dict[str, Any]]) -> bytes:
    """
    Export audit log to CSV format.

    Args:
        entries: List of formatted audit entries (from format_audit_entry_human)

    Returns:
        CSV file content as bytes (UTF-8 with BOM for Excel compatibility)
    """
    output = io.StringIO()

    headers = [
        "Timestamp",
        "Action",
        "Summary",
        "Status",
        "Actor",
        "Candidate",
        "Role",
        "Details",
        "Event ID",
    ]

    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()

    for entry in entries:
        raw = entry.get("raw", {})
        writer.writerow({
            "Timestamp": entry.get("timestamp", ""),
            "Action": entry.get("action_display", ""),
            "Summary": entry.get("summary", ""),
            "Status": entry.get("status", ""),
            "Actor": entry.get("actor_display", ""),
            "Candidate": raw.get("candidate_email", ""),
            "Role": raw.get("role_title", ""),
            "Details": entry.get("details", "") or "",
            "Event ID": raw.get("event_id", ""),
        })

    # Return with UTF-8 BOM for Excel compatibility
    return ("\ufeff" + output.getvalue()).encode("utf-8")


# ---------------------------------------------------------------------------
# Filter Helpers
# ---------------------------------------------------------------------------

def filter_interviews_for_export(
    interviews: List[Dict[str, Any]],
    status_filter: Optional[List[str]] = None,
    date_range: str = "All time",
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """Filter interviews based on export criteria."""
    filtered = interviews

    # Status filter
    if status_filter:
        status_lower = [s.lower() for s in status_filter]
        filtered = [
            i for i in filtered
            if (i.get("last_status") or "pending").lower() in status_lower
        ]

    # Date range filter
    today = datetime.now(tz.utc).date()

    if date_range == "Today":
        filtered = [i for i in filtered if _interview_date(i) == today]
    elif date_range == "This week":
        week_start = today - timedelta(days=today.weekday())
        filtered = [i for i in filtered if _interview_date(i) is not None and _interview_date(i) >= week_start]
    elif date_range == "This month":
        month_start = today.replace(day=1)
        filtered = [i for i in filtered if _interview_date(i) is not None and _interview_date(i) >= month_start]
    elif date_range == "Last 30 days":
        cutoff = today - timedelta(days=30)
        filtered = [i for i in filtered if _interview_date(i) is not None and _interview_date(i) >= cutoff]
    elif date_range == "Custom" and date_from and date_to:
        filtered = [
            i for i in filtered
            if _interview_date(i) is not None and date_from <= _interview_date(i) <= date_to
        ]

    return filtered


def _interview_date(interview: Dict[str, Any]) -> Optional[date]:
    """Extract date from interview record."""
    start_utc = interview.get("start_utc", "")
    if start_utc:
        try:
            return datetime.fromisoformat(start_utc.replace("Z", "+00:00")).date()
        except ValueError:
            pass
    return None


def filter_audit_entries(
    entries: List[Dict[str, Any]],
    action_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    search_term: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter audit entries based on criteria."""
    filtered = entries

    # Action filter (by human-readable name)
    if action_filter and action_filter != "All":
        # Find action codes that match this label
        matching_codes = [
            code for code, label in AUDIT_ACTION_DESCRIPTIONS.items()
            if label == action_filter
        ]
        if matching_codes:
            filtered = [e for e in filtered if e.get("action") in matching_codes]

    # Status filter
    if status_filter and status_filter != "All":
        status_lower = status_filter.lower()
        filtered = [e for e in filtered if (e.get("status") or "").lower() == status_lower]

    # Search filter
    if search_term:
        search_lower = search_term.lower()
        filtered = [
            e for e in filtered
            if search_lower in (e.get("candidate_email") or "").lower()
            or search_lower in (e.get("role_title") or "").lower()
            or search_lower in (e.get("actor") or "").lower()
        ]

    return filtered
