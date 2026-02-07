import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Optional


class ICSValidationError(Exception):
    """Raised when required ICS fields are missing or invalid."""
    pass


def stable_uid(*parts: str, domain: str = "powerdashhr.com") -> str:
    """
    Deterministic UID for ICS invites.
    Accepts multiple parts and combines them into one stable hash.
    """

    cleaned = []
    for p in parts:
        if p is None:
            continue
        p = str(p).strip()
        if p:
            cleaned.append(p)

    if not cleaned:
        raise ICSValidationError("stable_uid requires at least one non-empty part")

    seed = "|".join(cleaned)
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return f"{digest}@{domain}"

def _format_dt(dt: datetime) -> str:
    """Return UTC datetime in ICS format."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _escape_ics(text: str) -> str:
    """Escape special characters for ICS format."""
    if not text:
        return ""
    return (
        text.replace("\\", "\\\\")
        .replace(";", r"\;")
        .replace(",", r"\,")
        .replace("\n", r"\n")
    )


def build_ics_invite(
    organizer_email: str,
    organizer_name: str,
    required_attendees: List[Tuple[str, str]],
    optional_attendees: List[Tuple[str, str]],
    summary: str,
    description: str,
    dtstart_utc: datetime,
    dtend_utc: datetime,
    location: str = "",
    url: str = "",
    uid: Optional[str] = None,
    method: str = "REQUEST",
    status: str = "CONFIRMED",
    sequence: int = 0,
) -> bytes:
    """
    RFC 5545 ICS invite generator.
    Supports REQUEST and CANCEL methods.
    """

    if not organizer_email:
        raise ICSValidationError("Organizer email is required")

    if not summary:
        raise ICSValidationError("Summary is required")

    if dtstart_utc is None or dtend_utc is None:
        raise ICSValidationError("Start and end times are required")

    if uid is None:
        uid = f"{uuid.uuid4()}@powerdashhr.com"

    dtstamp = _format_dt(datetime.now(timezone.utc))
    dtstart = _format_dt(dtstart_utc)
    dtend = _format_dt(dtend_utc)

    summary = _escape_ics(summary)
    description = _escape_ics(description or "")
    location = _escape_ics(location or "")
    url = _escape_ics(url or "")

    if url:
        description = f"{description}\\n\\nJoin link: {url}".strip()

    method = (method or "REQUEST").upper()
    status = (status or "CONFIRMED").upper()

    lines = [
        "BEGIN:VCALENDAR",
        "PRODID:-//PowerDash HR//Interview Scheduler//EN",
        "VERSION:2.0",
        "CALSCALE:GREGORIAN",
        f"METHOD:{method}",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"SEQUENCE:{sequence}",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART:{dtstart}",
        f"DTEND:{dtend}",
        f"SUMMARY:{summary}",
        f"DESCRIPTION:{description}",
        f"LOCATION:{location}",
        f"STATUS:{status}",
        "TRANSP:OPAQUE",
        f"ORGANIZER;CN={_escape_ics(organizer_name)}:mailto:{organizer_email}",
    ]

    # Required attendees
    for email, name in required_attendees or []:
        email = (email or "").strip()
        if not email:
            continue
        name = _escape_ics(name or email)
        lines.append(
            f"ATTENDEE;CN={name};ROLE=REQ-PARTICIPANT;"
            f"PARTSTAT=NEEDS-ACTION;RSVP=TRUE:mailto:{email}"
        )

    # Optional attendees
    for email, name in optional_attendees or []:
        email = (email or "").strip()
        if not email:
            continue
        name = _escape_ics(name or email)
        lines.append(
            f"ATTENDEE;CN={name};ROLE=OPT-PARTICIPANT;"
            f"PARTSTAT=NEEDS-ACTION;RSVP=TRUE:mailto:{email}"
        )

    if url:
        lines.append(f"URL:{url}")

    # Alarm only makes sense for REQUEST
    if method == "REQUEST":
        lines.extend([
            "BEGIN:VALARM",
            "TRIGGER:-PT15M",
            "ACTION:DISPLAY",
            "DESCRIPTION:Interview Reminder",
            "END:VALARM",
        ])

    lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")

    ics_text = "\r\n".join(lines) + "\r\n"
    return ics_text.encode("utf-8")


@dataclass
class ICSInvite:
    """
    Backwards compatible wrapper class.
    """

    organizer_email: str
    organizer_name: str
    attendee_emails: List[str]
    summary: str
    description: str
    dtstart_utc: datetime
    dtend_utc: datetime
    location: str = ""
    url: str = ""
    uid: Optional[str] = None
    display_timezone: str = "UTC"

    def to_ics(self) -> bytes:
        return self.to_bytes()

    def to_bytes(self) -> bytes:
        required_attendees = [(e, e) for e in self.attendee_emails or []]
        optional_attendees = []

        return build_ics_invite(
            organizer_email=self.organizer_email,
            organizer_name=self.organizer_name,
            required_attendees=required_attendees,
            optional_attendees=optional_attendees,
            summary=self.summary,
            description=self.description,
            dtstart_utc=self.dtstart_utc,
            dtend_utc=self.dtend_utc,
            location=self.location,
            url=self.url,
            uid=self.uid,
            method="REQUEST",
            status="CONFIRMED",
            sequence=0,
        )    
    def to_bytes(self) -> bytes:
        required_attendees = [(e, e) for e in self.attendee_emails or []]
        optional_attendees = []

        return build_ics_invite(
            organizer_email=self.organizer_email,
            organizer_name=self.organizer_name,
            required_attendees=required_attendees,
            optional_attendees=optional_attendees,
            summary=self.summary,
            description=self.description,
            dtstart_utc=self.dtstart_utc,
            dtend_utc=self.dtend_utc,
            location=self.location,
            url=self.url,
            uid=self.uid,
            method="REQUEST",
            status="CONFIRMED",
            sequence=0,
        )


def create_ics_from_interview(
    organizer_email: str,
    organizer_name: str,
    attendees: List[Tuple[str, str]],
    optional_attendees: List[Tuple[str, str]],
    summary: str,
    description: str,
    start_utc: datetime,
    end_utc: datetime,
    location: str = "",
    join_url: str = "",
    uid_seed: Optional[str] = None,
    sequence: int = 0,
) -> bytes:
    """
    Main helper used by app.py for generating meeting invites.
    """

    uid = None
    if uid_seed:
        uid = stable_uid(uid_seed)

    return build_ics_invite(
        organizer_email=organizer_email,
        organizer_name=organizer_name,
        required_attendees=attendees,
        optional_attendees=optional_attendees,
        summary=summary,
        description=description,
        dtstart_utc=start_utc,
        dtend_utc=end_utc,
        location=location,
        url=join_url,
        uid=uid,
        method="REQUEST",
        status="CONFIRMED",
        sequence=sequence,
    )


def generate_cancellation_ics(
    organizer_email: str,
    organizer_name: str,
    attendees: List[Tuple[str, str]],
    optional_attendees: List[Tuple[str, str]],
    summary: str,
    description: str,
    start_utc: datetime,
    end_utc: datetime,
    uid: str,
    sequence: int = 1,
    location: str = "",
    url: str = "",
) -> bytes:
    """
    Generate an ICS cancellation invite (METHOD:CANCEL).
    Outlook requires UID + SEQUENCE.
    """

    if not uid:
        raise ICSValidationError("UID is required to cancel an invite")

    return build_ics_invite(
        organizer_email=organizer_email,
        organizer_name=organizer_name,
        required_attendees=attendees,
        optional_attendees=optional_attendees,
        summary=summary,
        description=description,
        dtstart_utc=start_utc,
        dtend_utc=end_utc,
        location=location,
        url=url,
        uid=uid,
        method="CANCEL",
        status="CANCELLED",
        sequence=sequence,
    )
