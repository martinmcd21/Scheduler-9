"""
SQLite audit log + lightweight interview persistence.

- audit_log table: append-only (as per requirements)
- interviews table: store created event IDs so we can reschedule/cancel later
- Structured logging for production diagnostics
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ----------------------------
# Structured Logging
# ----------------------------
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class InterviewStatus(Enum):
    """Interview lifecycle statuses for tracking."""
    PENDING = "pending"          # Invite created, awaiting response
    CONFIRMED = "confirmed"      # Candidate confirmed
    RESCHEDULED = "rescheduled"  # Time changed
    CANCELLED = "cancelled"      # Interview cancelled
    COMPLETED = "completed"      # Interview took place
    NO_SHOW = "no_show"          # Candidate didn't attend


def _setup_logger() -> logging.Logger:
    """Configure structured JSON-style logger for production diagnostics."""
    logger = logging.getLogger("powerdash")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","message":"%(message)s"}'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


_logger = _setup_logger()


def log_structured(
    level: LogLevel,
    message: str,
    *,
    action: str = "",
    error_type: str = "",
    details: Optional[Dict[str, Any]] = None,
    exc_info: bool = False,
) -> None:
    """Log with structured context for aggregation and diagnostics."""
    extra = {"action": action, "error_type": error_type}
    if details:
        extra.update(details)
    log_msg = f"{message} | {extra}"

    if exc_info:
        log_msg += f" | traceback={traceback.format_exc()}"

    getattr(_logger, level.value.lower())(log_msg)


# ----------------------------
# Utilities
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class AuditEntry:
    timestamp_utc: str
    action: str
    actor: str
    candidate_email: str
    hiring_manager_email: str
    recruiter_email: str
    role_title: str
    event_id: str
    payload_json: str
    status: str
    error_message: str


# ----------------------------
# AuditLog Class
# ----------------------------
class AuditLog:
    def __init__(self, db_path: str):
        self.path = Path(db_path)
        _ensure_parent(self.path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Create a thread-safe connection with WAL mode for Streamlit concurrency."""
        try:
            conn = sqlite3.connect(
                str(self.path),
                timeout=30.0,  # Wait up to 30s for locks
                check_same_thread=False,  # Streamlit runs in multiple threads
            )
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent read/write
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout
            return conn
        except sqlite3.Error as e:
            log_structured(
                LogLevel.ERROR,
                f"Database connection failed: {e}",
                action="db_connect",
                error_type="sqlite_error",
                exc_info=True,
            )
            raise

    def _init_db(self) -> None:
        """Initialize database tables."""
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp_utc TEXT NOT NULL,
                        action TEXT NOT NULL,
                        actor TEXT,
                        candidate_email TEXT,
                        hiring_manager_email TEXT,
                        recruiter_email TEXT,
                        role_title TEXT,
                        event_id TEXT,
                        payload_json TEXT,
                        status TEXT,
                        error_message TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS interviews (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_utc TEXT NOT NULL,
                        role_title TEXT,
                        candidate_email TEXT,
                        hiring_manager_email TEXT,
                        recruiter_email TEXT,
                        duration_minutes INTEGER,
                        start_utc TEXT,
                        end_utc TEXT,
                        display_timezone TEXT,
                        graph_event_id TEXT,
                        teams_join_url TEXT,
                        subject TEXT,
                        last_status TEXT
                    )
                    """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS processed_messages (
                        message_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        processed_utc TEXT NOT NULL,
                        PRIMARY KEY (message_id, action)
                    )
                    """
                )

                conn.commit()

                # Migration: Add candidate_timezone column if missing
                try:
                    conn.execute("SELECT candidate_timezone FROM interviews LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute("ALTER TABLE interviews ADD COLUMN candidate_timezone TEXT")
                    conn.commit()
                    log_structured(
                        LogLevel.INFO,
                        "Added candidate_timezone column to interviews table",
                        action="db_migration",
                    )

                # Migration: Add panel_interviewers_json column if missing
                try:
                    conn.execute("SELECT panel_interviewers_json FROM interviews LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute("ALTER TABLE interviews ADD COLUMN panel_interviewers_json TEXT")
                    conn.commit()
                    log_structured(
                        LogLevel.INFO,
                        "Added panel_interviewers_json column to interviews table",
                        action="db_migration",
                    )

                # Migration: Add is_panel_interview column if missing
                try:
                    conn.execute("SELECT is_panel_interview FROM interviews LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute("ALTER TABLE interviews ADD COLUMN is_panel_interview INTEGER DEFAULT 0")
                    conn.commit()
                    log_structured(
                        LogLevel.INFO,
                        "Added is_panel_interview column to interviews table",
                        action="db_migration",
                    )

                # Migration: Add candidates_json column for multi-candidate support
                try:
                    conn.execute("SELECT candidates_json FROM interviews LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute("ALTER TABLE interviews ADD COLUMN candidates_json TEXT")
                    conn.commit()
                    log_structured(
                        LogLevel.INFO,
                        "Added candidates_json column to interviews table",
                        action="db_migration",
                    )

                # Migration: Add is_group_interview column for scheduling mode
                try:
                    conn.execute("SELECT is_group_interview FROM interviews LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute("ALTER TABLE interviews ADD COLUMN is_group_interview INTEGER DEFAULT 0")
                    conn.commit()
                    log_structured(
                        LogLevel.INFO,
                        "Added is_group_interview column to interviews table",
                        action="db_migration",
                    )

                # Migration: Add status tracking columns
                try:
                    conn.execute("SELECT status_reason FROM interviews LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute("ALTER TABLE interviews ADD COLUMN status_reason TEXT")
                    conn.commit()
                    log_structured(
                        LogLevel.INFO,
                        "Added status_reason column to interviews table",
                        action="db_migration",
                    )

                try:
                    conn.execute("SELECT status_updated_utc FROM interviews LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute("ALTER TABLE interviews ADD COLUMN status_updated_utc TEXT")
                    conn.commit()
                    log_structured(
                        LogLevel.INFO,
                        "Added status_updated_utc column to interviews table",
                        action="db_migration",
                    )

                try:
                    conn.execute("SELECT status_updated_by FROM interviews LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute("ALTER TABLE interviews ADD COLUMN status_updated_by TEXT")
                    conn.commit()
                    log_structured(
                        LogLevel.INFO,
                        "Added status_updated_by column to interviews table",
                        action="db_migration",
                    )

                # Migration: Add ICS tracking columns
                try:
                    conn.execute("SELECT ics_sequence FROM interviews LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute("ALTER TABLE interviews ADD COLUMN ics_sequence INTEGER DEFAULT 0")
                    conn.commit()
                    log_structured(
                        LogLevel.INFO,
                        "Added ics_sequence column to interviews table",
                        action="db_migration",
                    )

                try:
                    conn.execute("SELECT ics_uid FROM interviews LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute("ALTER TABLE interviews ADD COLUMN ics_uid TEXT")
                    conn.commit()
                    log_structured(
                        LogLevel.INFO,
                        "Added ics_uid column to interviews table",
                        action="db_migration",
                    )

                # Migration: Add index for faster event_id lookups in audit_log
                try:
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_event_id ON audit_log(event_id)")
                    conn.commit()
                except sqlite3.OperationalError:
                    pass  # Index may already exist

            finally:
                conn.close()
        except sqlite3.Error as e:
            log_structured(
                LogLevel.ERROR,
                f"Database initialization failed: {e}",
                action="db_init",
                error_type="sqlite_error",
                exc_info=True,
            )
            raise

    @staticmethod
    def redact_payload(payload: Any) -> str:
        """
        Redact known secret-like fields from payloads before storing.
        Returns safe JSON string, never fails.
        """
        try:
            # Handle non-JSON-serializable objects
            if hasattr(payload, '__dict__') and not isinstance(payload, dict):
                payload = payload.__dict__
            s = json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError, RecursionError) as e:
            log_structured(
                LogLevel.WARNING,
                f"Payload serialization failed: {e}",
                action="redact_payload",
                error_type="serialization_error",
            )
            return f"<serialization-failed: {type(payload).__name__}>"

        # Redact sensitive keys with proper regex (JSON-aware)
        for key in ["client_secret", "authorization", "access_token", "refresh_token", "password", "api_key"]:
            s = re.sub(rf'"{key}":\s*"[^"]*"', f'"{key}": "[REDACTED]"', s, flags=re.IGNORECASE)
            s = re.sub(rf'{key}=[^\s&"]+', f'{key}=[REDACTED]', s, flags=re.IGNORECASE)
        return s

    def log(
        self,
        action: str,
        *,
        actor: str = "",
        candidate_email: str = "",
        hiring_manager_email: str = "",
        recruiter_email: str = "",
        role_title: str = "",
        event_id: str = "",
        payload: Any = None,
        status: str = "success",
        error_message: str = "",
    ) -> bool:
        """
        Append audit entry. Returns True on success, False on failure.
        Never raises - audit logging should not crash the app.
        """
        payload_json = self.redact_payload(payload) if payload is not None else ""

        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO audit_log (
                        timestamp_utc, action, actor, candidate_email, hiring_manager_email, recruiter_email,
                        role_title, event_id, payload_json, status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        utc_now_iso(),
                        action,
                        actor,
                        candidate_email,
                        hiring_manager_email,
                        recruiter_email,
                        role_title,
                        event_id,
                        payload_json,
                        status,
                        error_message[:2000] if error_message else "",
                    ),
                )
                conn.commit()  # Explicit commit - critical for durability
                return True
            finally:
                conn.close()
        except sqlite3.Error as e:
            log_structured(
                LogLevel.ERROR,
                f"Audit log write failed: {e}",
                action="audit_write",
                error_type="sqlite_error",
                details={"attempted_action": action},
                exc_info=True,
            )
            return False

    def upsert_interview(
        self,
        *,
        role_title: str,
        candidate_email: str,
        hiring_manager_email: str,
        recruiter_email: str,
        duration_minutes: int,
        start_utc: str,
        end_utc: str,
        display_timezone: str,
        candidate_timezone: str,
        graph_event_id: str,
        teams_join_url: str,
        subject: str,
        last_status: str,
        panel_interviewers_json: str = "",
        is_panel_interview: bool = False,
        candidates_json: str = "",
        is_group_interview: bool = False,
    ) -> bool:
        """
        Insert interview record. Returns True on success, False on failure.

        Args:
            candidate_timezone: IANA timezone for the candidate (used for invitation display)
            panel_interviewers_json: JSON array of panel interviewer objects [{name, email}, ...]
            is_panel_interview: True if this is a panel interview with multiple interviewers
            candidates_json: JSON array of candidate objects [{email, name}, ...] for multi-candidate
            is_group_interview: True if all candidates are in a single group meeting
        """
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO interviews (
                        created_utc, role_title, candidate_email, hiring_manager_email, recruiter_email,
                        duration_minutes, start_utc, end_utc, display_timezone, candidate_timezone,
                        graph_event_id, teams_join_url, subject, last_status,
                        panel_interviewers_json, is_panel_interview,
                        candidates_json, is_group_interview
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        utc_now_iso(),
                        role_title,
                        candidate_email,
                        hiring_manager_email,
                        recruiter_email,
                        int(duration_minutes),
                        start_utc,
                        end_utc,
                        display_timezone,
                        candidate_timezone,
                        graph_event_id,
                        teams_join_url,
                        subject,
                        last_status,
                        panel_interviewers_json,
                        1 if is_panel_interview else 0,
                        candidates_json,
                        1 if is_group_interview else 0,
                    ),
                )
                conn.commit()  # Explicit commit
                return True
            finally:
                conn.close()
        except sqlite3.Error as e:
            log_structured(
                LogLevel.ERROR,
                f"Interview upsert failed: {e}",
                action="interview_upsert",
                error_type="sqlite_error",
                exc_info=True,
            )
            return False

    def interview_exists(
        self,
        *,
        candidate_email: str,
        hiring_manager_email: str,
        role_title: str,
        start_utc: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if an interview already exists for this combination (idempotency check).
        Returns the existing interview dict if found, None otherwise.
        """
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT * FROM interviews
                    WHERE LOWER(candidate_email) = LOWER(?)
                    AND LOWER(hiring_manager_email) = LOWER(?)
                    AND role_title = ?
                    AND start_utc = ?
                    AND last_status NOT IN ('cancelled', 'deleted')
                    LIMIT 1
                    """,
                    (candidate_email, hiring_manager_email, role_title, start_utc),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()
        except sqlite3.Error as e:
            log_structured(
                LogLevel.WARNING,
                f"Interview exists check failed: {e}",
                action="interview_exists_check",
                error_type="sqlite_error",
                exc_info=True,
            )
            return None  # On error, allow creation (fail open)

    def list_recent_audit(self, limit: int = 200) -> List[Dict[str, Any]]:
        """List recent audit log entries. Returns empty list on error."""
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()
        except sqlite3.Error as e:
            log_structured(
                LogLevel.ERROR,
                f"Audit log read failed: {e}",
                action="audit_read",
                error_type="sqlite_error",
                exc_info=True,
            )
            return []

    def list_interviews(
        self,
        limit: int = 200,
        status_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List recent interviews with optional status filtering.

        Args:
            limit: Maximum number of interviews to return
            status_filter: Optional status to filter by (e.g., "pending", "cancelled")

        Returns empty list on error.
        """
        try:
            conn = self._connect()
            try:
                if status_filter:
                    rows = conn.execute(
                        "SELECT * FROM interviews WHERE last_status = ? ORDER BY id DESC LIMIT ?",
                        (status_filter.lower(), int(limit)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM interviews ORDER BY id DESC LIMIT ?",
                        (int(limit),),
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()
        except sqlite3.Error as e:
            log_structured(
                LogLevel.ERROR,
                f"Interview list read failed: {e}",
                action="interview_read",
                error_type="sqlite_error",
                exc_info=True,
            )
            return []

    def get_active_interviews(self) -> List[Dict[str, Any]]:
        """
        Get all active (non-cancelled, non-deleted) interviews.
        Used for filtering out already-scheduled time slots.

        Returns list of interviews with start_utc and end_utc fields.
        Returns empty list on error.
        """
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM interviews
                    WHERE last_status NOT IN ('cancelled', 'deleted')
                    ORDER BY start_utc ASC
                    """
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()
        except sqlite3.Error as e:
            log_structured(
                LogLevel.ERROR,
                f"Active interviews fetch failed: {e}",
                action="get_active_interviews",
                error_type="sqlite_error",
                exc_info=True,
            )
            return []

    def update_interview_status(
        self,
        event_id: str,
        new_status: InterviewStatus,
        reason: Optional[str] = None,
        updated_by: Optional[str] = None,
    ) -> bool:
        """
        Update interview status with audit trail.

        Args:
            event_id: Graph event ID of the interview
            new_status: New InterviewStatus value
            reason: Optional reason for the status change
            updated_by: Email of user who made the change

        Returns True on success, False on failure.
        """
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE interviews
                    SET last_status = ?,
                        status_reason = ?,
                        status_updated_utc = ?,
                        status_updated_by = ?
                    WHERE graph_event_id = ?
                    """,
                    (
                        new_status.value,
                        reason,
                        utc_now_iso(),
                        updated_by,
                        event_id,
                    ),
                )
                conn.commit()
                log_structured(
                    LogLevel.INFO,
                    f"Interview status updated to {new_status.value}",
                    action="status_update",
                    details={"event_id": event_id, "new_status": new_status.value},
                )
                return True
            finally:
                conn.close()
        except sqlite3.Error as e:
            log_structured(
                LogLevel.ERROR,
                f"Failed to update interview status: {e}",
                action="status_update",
                error_type="sqlite_error",
                exc_info=True,
            )
            return False

    def get_interview_history(self, event_id: str) -> List[Dict[str, Any]]:
        """
        Get complete audit history for a specific interview.

        Args:
            event_id: Graph event ID of the interview

        Returns list of audit log entries in descending order (newest first).
        """
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM audit_log
                    WHERE event_id = ?
                    ORDER BY timestamp_utc DESC
                    """,
                    (event_id,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()
        except sqlite3.Error as e:
            log_structured(
                LogLevel.ERROR,
                f"Failed to get interview history: {e}",
                action="get_history",
                error_type="sqlite_error",
                exc_info=True,
            )
            return []

    def increment_ics_sequence(self, event_id: str) -> int:
        """
        Increment and return the ICS sequence number for an interview.

        RFC 5545 requires incrementing SEQUENCE on any update to an event.
        This ensures calendar clients properly update their cached events.

        Args:
            event_id: Graph event ID of the interview

        Returns the new sequence number, or -1 on error.
        """
        try:
            conn = self._connect()
            try:
                # Increment and return in one transaction
                conn.execute(
                    """
                    UPDATE interviews
                    SET ics_sequence = COALESCE(ics_sequence, 0) + 1
                    WHERE graph_event_id = ?
                    """,
                    (event_id,),
                )
                conn.commit()

                # Get the new value
                row = conn.execute(
                    "SELECT ics_sequence FROM interviews WHERE graph_event_id = ?",
                    (event_id,),
                ).fetchone()

                return row["ics_sequence"] if row else -1
            finally:
                conn.close()
        except sqlite3.Error as e:
            log_structured(
                LogLevel.ERROR,
                f"Failed to increment ICS sequence: {e}",
                action="ics_sequence_increment",
                error_type="sqlite_error",
                exc_info=True,
            )
            return -1

    def get_interview_by_event_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Get interview record by Graph event ID.

        Args:
            event_id: Graph event ID of the interview

        Returns interview dict or None if not found.
        """
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM interviews WHERE graph_event_id = ?",
                    (event_id,),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()
        except sqlite3.Error as e:
            log_structured(
                LogLevel.ERROR,
                f"Failed to get interview by event ID: {e}",
                action="get_interview",
                error_type="sqlite_error",
                exc_info=True,
            )
            return None

    def update_interview_ics_uid(self, event_id: str, ics_uid: str) -> bool:
        """
        Store the ICS UID for an interview.

        Args:
            event_id: Graph event ID of the interview
            ics_uid: The stable ICS UID for calendar tracking

        Returns True on success, False on failure.
        """
        try:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE interviews SET ics_uid = ? WHERE graph_event_id = ?",
                    (ics_uid, event_id),
                )
                conn.commit()
                return True
            finally:
                conn.close()
        except sqlite3.Error as e:
            log_structured(
                LogLevel.ERROR,
                f"Failed to update ICS UID: {e}",
                action="update_ics_uid",
                error_type="sqlite_error",
                exc_info=True,
            )
            return False


def has_processed_message(message_id: str, action: str = "invite_sent") -> bool:
    """Check whether a message_id has already been processed for a given action."""
    if not message_id:
        return False
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM processed_messages WHERE message_id = ? AND action = ? LIMIT 1",
        (message_id, action),
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


def mark_message_processed(message_id: str, action: str = "invite_sent") -> None:
    """Mark a message_id as processed so it is never handled twice."""
    if not message_id:
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO processed_messages (message_id, action, processed_utc) VALUES (?, ?, ?)",
        (message_id, action, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


# ----------------------------
# Compatibility DB Helpers (used by dedupe functions)
# ----------------------------
_DEFAULT_DB_PATH = Path(__file__).with_name("audit_log.db")

def get_connection() -> sqlite3.Connection:
    """Return a sqlite connection to the audit DB (used by dedupe helpers)."""
    conn = sqlite3.connect(
        str(_DEFAULT_DB_PATH),
        timeout=30.0,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn

# ----------------------------
# Dedupe helpers (auto-send safety)
# ----------------------------
_DEFAULT_DB_PATH = str(Path(__file__).with_name("audit_log.db"))

def has_processed_message(message_id: str, action: str = "invite_sent") -> bool:
    """Check whether a Graph message_id has already been processed."""
    if not message_id:
        return False

    log = AuditLog(_DEFAULT_DB_PATH)
    conn = log._connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM processed_messages WHERE message_id = ? AND action = ? LIMIT 1",
            (message_id, action),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def mark_message_processed(message_id: str, action: str = "invite_sent") -> None:
    """Mark a Graph message_id as processed so we never auto-send twice."""
    if not message_id:
        return

    log = AuditLog(_DEFAULT_DB_PATH)
    conn = log._connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO processed_messages (message_id, action, processed_utc) VALUES (?, ?, ?)",
            (message_id, action, utc_now_iso()),
        )
        conn.commit()
    finally:
        conn.close()
