"""
Microsoft Graph client for app-only (client credentials) access to a scheduler mailbox.
Designed for Streamlit apps: no secrets are hard-coded; everything is injected via st.secrets.

Features:
- Client credentials token caching with automatic refresh
- Retry with exponential backoff for transient errors (429, 5xx)
- Automatic token refresh on 401 Unauthorized
- Error classification for appropriate handling
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import requests

from audit_log import LogLevel, log_structured


@dataclass(frozen=True)
class GraphConfig:
    tenant_id: str
    client_id: str
    client_secret: str
    scheduler_mailbox: str
    base_url: str = "https://graph.microsoft.com/v1.0"


class GraphAuthError(RuntimeError):
    """Raised when authentication with Microsoft Graph fails."""
    pass


class GraphAPIError(RuntimeError):
    """Raised when a Graph API call fails."""
    def __init__(self, message: str, status_code: int | None = None, response_json: Any | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_json = response_json


# ----------------------------
# Error Classification
# ----------------------------
class GraphErrorType(Enum):
    """Classification of Graph API errors for retry decisions."""
    TRANSIENT = "transient"      # Retry-able: 429, 5xx, network errors
    AUTH = "auth"                 # Token issue: 401
    CLIENT = "client"             # Bad request: 4xx (not 401, 429)
    UNKNOWN = "unknown"


def classify_error(status_code: int | None, exception: Exception | None = None) -> GraphErrorType:
    """Classify error type for retry decision."""
    if exception and isinstance(exception, (requests.exceptions.Timeout,
                                             requests.exceptions.ConnectionError)):
        return GraphErrorType.TRANSIENT
    if status_code is None:
        return GraphErrorType.UNKNOWN
    if status_code == 401:
        return GraphErrorType.AUTH
    if status_code == 429 or status_code >= 500:
        return GraphErrorType.TRANSIENT
    if 400 <= status_code < 500:
        return GraphErrorType.CLIENT
    return GraphErrorType.UNKNOWN


# ----------------------------
# Retry Decorator
# ----------------------------
T = TypeVar('T')


def with_retry(
    max_attempts: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    retry_on: tuple = (GraphErrorType.TRANSIENT,),
) -> Callable:
    """
    Retry decorator with exponential backoff.
    Only retries on specified error types. Logs retry attempts.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except GraphAPIError as e:
                    error_type = classify_error(e.status_code)
                    last_exception = e

                    if error_type not in retry_on:
                        raise  # Don't retry client errors

                    if attempt < max_attempts - 1:
                        delay = min(base_delay_s * (2 ** attempt), max_delay_s)
                        # Check for Retry-After header from 429 response
                        if e.status_code == 429 and e.response_json:
                            retry_after = None
                            if isinstance(e.response_json, dict):
                                retry_after = e.response_json.get("error", {}).get("retryAfterSeconds")
                            if retry_after:
                                delay = min(float(retry_after), max_delay_s)

                        log_structured(
                            LogLevel.WARNING,
                            f"Graph API call failed, retrying in {delay:.1f}s",
                            action=func.__name__,
                            error_type=error_type.value,
                            details={"attempt": attempt + 1, "max_attempts": max_attempts, "status_code": e.status_code},
                        )
                        time.sleep(delay)
                except requests.exceptions.RequestException as e:
                    last_exception = GraphAPIError(
                        f"Network error: {e}",
                        status_code=None,
                        response_json={"network_error": str(e), "type": type(e).__name__}
                    )
                    if attempt < max_attempts - 1:
                        delay = min(base_delay_s * (2 ** attempt), max_delay_s)
                        log_structured(
                            LogLevel.WARNING,
                            f"Network error, retrying in {delay:.1f}s: {e}",
                            action=func.__name__,
                            error_type="network_error",
                            details={"attempt": attempt + 1, "max_attempts": max_attempts},
                        )
                        time.sleep(delay)

            # All retries exhausted
            if last_exception:
                log_structured(
                    LogLevel.ERROR,
                    f"Graph API call failed after {max_attempts} attempts",
                    action=func.__name__,
                    error_type="max_retries_exceeded",
                )
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")
        return wrapper
    return decorator


# ----------------------------
# Graph Client
# ----------------------------
class GraphClient:
    """
    Minimal Graph wrapper with:
      - client credentials token caching with automatic refresh
      - retry logic for transient errors
      - create / patch / delete event under a scheduler mailbox
      - basic "diagnostics" helpers
    """

    def __init__(self, cfg: GraphConfig, timeout_s: int = 30):
        self.cfg = cfg
        self.timeout_s = timeout_s
        self._token: Optional[str] = None
        self._token_expiry_utc: Optional[datetime] = None

    # ---------------- Auth ----------------
    def _token_valid(self) -> bool:
        if not self._token or not self._token_expiry_utc:
            return False
        # refresh a little early (2 minutes before expiry)
        return datetime.now(timezone.utc) < (self._token_expiry_utc - timedelta(minutes=2))

    def _invalidate_token(self) -> None:
        """Invalidate cached token to force refresh."""
        self._token = None
        self._token_expiry_utc = None

    def get_token(self, force_refresh: bool = False) -> str:
        if (not force_refresh) and self._token_valid():
            return self._token  # type: ignore[return-value]

        token_url = f"https://login.microsoftonline.com/{self.cfg.tenant_id}/oauth2/v2.0/token"
        data = {
            "client_id": self.cfg.client_id,
            "client_secret": self.cfg.client_secret,
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials",
        }

        # Retry logic for token acquisition (3 attempts with backoff)
        max_attempts = 3
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                resp = requests.post(token_url, data=data, timeout=self.timeout_s)
                break  # Success, exit retry loop
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_attempts - 1:
                    delay = min(1.0 * (2 ** attempt), 10.0)
                    log_structured(
                        LogLevel.WARNING,
                        f"Token request network error, retrying in {delay:.1f}s: {e}",
                        action="get_token",
                        error_type="network_error",
                        details={"attempt": attempt + 1, "max_attempts": max_attempts},
                    )
                    time.sleep(delay)
                else:
                    log_structured(
                        LogLevel.ERROR,
                        f"Token request failed after {max_attempts} attempts: {e}",
                        action="get_token",
                        error_type="network_error",
                    )
                    raise GraphAuthError(f"Token request failed: {e}") from e

        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text}

        if resp.status_code >= 400:
            log_structured(
                LogLevel.ERROR,
                f"Token request failed: {resp.status_code}",
                action="get_token",
                error_type="auth_error",
                details={"status_code": resp.status_code},
            )
            raise GraphAuthError(f"Token request failed ({resp.status_code}): {payload}")

        access_token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 3599))
        if not access_token:
            raise GraphAuthError(f"Token response missing access_token: {payload}")

        self._token = access_token
        self._token_expiry_utc = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        return access_token

    def _headers(self) -> Dict[str, str]:
        token = self.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    # ---------------- Core HTTP ----------------
    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Dict[str, Any] | None = None,
        json_body: Any | None = None,
        _retry_auth: bool = True,  # Internal flag to prevent infinite retry
    ) -> Tuple[int, Any]:
        """
        Make HTTP request with automatic token refresh on 401.
        """
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers(),
                params=params,
                json=json_body,
                timeout=self.timeout_s,
            )
        except requests.exceptions.RequestException as e:
            raise GraphAPIError(
                f"Network error: {e}",
                status_code=None,
                response_json={"network_error": str(e), "type": type(e).__name__}
            )

        try:
            body = resp.json() if resp.text else None
        except Exception:
            body = {"raw": resp.text}

        # Handle 401 with automatic token refresh (once)
        if resp.status_code == 401 and _retry_auth:
            log_structured(
                LogLevel.INFO,
                "Received 401, refreshing token and retrying",
                action="_request",
                details={"method": method, "url": url},
            )
            self._invalidate_token()
            self.get_token(force_refresh=True)
            return self._request(method, url, params=params, json_body=json_body, _retry_auth=False)

        if resp.status_code >= 400:
            # Extract error message from Graph response for better diagnostics
            error_msg = f"Graph {method} failed ({resp.status_code})"
            if isinstance(body, dict) and "error" in body:
                graph_error = body.get("error", {})
                if isinstance(graph_error, dict):
                    error_msg += f": {graph_error.get('message', '')}"

            raise GraphAPIError(
                error_msg,
                status_code=resp.status_code,
                response_json=body,
            )
        return resp.status_code, body

    # ---------------- Events ----------------
    @with_retry(max_attempts=3, base_delay_s=1.0)
    def create_event(self, event_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a calendar event. Retries on transient errors."""
        url = f"{self.cfg.base_url}/users/{self.cfg.scheduler_mailbox}/events"
        _, body = self._request("POST", url, json_body=event_payload)
        return body or {}

    @with_retry(max_attempts=3, base_delay_s=1.0)
    def patch_event(self, event_id: str, patch_payload: Dict[str, Any], send_updates: str = "all") -> Dict[str, Any]:
        """Update a calendar event. Retries on transient errors."""
        url = f"{self.cfg.base_url}/users/{self.cfg.scheduler_mailbox}/events/{event_id}"
        params = {"sendUpdates": send_updates} if send_updates else None
        _, body = self._request("PATCH", url, params=params, json_body=patch_payload)
        return body or {}

    @with_retry(max_attempts=3, base_delay_s=1.0)
    def delete_event(self, event_id: str) -> None:
        """Delete a calendar event. Retries on transient errors."""
        url = f"{self.cfg.base_url}/users/{self.cfg.scheduler_mailbox}/events/{event_id}"
        self._request("DELETE", url)

    @with_retry(max_attempts=3, base_delay_s=1.0)
    def get_event(self, event_id: str) -> Dict[str, Any]:
        """Fetch a single calendar event by ID. Retries on transient errors."""
        url = f"{self.cfg.base_url}/users/{self.cfg.scheduler_mailbox}/events/{event_id}"
        _, body = self._request("GET", url)
        return body or {}

    # ---------------- Diagnostics ----------------
    def me(self) -> Dict[str, Any]:
        """Get current user info (app-only tokens usually cannot call /me)."""
        url = f"{self.cfg.base_url}/me"
        _, body = self._request("GET", url)
        return body or {}

    @with_retry(max_attempts=3, base_delay_s=1.0)
    def test_calendar_read(self, top: int = 5) -> Dict[str, Any]:
        """Test calendar read access. Retries on transient errors."""
        url = f"{self.cfg.base_url}/users/{self.cfg.scheduler_mailbox}/calendar/events"
        params = {"$top": str(top), "$orderby": "start/dateTime desc"}
        _, body = self._request("GET", url, params=params)
        return body or {}

    def create_dummy_event(self, subject: str, start_dt: Dict[str, str], end_dt: Dict[str, str], dry_run: bool = True) -> Dict[str, Any]:
        """Create a test event for diagnostics."""
        payload = {
            "subject": subject,
            "body": {"contentType": "HTML", "content": "PowerDash Graph diagnostics dummy event."},
            "start": start_dt,
            "end": end_dt,
            "location": {"displayName": "Diagnostics"},
            "attendees": [],
        }
        if dry_run:
            return {"dry_run": True, "payload": payload}
        return self.create_event(payload)

    # ---------------- Mail ----------------
    @with_retry(max_attempts=3, base_delay_s=1.0)
    def send_mail(
        self,
        subject: str,
        body: str,
        to_recipients: list[str],
        cc_recipients: list[str] | None = None,
        content_type: str = "Text",
        attachment: Dict[str, Any] | None = None,
        save_to_sent: bool = True,
    ) -> Dict[str, Any]:
        """
        Send an email via Microsoft Graph API. Retries on transient errors.

        Args:
            subject: Email subject
            body: Email body content
            to_recipients: List of recipient email addresses
            cc_recipients: Optional list of CC email addresses
            content_type: "Text" or "HTML"
            attachment: Optional dict with keys: name, contentBytes (base64), contentType
            save_to_sent: Whether to save the email to Sent Items
        """
        url = f"{self.cfg.base_url}/users/{self.cfg.scheduler_mailbox}/sendMail"

        message: Dict[str, Any] = {
            "subject": subject,
            "body": {
                "contentType": content_type,
                "content": body,
            },
            "toRecipients": [
                {"emailAddress": {"address": addr}} for addr in to_recipients if addr
            ],
        }

        if cc_recipients:
            message["ccRecipients"] = [
                {"emailAddress": {"address": addr}} for addr in cc_recipients if addr
            ]

        if attachment:
            import base64
            # Ensure content is base64 encoded
            content_bytes = attachment.get("contentBytes")
            if isinstance(content_bytes, bytes):
                content_bytes = base64.b64encode(content_bytes).decode("utf-8")

            message["attachments"] = [{
                "@odata.type": "#microsoft.graph.fileAttachment",
                "name": attachment.get("name", "attachment.bin"),
                "contentType": attachment.get("contentType", "application/octet-stream"),
                "contentBytes": content_bytes,
            }]

        payload = {
            "message": message,
            "saveToSentItems": save_to_sent,
        }

        _, response_body = self._request("POST", url, json_body=payload)
        return response_body or {"status": "sent"}

    @with_retry(max_attempts=3, base_delay_s=1.0)
    def fetch_unread_messages(self, top: int = 50, include_read: bool = False) -> list[Dict[str, Any]]:
        """
        Fetch messages from the scheduler mailbox via Graph API.
        Returns list of message dicts with id, subject, from, receivedDateTime, bodyPreview.

        Args:
            top: Maximum number of messages to fetch
            include_read: If True, fetches all recent messages (not just unread)
        """
        url = f"{self.cfg.base_url}/users/{self.cfg.scheduler_mailbox}/mailFolders/Inbox/messages"
        params = {
            "$top": str(top),
            "$orderby": "receivedDateTime desc",
            "$select": "id,subject,from,receivedDateTime,bodyPreview,body,isRead",
        }
        if not include_read:
            params["$filter"] = "isRead eq false"
        _, body = self._request("GET", url, params=params)
        return body.get("value", []) if body else []

    @with_retry(max_attempts=3, base_delay_s=1.0)
    def mark_message_read(self, message_id: str) -> None:
        """Mark a message as read."""
        url = f"{self.cfg.base_url}/users/{self.cfg.scheduler_mailbox}/messages/{message_id}"
        self._request("PATCH", url, json_body={"isRead": True})
