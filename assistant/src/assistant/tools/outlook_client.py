from __future__ import annotations

"""
Microsoft Graph API client for saving email drafts to Outlook.

Authentication uses the Device Code Flow (OAuth2), which is suitable for
CLI/server applications where the user can open a browser to authorize.
Tokens are cached locally so the user only needs to authenticate once.

Required .env variables:
  OUTLOOK_CLIENT_ID   - Azure AD application (client) ID
  OUTLOOK_TENANT_ID   - Azure AD tenant ID (or "common" for personal accounts)

Optional:
  OUTLOOK_TOKEN_CACHE - Path to token cache file (default: .outlook_token_cache)
"""

import json
import logging
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
SCOPES = ["Mail.Send"]


class OutlookAuthError(RuntimeError):
    """Raised when authentication with Microsoft cannot be completed."""


class OutlookClient:
    """Saves email drafts to the authenticated user's Outlook mailbox."""

    def __init__(
        self,
        client_id: str,
        tenant_id: str,
        token_cache_path: str = ".outlook_token_cache",
    ) -> None:
        if not client_id or not tenant_id:
            raise OutlookAuthError(
                "OUTLOOK_CLIENT_ID and OUTLOOK_TENANT_ID must be set in .env"
            )
        self._client_id = client_id
        self._tenant_id = tenant_id
        self._cache_path = Path(token_cache_path)
        self._app = self._build_app()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_access_token(self) -> str:
        """Return a valid access token, triggering device-code flow if needed."""
        try:
            import msal
        except ImportError as exc:
            raise OutlookAuthError(
                "msal package is required. Run: pip install msal"
            ) from exc

        accounts = self._app.get_accounts()
        if accounts:
            result = self._app.acquire_token_silent(SCOPES, account=accounts[0])
            if result and "access_token" in result:
                self._save_cache()
                return result["access_token"]

        # Device code flow — prints a URL and code for the user to enter
        flow = self._app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            raise OutlookAuthError(f"Failed to create device flow: {flow}")

        print(flow["message"])  # noqa: T201  — intentional user prompt
        result = self._app.acquire_token_by_device_flow(flow)

        if "access_token" not in result:
            raise OutlookAuthError(
                f"Authentication failed: {result.get('error_description', result)}"
            )
        self._save_cache()
        return result["access_token"]

    def save_draft(
        self,
        subject: str,
        body: str,
        to: str | None = None,
    ) -> dict[str, Any]:
        """
        Save a draft email in the user's Outlook Drafts folder.

        Returns the Graph API response dict with 'id' and 'webLink'.
        """
        token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        to_recipients = []
        if to:
            to_recipients = [{"emailAddress": {"address": to}}]

        payload: dict[str, Any] = {
            "subject": subject,
            "body": {
                "contentType": "Text",
                "content": body,
            },
            "toRecipients": to_recipients,
        }

        response = requests.post(
            f"{GRAPH_BASE}/me/messages",
            headers=headers,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        log.info("Draft saved to Outlook. Message ID: %s", data.get("id"))
        return {
            "status": "draft_saved",
            "message_id": data.get("id"),
            "web_link": data.get("webLink"),
        }

    def send_message(
        self,
        subject: str,
        body: str,
        to: str,
    ) -> dict[str, Any]:
        """Send an email immediately via Microsoft Graph sendMail endpoint."""
        token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "message": {
                "subject": subject,
                "body": {"contentType": "Text", "content": body},
                "toRecipients": [{"emailAddress": {"address": to}}],
            },
            "saveToSentItems": "true",
        }
        response = requests.post(
            f"{GRAPH_BASE}/me/sendMail",
            headers=headers,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        log.info("Email sent to %s", to)
        return {"status": "sent", "to": to}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_app(self) -> Any:
        try:
            import msal
        except ImportError as exc:
            raise OutlookAuthError(
                "msal package is required. Run: pip install msal"
            ) from exc

        cache = msal.SerializableTokenCache()
        if self._cache_path.exists():
            cache.deserialize(self._cache_path.read_text(encoding="utf-8"))

        authority = f"https://login.microsoftonline.com/{self._tenant_id}"
        app = msal.PublicClientApplication(
            client_id=self._client_id,
            authority=authority,
            token_cache=cache,
        )
        self._cache = cache
        return app

    def _save_cache(self) -> None:
        if hasattr(self, "_cache") and self._cache.has_state_changed:
            self._cache_path.write_text(
                self._cache.serialize(), encoding="utf-8"
            )
