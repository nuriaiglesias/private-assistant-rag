from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class EmailDraft:
    to: str | None
    subject: str
    body: str
    requires_confirmation: bool = True
    status: str = "draft"
    outlook_message_id: str | None = None
    outlook_web_link: str | None = None


class EmailTool:
    """
    Email drafting tool with optional Outlook integration.

    When OUTLOOK_CLIENT_ID and OUTLOOK_TENANT_ID are set in the environment,
    drafts are saved directly to the user's Outlook Drafts folder via the
    Microsoft Graph API (device-code OAuth2 flow).

    Without those variables the tool falls back to local draft generation
    only (no email is sent or saved externally).
    """

    def __init__(self) -> None:
        self._outlook_client_id = os.environ.get("OUTLOOK_CLIENT_ID", "")
        self._outlook_tenant_id = os.environ.get("OUTLOOK_TENANT_ID", "")
        self._outlook_cache = os.environ.get(
            "OUTLOOK_TOKEN_CACHE", ".outlook_token_cache"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draft_email(
        self,
        user_request: str,
        supporting_context: str,
        to: str | None = None,
    ) -> EmailDraft:
        """Build an EmailDraft and, if Outlook is configured, save it remotely."""
        subject, body = self._build_subject_and_body(user_request, supporting_context)
        draft = EmailDraft(
            to=to,
            subject=subject,
            body=body,
            requires_confirmation=True,
            status="draft",
        )

        if self._outlook_enabled():
            draft = self._save_to_outlook(draft)

        return draft

    def send_email(self, draft: EmailDraft, confirmed: bool = False) -> dict[str, str]:
        """
        Sending is intentionally not supported — drafts are saved to Outlook
        for the user to review and send manually.
        """
        if not confirmed:
            return {
                "status": "not_sent",
                "reason": "Se requiere confirmacion explicita del usuario.",
            }
        if draft.outlook_web_link:
            return {
                "status": "draft_ready",
                "reason": "El borrador ha sido guardado en Outlook. Revisalo y envialo manualmente.",
                "web_link": draft.outlook_web_link,
            }
        return {
            "status": "draft_only",
            "reason": "Borrador generado localmente. Configura OUTLOOK_CLIENT_ID para guardarlo en Outlook.",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _outlook_enabled(self) -> bool:
        return bool(self._outlook_client_id and self._outlook_tenant_id)

    def _build_subject_and_body(
        self, user_request: str, supporting_context: str
    ) -> tuple[str, str]:
        subject = "Solicitud de informacion"
        body = (
            "Buenos dias,\n\n"
            "Me gustaria solicitar informacion sobre la siguiente consulta:\n\n"
            f"{user_request}\n\n"
            "Informacion relevante recopilada:\n"
            f"{supporting_context}\n\n"
            "Muchas gracias.\n"
        )
        return subject, body

    def _save_to_outlook(self, draft: EmailDraft) -> EmailDraft:
        try:
            from assistant.tools.outlook_client import OutlookClient, OutlookAuthError

            client = OutlookClient(
                client_id=self._outlook_client_id,
                tenant_id=self._outlook_tenant_id,
                token_cache_path=self._outlook_cache,
            )
            result = client.save_draft(
                subject=draft.subject,
                body=draft.body,
                to=draft.to,
            )
            return EmailDraft(
                to=draft.to,
                subject=draft.subject,
                body=draft.body,
                requires_confirmation=draft.requires_confirmation,
                status="draft_saved_outlook",
                outlook_message_id=result.get("message_id"),
                outlook_web_link=result.get("web_link"),
            )
        except Exception as exc:
            log.warning("Could not save draft to Outlook: %s", exc)
            return draft
