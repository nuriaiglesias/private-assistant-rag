from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmailDraft:
	to: str | None
	subject: str
	body: str
	requires_confirmation: bool = True
	status: str = "draft"


class EmailTool:
	"""
	Local email drafting tool.

	This implementation does not send emails. It only prepares drafts
	and keeps a placeholder for future MCP integration.
	"""

	def draft_email(
		self,
		user_request: str,
		supporting_context: str,
		to: str | None = None,
	) -> EmailDraft:
		subject = "Solicitud de informacion"
		body = (
			"Buenos dias,\n\n"
			"Me gustaria solicitar informacion sobre la siguiente consulta:\n\n"
			f"{user_request}\n\n"
			"Informacion relevante recopilada:\n"
			f"{supporting_context}\n\n"
			"Muchas gracias.\n"
		)

		return EmailDraft(
			to=to,
			subject=subject,
			body=body,
			requires_confirmation=True,
			status="draft",
		)

	def send_email(self, draft: EmailDraft, confirmed: bool = False) -> dict[str, str]:
		if not confirmed:
			return {
				"status": "not_sent",
				"reason": "Se requiere confirmacion explicita del usuario.",
			}

		return {
			"status": "not_implemented",
			"reason": "El envio real se implementara mediante MCP en una fase posterior.",
		}
