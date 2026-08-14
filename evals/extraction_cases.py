"""
Ground-truth test cases for the extraction-agent offline eval (requirements.md §7.2).

Each case is a chunk of BRD text plus the features we EXPECT the agent to extract
from it. The eval (extraction_eval.py) runs each chunk through the extraction agent
and an LLM-judge checks — by MEANING, not exact wording — whether:
  1. every expected feature was captured (recall / completeness), and
  2. nothing junk was added (e.g. NFRs, objectives, scaffolding).

How to add a case:
  - `name`            : short id for the case.
  - `chunk_text`      : a slice of real BRD text (roughly what one chunk looks like).
  - `expected_features`: short capability descriptions we expect the agent to find.
                         Keep them plain — the judge matches on meaning.
  - `should_not_extract`: (optional) kinds of content that must NOT become features,
                          so the judge can penalise junk.
"""

CASES = [
    {
        "name": "auth_module",
        "chunk_text": """6.1 Customer Identity, Registration & Authentication

FR-1.1 Self-service customer registration. The system shall allow a new customer to register using email address or mobile number, with a verification step (OTP or email link) before the account becomes active. A mobile number or email may be associated with only one active account.

FR-1.2 Guest checkout. The system shall allow customers to place an order without creating an account, capturing only the contact and delivery details required to fulfil the order. Guest orders are linked by email/mobile so they can later be claimed by registering with the same identifier.

FR-1.3 Login with password. The system shall authenticate registered customers using their email/mobile and password. After 5 consecutive failed attempts within 15 minutes the account is temporarily locked for 30 minutes.

FR-1.4 Social and federated login (SSO). The system shall support login via Google and Apple identity providers using OAuth 2.0 / OIDC, matching the returned email to an existing account or provisioning a new one.

FR-1.5 Multi-factor authentication (MFA). The system shall allow customers to enable OTP-based MFA over SMS or an authenticator app. MFA is mandatory for high-value accounts or after a suspicious-login event.

FR-1.6 Password reset and recovery. The system shall provide a secure self-service flow to reset a forgotten password via a time-limited link or OTP; on reset, all active sessions are invalidated.

FR-1.7 Unified customer profile. The system shall maintain a single customer profile containing personal details, saved addresses, saved payment tokens, and communication preferences, consistent across web, app, and store.

FR-1.8 Consent and communication preferences. The system shall capture and honour customer consent for marketing communications per channel (email, SMS, push, WhatsApp), with opt-in/opt-out. Transactional messages are always sent regardless of marketing consent.""",
        "expected_features": [
            "Self-service customer registration using email or mobile, with verification before activation",
            "Guest checkout without creating an account",
            "Password-based login with temporary lockout after repeated failed attempts",
            "Social / federated login via Google and Apple (SSO)",
            "Multi-factor authentication (OTP via SMS or authenticator app)",
            "Password reset and recovery (self-service, invalidates other sessions)",
            "Unified customer profile consistent across channels",
            "Marketing consent and per-channel communication preferences",
        ],
        "should_not_extract": [
            "non-functional / quality requirements (performance, uptime, security standards)",
            "high-level objectives or summary statements",
        ],
    },

    # TODO: add ~2 more cases (e.g. a payments section and an inventory section)
    # covering different feature areas. Same shape as above.
]
