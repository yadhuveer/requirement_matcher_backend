"""
Ground-truth cases for the `classify`-node offline eval (requirements.md §7.2).

`classify` decides, for a requirement + the covering set of features coverage
already picked, whether that set is an "exact_match" (reusable essentially as-is)
or "needs_modification" (logic reusable but must be adapted). It's a BINARY label,
so this eval scores by EXACT label match (predicted == expected), no LLM-judge.

The eval (classify_eval.py) calls the real `classify` node directly on each case's
{requirement, selected_features} — bypassing BOTH search and coverage (we hardcode
the already-selected set) — and compares the returned `status` to `expected_status`.

Each case:
  - name             : short id.
  - requirement      : {name, description, domain}. classify's prompt uses domain +
                       description (name is kept for readability only).
  - selected_features: the covering set coverage would have handed to classify,
                       hardcoded. EACH feature MUST have name + domain + description
                       (the node indexes those keys directly).
  - expected_status  : "exact_match" | "needs_modification".

Candidate/feature descriptions are the real ORP (retail) library features; the
requirements are FDP (food-delivery) style, so most cases are cross-domain.

Scenarios: exact single, exact over a set, needs_modification (domain + extra
concepts), needs_modification (missing behaviour over a set).
"""

CASES = [
    # 1. EXACT_MATCH — single, domain-agnostic feature reused as-is.
    #    An audit log is an audit log regardless of retail vs food; no real change.
    {
        "name": "audit_log_exact",
        "requirement": {
            "name": "Immutable Audit Log",
            "description": (
                "Record every security-relevant and financial action (who did what, and when) in a "
                "tamper-proof, immutable log that authorised users can search for investigations and compliance."
            ),
            "domain": "food delivery",
        },
        "selected_features": [
            {
                "feature_id": 201,
                "name": "Immutable Audit Log",
                "description": (
                    "Automatically records every security-relevant and business-critical action — who "
                    "did it, when, and what changed — in a tamper-evident log searchable by authorised users."
                ),
                "domain": "retail",
            },
        ],
        "expected_status": "exact_match",
    },

    # 2. EXACT_MATCH — over a SET. "Sign in with password OR Google/Apple" is fully
    #    covered by the two existing login features together; login is domain-agnostic.
    {
        "name": "login_set_exact",
        "requirement": {
            "name": "Customer Sign-In Options",
            "description": (
                "Let customers sign in either with their password or via their existing Google or Apple "
                "account. No separate behaviour beyond standard password and social sign-in is needed."
            ),
            "domain": "food delivery",
        },
        "selected_features": [
            {
                "feature_id": 102,
                "name": "Password-Based Customer Login",
                "description": (
                    "Registered customers sign in using their email or mobile and password. After five "
                    "failed attempts within 15 minutes the account is temporarily locked."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 103,
                "name": "Social and Federated Login (SSO)",
                "description": (
                    "Customers sign in using their existing Google or Apple accounts instead of a separate "
                    "password; the social identity is matched to an account or a new one created."
                ),
                "domain": "retail",
            },
        ],
        "expected_status": "exact_match",
    },

    # 3. NEEDS_MODIFICATION — domain re-skin PLUS lots of extra concepts the existing
    #    feature doesn't have (menu, dishes, modifiers, veg flags, availability toggle).
    {
        "name": "menu_management_needs_mod",
        "requirement": {
            "name": "Restaurant Menu Management",
            "description": (
                "Restaurant partners build a menu with categories and dishes (price, veg/non-veg flag, tax "
                "class, packaging charge), define customisable modifiers/add-ons with price deltas, mark "
                "individual dishes out of stock, and set weekly opening hours. Changes need publisher approval."
            ),
            "domain": "food delivery",
        },
        "selected_features": [
            {
                "feature_id": 207,
                "name": "Catalog Publishing Workflow",
                "description": (
                    "A draft -> review -> approve -> publish process so catalog changes are checked and "
                    "approved by a Publisher before going live. All approvals are logged."
                ),
                "domain": "retail",
            },
        ],
        "expected_status": "needs_modification",
    },

    # 4. NEEDS_MODIFICATION — the set covers the CORE but the requirement needs extra
    #    behaviour the features lack (OTP login, password reset, profile/dietary/consent).
    {
        "name": "auth_missing_behaviour_needs_mod",
        "requirement": {
            "name": "Customer Accounts & Authentication",
            "description": (
                "Customers register via OTP/email, sign in with password, OTP, OR social login, reset a "
                "forgotten password (signing out other sessions), and manage a profile with dietary "
                "preferences, multiple saved addresses, and per-channel communication consent."
            ),
            "domain": "food delivery",
        },
        "selected_features": [
            {
                "feature_id": 101,
                "name": "Self-Service Customer Registration",
                "description": (
                    "New customers create an account using their email or mobile number, with a verification "
                    "step (one-time code or email link) before activation."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 102,
                "name": "Password-Based Customer Login",
                "description": (
                    "Registered customers sign in using their email or mobile and password, with temporary "
                    "lockout after repeated failed attempts."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 103,
                "name": "Social and Federated Login (SSO)",
                "description": (
                    "Customers sign in using their existing Google or Apple accounts instead of a separate password."
                ),
                "domain": "retail",
            },
        ],
        "expected_status": "needs_modification",
    },
]
