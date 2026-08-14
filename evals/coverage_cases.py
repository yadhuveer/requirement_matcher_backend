"""
Ground-truth cases for the `coverage`-node offline eval (requirements.md §7.2).

Unlike the extraction eval, this one bypasses retrieval entirely: each case
HARDCODES the candidate features, so we test the `coverage` node's JUDGMENT in
isolation (given these candidates, does it pick the right covering set?) — no
embeddings, no Qdrant.

The eval (coverage_eval.py) calls the real `coverage` node directly on each
case's {requirement, candidates}, reads back `selected_features` (the node maps
its internal `selected_ids` indices to candidate dicts for us), and an LLM-judge
scores the selected set BY MEANING against the ground truth below — penalising:
  - misses   : a needed feature left out of the covering set,
  - padding  : a distractor wrongly selected,
  - a wrong `relevant` flag (selected something when nothing fit, or vice versa).

Each case:
  - name               : short id.
  - requirement        : {name, description, domain} — the NEW client's ask.
  - candidates         : hardcoded list of {feature_id, name, description, domain}.
                         Order is fixed (the node selects by index internally).
  - expected_relevant  : should ANY candidate be selected?
  - expected_selected  : names of the candidates that SHOULD form the covering set
                         (empty when expected_relevant is False). Matched by MEANING.

Candidate descriptions are the real ORP (retail) library features; the
requirements are FDP (food-delivery) style asks, so most cases are cross-domain.

The 4 scenarios: single-cover, composite-cover, not-relevant, anti-padding.
"""

CASES = [
    # 1. COMPOSITE-COVER: one requirement genuinely needs a COMBINATION of three
    #    auth features; two clear distractors must be excluded.
    {
        "name": "auth_composite",
        "requirement": {
            "name": "Customer Accounts & Authentication",
            "description": (
                "Customers register using mobile number or email verified via OTP or link, "
                "log in with a password or social login (Google/Apple), and reset a forgotten "
                "password. One active account is allowed per mobile/email."
            ),
            "domain": "food delivery",
        },
        "candidates": [
            {
                "feature_id": 101,
                "name": "Self-Service Customer Registration",
                "description": (
                    "New customers create an account using their email or mobile number, with a "
                    "verification step (one-time code or email link) before the account is activated. "
                    "Each mobile/email is tied to only one active account."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 102,
                "name": "Password-Based Customer Login",
                "description": (
                    "Registered customers sign in using their email or mobile and password. After "
                    "five failed attempts within 15 minutes the account is temporarily locked."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 103,
                "name": "Social and Federated Login (SSO)",
                "description": (
                    "Customers sign in using their existing Google or Apple accounts instead of a "
                    "separate password; the social identity is matched to an account or a new one created."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 104,
                "name": "Shipping Rate Calculation",
                "description": (
                    "Calculates shipping cost based on parcel weight, dimensions, destination, and "
                    "service level, applying free-shipping thresholds where relevant."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 105,
                "name": "Verified Product Reviews and Ratings",
                "description": (
                    "Customers who purchased a product can submit a star rating and written review, "
                    "shown as an aggregate after moderation."
                ),
                "domain": "retail",
            },
        ],
        "expected_relevant": True,
        "expected_selected": [
            "Self-Service Customer Registration",
            "Password-Based Customer Login",
            "Social and Federated Login (SSO)",
        ],
    },

    # 2. SINGLE-COVER: exactly one candidate covers it; the rest are same-area
    #    admin/reporting distractors that must NOT be added.
    {
        "name": "audit_log_single",
        "requirement": {
            "name": "Immutable Audit Log",
            "description": (
                "Record every security-relevant and financial action (who did what, and when) in a "
                "tamper-proof, immutable log that authorised users can search for investigations and compliance."
            ),
            "domain": "food delivery",
        },
        "candidates": [
            {
                "feature_id": 201,
                "name": "Immutable Audit Log",
                "description": (
                    "Automatically records every security-relevant and business-critical action — who "
                    "did it, when, and what changed — in a tamper-evident log searchable by authorised users."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 202,
                "name": "Role-Based Access Control (RBAC)",
                "description": (
                    "Internal staff can only perform actions their role permits; roles are built from "
                    "granular permissions and unauthorised attempts are blocked and logged."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 203,
                "name": "Internal User Account Management",
                "description": (
                    "Administrators create, deactivate, and manage internal staff accounts and their role "
                    "assignments; deactivated users lose access immediately."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 204,
                "name": "Sales and Revenue Reporting",
                "description": (
                    "Reports on sales totals, revenue, order volumes, and average order value, filterable "
                    "by date range, channel, category, and location."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 205,
                "name": "Operational Dashboards",
                "description": (
                    "A live view of orders in every processing stage, backlogs, and exceptions, with "
                    "drill-down into the underlying orders."
                ),
                "domain": "retail",
            },
        ],
        "expected_relevant": True,
        "expected_selected": [
            "Immutable Audit Log",
        ],
    },

    # 3. NOT-RELEVANT: a genuinely novel capability (voice-call number masking) with
    #    NO covering candidate. The closest one (notifications) is different logic.
    {
        "name": "caller_masking_none",
        "requirement": {
            "name": "Privacy-Preserving Caller Masking",
            "description": (
                "When a customer calls their delivery rider (or vice versa), route the voice call through "
                "a masked-number provider so neither party ever sees the other's real phone number."
            ),
            "domain": "food delivery",
        },
        "candidates": [
            {
                "feature_id": 301,
                "name": "Multi-Channel Notification Delivery",
                "description": (
                    "Delivers notifications to customers via their preferred channel — email, SMS, push, "
                    "or WhatsApp — with fallback if the primary channel fails. Every send is logged."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 302,
                "name": "Order and Shipment Tracking",
                "description": (
                    "Real-time visibility into each shipment's delivery status by pulling carrier updates "
                    "and showing them as normalised milestones on an order-status page."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 303,
                "name": "Multiple Payment Methods",
                "description": (
                    "Customers pay using cards, UPI, net banking, digital wallets, and cash on delivery "
                    "where eligible; only applicable methods are offered."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 304,
                "name": "Faceted Search Filtering",
                "description": (
                    "Narrow search/browse results by filters such as category, price range, brand, size, "
                    "colour, rating, and availability, combined with logical AND."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 305,
                "name": "Verified Product Reviews and Ratings",
                "description": (
                    "Customers who purchased a product can submit a star rating and written review, shown "
                    "as an aggregate after moderation."
                ),
                "domain": "retail",
            },
        ],
        "expected_relevant": False,
        "expected_selected": [],
    },

    # 4. ANTI-PADDING: exactly ONE candidate covers it, but three tempting
    #    "rewards/discounts/tender"-adjacent distractors must be REJECTED.
    {
        "name": "loyalty_anti_padding",
        "requirement": {
            "name": "Loyalty Programme",
            "description": (
                "Customers earn points on qualifying orders and redeem them for discounts or perks; earn "
                "rates, redemption values, and point expiry are all configurable."
            ),
            "domain": "food delivery",
        },
        "candidates": [
            {
                "feature_id": 401,
                "name": "Loyalty Points Earn and Redemption",
                "description": (
                    "Customers accumulate loyalty points on qualifying purchases and spend them for "
                    "discounts or rewards; earn rates, redemption values, and expiry are configurable."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 402,
                "name": "Store Credit and Gift Cards",
                "description": (
                    "Customers pay using store credit or gift cards at checkout, alone or combined with "
                    "other methods; balances reduce as used and any remainder is retained."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 403,
                "name": "Coupon Code Management",
                "description": (
                    "Create and distribute single-use or multi-use coupon codes with per-code and "
                    "per-customer usage limits; invalid or expired codes are rejected with a reason."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 404,
                "name": "Customer Referral Program",
                "description": (
                    "Existing customers invite new customers via a referral link or code, and both parties "
                    "receive a reward once the referred customer completes a qualifying first purchase."
                ),
                "domain": "retail",
            },
            {
                "feature_id": 405,
                "name": "Multiple Payment Methods",
                "description": (
                    "Customers pay using cards, UPI, net banking, digital wallets, and cash on delivery "
                    "where eligible; only applicable methods are offered."
                ),
                "domain": "retail",
            },
        ],
        "expected_relevant": True,
        "expected_selected": [
            "Loyalty Points Earn and Redemption",
        ],
    },
]
