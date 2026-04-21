"""JWKS-based verifier for Firebase ID tokens issued by Omi's production Firebase project.

Replaces firebase_admin.auth.verify_id_token() at the single auth chokepoint in
utils/other/endpoints.py:verify_token. We do NOT use Firebase Admin SDK for token
verification because the Admin SDK is tied to our own Firebase project, and Omi's app
presents tokens signed for Omi's project. We fetch Google's secure-token public keys
directly and validate iss/aud/exp ourselves.

Env vars:
    OMI_FIREBASE_PROJECT_ID: Omi's public Firebase project ID (used as JWT audience).
    OMI_JWKS_URL: JWKS endpoint URL. Default: Google's secure-token x509 endpoint
                  (converted to JWK format). For tests, point at a file:// URL.
"""

import os
from typing import Optional

import jwt
from jwt import PyJWKClient, InvalidTokenError


class InvalidOmiTokenError(Exception):
    """Raised when an Omi-issued Firebase ID token fails validation."""


_DEFAULT_JWKS_URL = (
    "https://www.googleapis.com/service_accounts/v1/jwk/"
    "securetoken@system.gserviceaccount.com"
)

_jwks_client: Optional[PyJWKClient] = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        url = os.environ.get("OMI_JWKS_URL", _DEFAULT_JWKS_URL)
        _jwks_client = PyJWKClient(url, cache_keys=True, lifespan=3600)
    return _jwks_client


def _reset_jwks_client_for_testing() -> None:
    """Reset the module-level client cache so tests can swap OMI_JWKS_URL."""
    global _jwks_client
    _jwks_client = None


def verify_omi_id_token(token: str) -> dict:
    """Validate an Omi-issued Firebase ID token.

    Returns the decoded claims dict on success; raises InvalidOmiTokenError
    on any validation failure.
    """
    project_id = os.environ.get("OMI_FIREBASE_PROJECT_ID")
    if not project_id:
        raise InvalidOmiTokenError(
            "OMI_FIREBASE_PROJECT_ID env var not configured"
        )

    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token).key
    except Exception as e:
        raise InvalidOmiTokenError(f"failed to resolve signing key: {e}") from e

    try:
        decoded = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=project_id,
            issuer=f"https://securetoken.google.com/{project_id}",
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise InvalidOmiTokenError("token expired") from e
    except jwt.InvalidAudienceError as e:
        raise InvalidOmiTokenError("invalid audience") from e
    except jwt.InvalidIssuerError as e:
        raise InvalidOmiTokenError("invalid issuer") from e
    except InvalidTokenError as e:
        raise InvalidOmiTokenError(f"invalid token: {e}") from e

    return decoded
