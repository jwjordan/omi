"""Tests for the JWKS-based Firebase token validator.

We sign test tokens with a locally-generated RSA keypair and serve the
corresponding JWK via a pytest fixture so `PyJWKClient` fetches our key
instead of Google's. This isolates the validator from the network.
"""

import json
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from utils.other.jwks_auth import (
    InvalidOmiTokenError,
    verify_omi_id_token,
)

OMI_PROJECT_ID = "based-hardware-test"  # test fixture value
ISSUER = f"https://securetoken.google.com/{OMI_PROJECT_ID}"


@pytest.fixture(scope="session")
def rsa_keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key


@pytest.fixture(scope="session")
def jwks_url(rsa_keypair, tmp_path_factory, monkeypatch_session):
    """Serve a JWKS over a local file:// URL and patch the validator to use it."""
    from jwt.algorithms import RSAAlgorithm

    pub = rsa_keypair.public_key()
    jwk = json.loads(RSAAlgorithm.to_jwk(pub))
    jwk["kid"] = "test-kid-1"
    jwk["use"] = "sig"
    jwk["alg"] = "RS256"

    jwks = {"keys": [jwk]}
    d = tmp_path_factory.mktemp("jwks")
    path = d / "jwks.json"
    path.write_text(json.dumps(jwks))
    url = path.as_uri()

    monkeypatch_session.setenv("OMI_FIREBASE_PROJECT_ID", OMI_PROJECT_ID)
    monkeypatch_session.setenv("OMI_JWKS_URL", url)
    import utils.other.jwks_auth as mod
    mod._reset_jwks_client_for_testing()
    try:
        yield url
    finally:
        mod._reset_jwks_client_for_testing()


@pytest.fixture(scope="session")
def monkeypatch_session(request):
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _sign(rsa_keypair, claims: dict, kid: str = "test-kid-1", alg: str = "RS256") -> str:
    pem = rsa_keypair.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return jwt.encode(claims, pem, algorithm=alg, headers={"kid": kid})


def _base_claims():
    now = int(time.time())
    return {
        "iss": ISSUER,
        "aud": OMI_PROJECT_ID,
        "sub": "uid-abc-123",
        "iat": now - 60,
        "exp": now + 3600,
        "email": "james@example.com",
    }


def test_valid_token_returns_decoded_claims(rsa_keypair, jwks_url):
    token = _sign(rsa_keypair, _base_claims())
    decoded = verify_omi_id_token(token)
    assert decoded["sub"] == "uid-abc-123"
    assert decoded["email"] == "james@example.com"


def test_expired_token_rejected(rsa_keypair, jwks_url):
    claims = _base_claims()
    claims["exp"] = int(time.time()) - 1
    token = _sign(rsa_keypair, claims)
    with pytest.raises(InvalidOmiTokenError, match="expired"):
        verify_omi_id_token(token)


def test_wrong_issuer_rejected(rsa_keypair, jwks_url):
    claims = _base_claims()
    claims["iss"] = "https://accounts.google.com"
    token = _sign(rsa_keypair, claims)
    with pytest.raises(InvalidOmiTokenError, match="issuer"):
        verify_omi_id_token(token)


def test_wrong_audience_rejected(rsa_keypair, jwks_url):
    claims = _base_claims()
    claims["aud"] = "some-other-project"
    token = _sign(rsa_keypair, claims)
    with pytest.raises(InvalidOmiTokenError, match="audience"):
        verify_omi_id_token(token)


def test_bad_signature_rejected(rsa_keypair, jwks_url):
    token = _sign(rsa_keypair, _base_claims())
    parts = token.split(".")
    parts[2] = "A" * len(parts[2])
    bad = ".".join(parts)
    with pytest.raises(InvalidOmiTokenError):
        verify_omi_id_token(bad)


def test_malformed_token_rejected(jwks_url):
    with pytest.raises(InvalidOmiTokenError):
        verify_omi_id_token("not-a-jwt")


def test_unknown_kid_rejected(rsa_keypair, jwks_url):
    token = _sign(rsa_keypair, _base_claims(), kid="unknown-kid-999")
    with pytest.raises(InvalidOmiTokenError):
        verify_omi_id_token(token)


def test_missing_bearer_scheme_not_this_functions_concern(rsa_keypair, jwks_url):
    """Sanity: this validator just takes the raw token. Bearer-prefix stripping
    is done by the caller in endpoints.py."""
    token = _sign(rsa_keypair, _base_claims())
    assert verify_omi_id_token(token)["sub"] == "uid-abc-123"
