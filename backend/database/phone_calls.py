import copy
import hashlib
import json
from datetime import datetime, timezone
from typing import List, Optional

from ._client import db
from .helpers import set_data_protection_level, prepare_for_write, prepare_for_read
from utils import encryption


# ************************************************
# *********** ENCRYPTION HELPERS *****************
# ************************************************


def _hash_phone_number(phone_number: str) -> str:
    """Create a deterministic hash of a phone number for queryable lookup."""
    return hashlib.sha256(phone_number.encode('utf-8')).hexdigest()


def _prepare_phone_number_for_write(data: dict, uid: str, level: str) -> dict:
    """Encrypt phone_number field if data protection level is enhanced."""
    data = copy.deepcopy(data)
    if level == 'enhanced' and 'phone_number' in data:
        # Store hash for lookup queries
        data['phone_number_hash'] = _hash_phone_number(data['phone_number'])
        # Encrypt the actual phone number
        data['phone_number'] = encryption.encrypt(data['phone_number'], uid)
    return data


def _prepare_phone_number_for_read(data: dict, uid: str) -> dict:
    """Decrypt phone_number field if data protection level is enhanced."""
    if not data:
        return data
    data = copy.deepcopy(data)
    level = data.get('data_protection_level')
    if level == 'enhanced' and 'phone_number' in data:
        data['phone_number'] = encryption.decrypt(data['phone_number'], uid)
    return data


# ************************************************
# *********** VERIFIED PHONE NUMBERS *************
# ************************************************


@set_data_protection_level(data_arg_name='phone_number_data')
@prepare_for_write(data_arg_name='phone_number_data', prepare_func=_prepare_phone_number_for_write)
def upsert_phone_number(uid: str, phone_number_data: dict):
    """Create or update a verified phone number for a user."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO phone_calls (uid, id, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                SET data = EXCLUDED.data
                """,
                (uid, phone_number_data['id'], json.dumps(phone_number_data)),
            )


@prepare_for_read(decrypt_func=_prepare_phone_number_for_read)
def get_phone_numbers(uid: str) -> List[dict]:
    """Get all verified phone numbers for a user."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data
                FROM phone_calls
                WHERE uid = %s
                ORDER BY created_at DESC
                """,
                (uid,),
            )
            items = []
            for row in cur.fetchall():
                phone_id, row_data = row
                result = dict(row_data or {})
                result['id'] = phone_id
                items.append(result)
            return items


@prepare_for_read(decrypt_func=_prepare_phone_number_for_read)
def get_phone_number(uid: str, phone_number_id: str) -> Optional[dict]:
    """Get a specific verified phone number."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data
                FROM phone_calls
                WHERE uid = %s AND id = %s
                """,
                (uid, phone_number_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            phone_id, row_data = row
            result = dict(row_data or {})
            result['id'] = phone_id
            return result


def get_phone_number_by_number(uid: str, phone_number: str) -> Optional[dict]:
    """Get a verified phone number by the actual phone number string.

    For enhanced protection, queries by hash since the phone_number field is encrypted.
    Falls back to plaintext query for standard protection (backward compatibility).
    """
    phone_hash = _hash_phone_number(phone_number)

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Try hash-based lookup first (encrypted records)
            cur.execute(
                """
                SELECT id, data
                FROM phone_calls
                WHERE uid = %s AND data->>'phone_number_hash' = %s
                LIMIT 1
                """,
                (uid, phone_hash),
            )
            row = cur.fetchone()
            if row is not None:
                phone_id, row_data = row
                result = dict(row_data or {})
                result['id'] = phone_id
                return _prepare_phone_number_for_read(result, uid)

            # Fallback: plaintext query for records written before encryption was enabled
            cur.execute(
                """
                SELECT id, data
                FROM phone_calls
                WHERE uid = %s AND data->>'phone_number' = %s
                LIMIT 1
                """,
                (uid, phone_number),
            )
            row = cur.fetchone()
            if row is not None:
                phone_id, row_data = row
                result = dict(row_data or {})
                result['id'] = phone_id
                return result

    return None


def delete_phone_number(uid: str, phone_number_id: str):
    """Delete a verified phone number."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM phone_calls
                WHERE uid = %s AND id = %s
                """,
                (uid, phone_number_id),
            )


@prepare_for_read(decrypt_func=_prepare_phone_number_for_read)
def get_primary_phone_number(uid: str) -> Optional[dict]:
    """Get the user's primary verified phone number."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data
                FROM phone_calls
                WHERE uid = %s AND data->>'is_primary' = 'true'
                LIMIT 1
                """,
                (uid,),
            )
            row = cur.fetchone()
            if row is not None:
                phone_id, row_data = row
                result = dict(row_data or {})
                result['id'] = phone_id
                return result

    # Fallback to first available number
    all_numbers = get_phone_numbers(uid)
    if all_numbers:
        return all_numbers[0]
    return None


# ************************************************
# ********** PENDING VERIFICATIONS ***************
# ************************************************

PENDING_VERIFICATION_TTL_SECONDS = 300  # 5 minutes


def set_pending_verification(uid: str, phone_number: str):
    """Record that a user initiated verification for a phone number.

    Uses a hash of the phone number as the document ID for efficient lookup.
    """
    doc_id = _hash_phone_number(phone_number)
    now = datetime.now(timezone.utc)
    data = {
        'uid': uid,
        'phone_number_hash': doc_id,
        'created_at': now.isoformat(),
    }
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pending_verifications (id, data)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE
                SET data = EXCLUDED.data
                """,
                (doc_id, json.dumps(data)),
            )


def get_pending_verification_uid(phone_number: str) -> Optional[str]:
    """Get the UID of the user who initiated verification for a phone number.

    Returns None if no pending verification exists or if it has expired.
    """
    doc_id = _hash_phone_number(phone_number)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT data
                FROM pending_verifications
                WHERE id = %s
                """,
                (doc_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None

            row_data = row[0] or {}
            created_at = datetime.fromisoformat(row_data['created_at'])
            elapsed = (datetime.now(timezone.utc) - created_at).total_seconds()
            if elapsed > PENDING_VERIFICATION_TTL_SECONDS:
                cur.execute(
                    """
                    DELETE FROM pending_verifications
                    WHERE id = %s
                    """,
                    (doc_id,),
                )
                return None
            return row_data.get('uid')


def delete_pending_verification(phone_number: str):
    """Delete a pending verification record after it has been processed."""
    doc_id = _hash_phone_number(phone_number)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM pending_verifications
                WHERE id = %s
                """,
                (doc_id,),
            )
