"""Folders: collections of conversations. Postgres-backed port.

Table structure:
- folders (uid, id) PK — folder metadata stored in `data` JSONB
  (name, description, color, icon, order, is_default, is_system, category_mapping, conversation_count, etc.)

Conversations reference folders via data->>'folder_id'. When a folder is deleted,
conversations can be reparented to another folder or have their folder_id cleared.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from ._client import db

# System folders that are created for new users
SYSTEM_FOLDERS = [
    {
        'name': 'Work',
        'category_mapping': 'work',
        'icon': '💼',
        'color': '#3B82F6',
        'description': 'Work, business, professional, and career-related conversations',
    },
    {
        'name': 'Personal',
        'category_mapping': 'personal',
        'icon': '👤',
        'color': '#10B981',
        'description': 'Personal life, family, health, hobbies, and self-improvement',
    },
    {
        'name': 'Social',
        'category_mapping': 'social',
        'icon': '👥',
        'color': '#8B5CF6',
        'description': 'Friends, social gatherings, entertainment, and casual conversations',
    },
]

# Map all categories to one of the 3 system folders
CATEGORY_TO_FOLDER_MAPPING = {
    # Work folder - professional/business/career related
    'work': 'work',
    'business': 'work',
    'entrepreneurship': 'work',
    'technology': 'work',
    'finance': 'work',
    'economics': 'work',
    'legal': 'work',
    'education': 'work',  # Often career/learning related
    'science': 'work',
    'architecture': 'work',
    'design': 'work',
    # Personal folder - individual/self/family related
    'personal': 'personal',
    'health': 'personal',
    'family': 'personal',
    'parenting': 'personal',
    'romance': 'personal',
    'romantic': 'personal',
    'spiritual': 'personal',
    'inspiration': 'personal',
    'travel': 'personal',
    'sports': 'personal',
    'philosophy': 'personal',
    'psychology': 'personal',
    'literature': 'personal',
    'history': 'personal',
    # Social folder - friends/entertainment/casual related
    'social': 'social',
    'entertainment': 'social',
    'music': 'social',
    'politics': 'social',
    'news': 'social',
    'weather': 'social',
    'environment': 'social',
    'real': 'social',
    'other': 'personal',
}


def _row_to_folder(row) -> Dict[str, Any]:
    """Merge id + data JSONB into a single folder dict."""
    folder_id, data = row
    out = dict(data or {})
    out['id'] = folder_id
    return out


def get_folders(uid: str) -> List[dict]:
    """Get all folders for a user, sorted by order."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM folders
                WHERE uid = %s
                ORDER BY (data->>'order')::int NULLS LAST, created_at
                """,
                (uid,),
            )
            return [_row_to_folder(row) for row in cur.fetchall()]


def get_folder(uid: str, folder_id: str) -> Optional[dict]:
    """Get a specific folder by ID."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, data FROM folders WHERE uid = %s AND id = %s",
                (uid, folder_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_folder(row)


def create_folder(
    uid: str,
    name: str,
    description: Optional[str] = None,
    color: Optional[str] = None,
    icon: Optional[str] = None,
) -> dict:
    """Create a new custom folder for a user."""
    # Get the highest order number
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX((data->>'order')::int), -1) FROM folders WHERE uid = %s",
                (uid,),
            )
            max_order = cur.fetchone()[0]

    folder_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    folder_data = {
        'name': name,
        'description': description,
        'color': color or '#6B7280',
        'icon': icon or '📁',
        'created_at': now.isoformat(),
        'updated_at': now.isoformat(),
        'order': max_order + 1,
        'is_default': False,
        'is_system': False,
        'category_mapping': None,
        'conversation_count': 0,
    }

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO folders (uid, id, data) VALUES (%s, %s, %s::jsonb)",
                (uid, folder_id, json.dumps(folder_data)),
            )

    return {'id': folder_id, **folder_data}


def update_folder(uid: str, folder_id: str, update_data: dict) -> bool:
    """Update a folder's metadata via JSONB merge."""
    # Add updated_at timestamp
    update_data['updated_at'] = datetime.now(timezone.utc).isoformat()

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Build a list of keys to update
            set_clauses = []
            params = []
            for key, value in update_data.items():
                set_clauses.append(f"data = jsonb_set(data, %s, to_jsonb(%s))")
                params.extend([f'{{{key}}}', value])

            if not set_clauses:
                return True

            sql = f"UPDATE folders SET {', '.join(set_clauses)} WHERE uid = %s AND id = %s"
            params.extend([uid, folder_id])

            # Simpler approach: reconstruct full data and replace
            folder = get_folder(uid, folder_id)
            if not folder:
                return False

            folder.update(update_data)
            # Remove id from data before storing
            data_to_store = {k: v for k, v in folder.items() if k != 'id'}

            cur.execute(
                "UPDATE folders SET data = %s::jsonb WHERE uid = %s AND id = %s",
                (json.dumps(data_to_store), uid, folder_id),
            )

    return True


def delete_folder(uid: str, folder_id: str, move_to_folder_id: Optional[str] = None) -> bool:
    """
    Delete a folder and optionally move its conversations.
    If move_to_folder_id is provided, reparent conversations to it.
    If not provided, clear the folder_id field from conversations.
    """
    with db.batch() as conn:
        with conn.cursor() as cur:
            # Move conversations if target folder specified
            if move_to_folder_id:
                cur.execute(
                    "UPDATE conversations SET data = jsonb_set(data, '{folder_id}', to_jsonb(%s::text)) "
                    "WHERE uid = %s AND data->>'folder_id' = %s",
                    (move_to_folder_id, uid, folder_id),
                )
            else:
                # Clear folder_id from conversations
                cur.execute(
                    "UPDATE conversations SET data = data - 'folder_id' "
                    "WHERE uid = %s AND data->>'folder_id' = %s",
                    (uid, folder_id),
                )

            # Delete the folder
            cur.execute(
                "DELETE FROM folders WHERE uid = %s AND id = %s",
                (uid, folder_id),
            )
            deleted = cur.rowcount > 0

    return deleted


def reorder_folders(uid: str, folder_ids: List[str]) -> bool:
    """Reorder folders by providing an ordered list of folder IDs."""
    now = datetime.now(timezone.utc).isoformat()

    with db.batch() as conn:
        with conn.cursor() as cur:
            for i, folder_id in enumerate(folder_ids):
                cur.execute(
                    "UPDATE folders SET data = jsonb_set(jsonb_set(data, '{order}', to_jsonb(%s::int)), "
                    "'{updated_at}', to_jsonb(%s)) WHERE uid = %s AND id = %s",
                    (i, now, uid, folder_id),
                )

    return True


def initialize_system_folders(uid: str) -> List[dict]:
    """
    Create system folders for a new user or user without folders.
    Returns the list of created folders. Idempotent.
    """
    # Check if already initialized
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM folders WHERE uid = %s",
                (uid,),
            )
            if cur.fetchone()[0] > 0:
                return get_folders(uid)

    created_folders = []
    now = datetime.now(timezone.utc)

    with db.batch() as conn:
        with conn.cursor() as cur:
            for i, folder_config in enumerate(SYSTEM_FOLDERS):
                folder_id = str(uuid.uuid4())
                folder_data = {
                    'name': folder_config['name'],
                    'description': folder_config['description'],
                    'color': folder_config['color'],
                    'icon': folder_config['icon'],
                    'created_at': now.isoformat(),
                    'updated_at': now.isoformat(),
                    'order': i,
                    'is_default': folder_config['category_mapping'] == 'other',
                    'is_system': True,
                    'category_mapping': folder_config['category_mapping'],
                    'conversation_count': 0,
                }
                cur.execute(
                    "INSERT INTO folders (uid, id, data) VALUES (%s, %s, %s::jsonb)",
                    (uid, folder_id, json.dumps(folder_data)),
                )
                created_folders.append({'id': folder_id, **folder_data})

    return created_folders


def get_conversations_in_folder(
    uid: str,
    folder_id: str,
    limit: int = 100,
    offset: int = 0,
    include_discarded: bool = False,
) -> List[dict]:
    """Get all conversations in a specific folder."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            # Build query conditionally
            where_clause = "WHERE uid = %s AND data->>'folder_id' = %s"
            params = [uid, folder_id]

            if not include_discarded:
                where_clause += " AND discarded = FALSE"

            sql = (
                f"SELECT uid, id, status, discarded, created_at, started_at, finished_at, data "
                f"FROM conversations {where_clause} "
                f"ORDER BY created_at DESC LIMIT %s OFFSET %s"
            )
            params.extend([limit, offset])

            cur.execute(sql, tuple(params))
            # Return full conversation dicts by merging typed cols into data
            conversations = []
            for row in cur.fetchall():
                _uid, cid, status, discarded, created_at, started_at, finished_at, data = row
                conv = dict(data or {})
                conv['id'] = cid
                conv['status'] = status
                conv['discarded'] = discarded
                conv['created_at'] = created_at
                conv['started_at'] = started_at
                conv['finished_at'] = finished_at
                conversations.append(conv)
            return conversations


def move_conversation_to_folder(
    uid: str,
    conversation_id: str,
    folder_id: Optional[str],
) -> bool:
    """Move a conversation to a different folder."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            # Get the old folder_id to update counts
            cur.execute(
                "SELECT data->>'folder_id' FROM conversations WHERE uid = %s AND id = %s",
                (uid, conversation_id),
            )
            row = cur.fetchone()
            if not row:
                return False

            old_folder_id = row[0]

            # Update the conversation's folder_id
            if folder_id:
                cur.execute(
                    "UPDATE conversations SET data = jsonb_set(data, '{folder_id}', to_jsonb(%s::text)) "
                    "WHERE uid = %s AND id = %s",
                    (folder_id, uid, conversation_id),
                )
            else:
                cur.execute(
                    "UPDATE conversations SET data = data - 'folder_id' "
                    "WHERE uid = %s AND id = %s",
                    (uid, conversation_id),
                )

    # Update folder counts
    if old_folder_id:
        update_folder_conversation_count(uid, old_folder_id)
    if folder_id:
        update_folder_conversation_count(uid, folder_id)

    return True


def bulk_move_conversations_to_folder(
    uid: str,
    conversation_ids: List[str],
    folder_id: str,
) -> int:
    """Move multiple conversations to a folder. Returns count of moved conversations."""
    if not conversation_ids:
        return 0

    affected_folders = set()

    with db.batch() as conn:
        with conn.cursor() as cur:
            # Get old folder_ids for all conversations
            cur.execute(
                "SELECT DISTINCT data->>'folder_id' FROM conversations "
                "WHERE uid = %s AND id = ANY(%s::text[])",
                (uid, conversation_ids),
            )
            for row in cur.fetchall():
                if row[0]:
                    affected_folders.add(row[0])

            # Update all conversations to new folder
            cur.execute(
                "UPDATE conversations SET data = jsonb_set(data, '{folder_id}', to_jsonb(%s::text)) "
                "WHERE uid = %s AND id = ANY(%s::text[])",
                (folder_id, uid, conversation_ids),
            )
            moved = cur.rowcount

    affected_folders.add(folder_id)
    for fid in affected_folders:
        update_folder_conversation_count(uid, fid)

    return moved


def update_folder_conversation_count(uid: str, folder_id: str) -> int:
    """Update the conversation count for a folder."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            # Count conversations in folder that are not discarded
            cur.execute(
                "SELECT COUNT(*) FROM conversations WHERE uid = %s AND data->>'folder_id' = %s AND discarded = FALSE",
                (uid, folder_id),
            )
            count = cur.fetchone()[0]

            # Update folder's conversation_count
            cur.execute(
                "UPDATE folders SET data = jsonb_set(data, '{conversation_count}', to_jsonb(%s::int)) "
                "WHERE uid = %s AND id = %s",
                (count, uid, folder_id),
            )

    return count


def get_folder_by_category_mapping(uid: str, category_mapping: str) -> Optional[dict]:
    """Get a folder by its category_mapping value."""
    folders = get_folders(uid)
    return next((f for f in folders if f.get('category_mapping') == category_mapping), None)
