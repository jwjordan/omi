"""Knowledge graph nodes and edges backed by Postgres.

Two tables:

- knowledge_graph_nodes    (uid, id) PK — stores node label, type, data JSONB.
- knowledge_graph_edges    (uid, id) PK — stores edge relation, source/target node IDs, data JSONB.

Replaces the former Firestore structure:
  users/{uid}/knowledge_graph/nodes/{id}
  users/{uid}/knowledge_graph/edges/{id}

The data model supports:
- KnowledgeNode with id, label, node_type, aliases, memory_ids, timestamps
- KnowledgeEdge with id, source_id, target_id, label, memory_ids, timestamps

ID sanitization: edge IDs have '/' replaced with '_' (issue #4929) to ensure
valid Postgres identifiers. This logic is preserved exactly from Firestore.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from ._client import db


def _json_default(value):
    """JSON serializer for datetime values going into JSONB."""
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class KnowledgeNode:
    def __init__(
        self,
        id: str,
        label: str,
        node_type: str = 'concept',
        aliases: List[str] = None,
        memory_ids: List[str] = None,
        created_at: datetime = None,
        updated_at: datetime = None,
    ):
        self.id = id
        self.label = label
        self.node_type = node_type
        self.aliases = aliases or []
        self.memory_ids = memory_ids or []
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.label_lower = label.lower() if label else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'label': self.label,
            'node_type': self.node_type,
            'aliases': self.aliases,
            'memory_ids': self.memory_ids,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'label_lower': self.label_lower,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'KnowledgeNode':
        return KnowledgeNode(
            id=data.get('id'),
            label=data.get('label'),
            node_type=data.get('node_type', 'concept'),
            aliases=data.get('aliases', []),
            memory_ids=data.get('memory_ids', []),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
        )


class KnowledgeEdge:
    def __init__(
        self,
        id: str,
        source_id: str,
        target_id: str,
        label: str,
        memory_ids: List[str] = None,
        created_at: datetime = None,
    ):
        self.id = id
        self.source_id = source_id
        self.target_id = target_id
        self.label = label
        self.memory_ids = memory_ids or []
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'label': self.label,
            'memory_ids': self.memory_ids,
            'created_at': self.created_at,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'KnowledgeEdge':
        return KnowledgeEdge(
            id=data.get('id'),
            source_id=data.get('source_id'),
            target_id=data.get('target_id'),
            label=data.get('label'),
            memory_ids=data.get('memory_ids', []),
            created_at=data.get('created_at'),
        )


def get_knowledge_nodes(uid: str) -> List[Dict[str, Any]]:
    """Fetch all knowledge nodes for a user."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, label, node_type, data
                FROM knowledge_graph_nodes
                WHERE uid = %s
                ORDER BY created_at
                """,
                (uid,),
            )
            nodes = []
            for node_id, label, node_type, data in cur.fetchall():
                node_dict = dict(data or {})
                node_dict['id'] = node_id
                node_dict['label'] = label
                node_dict['node_type'] = node_type
                nodes.append(node_dict)
            return nodes


def get_knowledge_node(uid: str, node_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single knowledge node by ID."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, label, node_type, data
                FROM knowledge_graph_nodes
                WHERE uid = %s AND id = %s
                """,
                (uid, node_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            node_id, label, node_type, data = row
            node_dict = dict(data or {})
            node_dict['id'] = node_id
            node_dict['label'] = label
            node_dict['node_type'] = node_type
            return node_dict


def upsert_knowledge_node(uid: str, node_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert or update a knowledge node. Merges aliases and memory_ids with existing.

    If no ID is provided, tries to find by label/alias. If still not found, generates a UUID.
    """
    node_id = node_data.get('id')
    if not node_id:
        existing_node = find_node_by_label_or_alias(uid, node_data.get('label', ''))
        if existing_node:
            node_id = existing_node['id']
            node_data['id'] = node_id
        else:
            node_id = str(uuid.uuid4())
        node_data['id'] = node_id

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Check if node exists
            cur.execute(
                "SELECT data FROM knowledge_graph_nodes WHERE uid = %s AND id = %s",
                (uid, node_id),
            )
            existing_row = cur.fetchone()
            existing_data = dict(existing_row[0]) if existing_row else None

            if existing_data:
                # Merge memory_ids and aliases
                existing_memory_ids = set(existing_data.get('memory_ids', []))
                new_memory_ids = set(node_data.get('memory_ids', []))
                merged_memory_ids = list(existing_memory_ids | new_memory_ids)

                existing_aliases = set(existing_data.get('aliases', []))
                new_aliases = set(node_data.get('aliases', []))
                merged_aliases = list(existing_aliases | new_aliases)

                node_data['memory_ids'] = merged_memory_ids
                node_data['aliases'] = merged_aliases
                node_data['updated_at'] = datetime.now(timezone.utc)
                node_data['created_at'] = existing_data.get('created_at', datetime.now(timezone.utc))
            else:
                node_data['created_at'] = datetime.now(timezone.utc)
                node_data['updated_at'] = datetime.now(timezone.utc)

            node_data['label_lower'] = node_data.get('label', '').lower()
            node_data['aliases_lower'] = [a.lower() for a in node_data.get('aliases', [])]

            # Extract typed columns
            label = node_data.pop('label', '')
            node_type = node_data.pop('node_type', 'concept')

            # Store everything else in data JSONB
            cur.execute(
                """
                INSERT INTO knowledge_graph_nodes (uid, id, label, node_type, data)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                    SET label = EXCLUDED.label,
                        node_type = EXCLUDED.node_type,
                        data = EXCLUDED.data
                """,
                (uid, node_id, label, node_type, json.dumps(node_data, default=_json_default)),
            )

    # Restore for return value
    node_data['label'] = label
    node_data['node_type'] = node_type
    node_data['id'] = node_id
    return node_data


def find_node_by_label_or_alias(uid: str, label: str) -> Optional[Dict[str, Any]]:
    """Find a node by label or alias (case-insensitive)."""
    if not label:
        return None

    label_lower = label.lower()

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Check exact label match
            cur.execute(
                """
                SELECT id, label, node_type, data
                FROM knowledge_graph_nodes
                WHERE uid = %s AND (data->>'label_lower') = %s
                LIMIT 1
                """,
                (uid, label_lower),
            )
            row = cur.fetchone()
            if row:
                node_id, label, node_type, data = row
                node_dict = dict(data or {})
                node_dict['id'] = node_id
                node_dict['label'] = label
                node_dict['node_type'] = node_type
                return node_dict

            # Check alias match (JSONB array contains)
            cur.execute(
                """
                SELECT id, label, node_type, data
                FROM knowledge_graph_nodes
                WHERE uid = %s AND data->'aliases_lower' ? %s
                LIMIT 1
                """,
                (uid, label_lower),
            )
            row = cur.fetchone()
            if row:
                node_id, label, node_type, data = row
                node_dict = dict(data or {})
                node_dict['id'] = node_id
                node_dict['label'] = label
                node_dict['node_type'] = node_type
                return node_dict

    return None


def get_knowledge_edges(uid: str) -> List[Dict[str, Any]]:
    """Fetch all knowledge edges for a user."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, source_node_id, target_node_id, relation, data
                FROM knowledge_graph_edges
                WHERE uid = %s
                ORDER BY created_at
                """,
                (uid,),
            )
            edges = []
            for edge_id, source_id, target_id, relation, data in cur.fetchall():
                edge_dict = dict(data or {})
                edge_dict['id'] = edge_id
                edge_dict['source_id'] = source_id
                edge_dict['target_id'] = target_id
                edge_dict['label'] = relation
                edges.append(edge_dict)
            return edges


def upsert_knowledge_edge(uid: str, edge_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert or update a knowledge edge. Merges memory_ids with existing.

    Sanitizes edge ID by replacing '/' with '_' (issue #4929).
    """
    edge_id = edge_data.get('id')
    if not edge_id:
        edge_id = f"{edge_data['source_id']}_{edge_data['label']}_{edge_data['target_id']}"
    edge_id = edge_id.replace('/', '_')
    edge_data['id'] = edge_id

    source_id = edge_data.get('source_id', '')
    target_id = edge_data.get('target_id', '')
    relation = edge_data.get('label', '')

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Check if edge exists
            cur.execute(
                "SELECT data FROM knowledge_graph_edges WHERE uid = %s AND id = %s",
                (uid, edge_id),
            )
            existing_row = cur.fetchone()
            existing_data = dict(existing_row[0]) if existing_row else None

            if existing_data:
                # Merge memory_ids
                existing_memory_ids = set(existing_data.get('memory_ids', []))
                new_memory_ids = set(edge_data.get('memory_ids', []))
                merged_memory_ids = list(existing_memory_ids | new_memory_ids)

                edge_data['memory_ids'] = merged_memory_ids
                edge_data['created_at'] = existing_data.get('created_at', datetime.now(timezone.utc))
            else:
                edge_data['created_at'] = datetime.now(timezone.utc)

            # Remove typed columns from data blob
            edge_data.pop('source_id', None)
            edge_data.pop('target_id', None)
            edge_data.pop('label', None)

            cur.execute(
                """
                INSERT INTO knowledge_graph_edges (uid, id, source_node_id, target_node_id, relation, data)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                    SET source_node_id = EXCLUDED.source_node_id,
                        target_node_id = EXCLUDED.target_node_id,
                        relation = EXCLUDED.relation,
                        data = EXCLUDED.data
                """,
                (uid, edge_id, source_id, target_id, relation, json.dumps(edge_data, default=_json_default)),
            )

    # Restore for return value
    edge_data['id'] = edge_id
    edge_data['source_id'] = source_id
    edge_data['target_id'] = target_id
    edge_data['label'] = relation
    return edge_data


def get_knowledge_graph(uid: str) -> Dict[str, Any]:
    """Fetch the full knowledge graph (nodes + edges) for a user."""
    return {
        'nodes': get_knowledge_nodes(uid),
        'edges': get_knowledge_edges(uid),
    }


def delete_knowledge_graph(uid: str) -> None:
    """
    Delete all nodes and edges for a user in a single atomic transaction.
    """
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM knowledge_graph_edges WHERE uid = %s",
                (uid,),
            )
            cur.execute(
                "DELETE FROM knowledge_graph_nodes WHERE uid = %s",
                (uid,),
            )
