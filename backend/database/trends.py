import json
import logging
from collections import defaultdict
from typing import Dict, List

from models.trend import Trend, valid_items
from ._client import db, document_id_from_seed

logger = logging.getLogger(__name__)


def get_trends_data() -> List[Dict]:
    """Fetch all trends, returning a list of category dicts with nested topics.

    Returns:
        List of dicts with shape: [
            {
                "id": category_id,
                "category": "ceo",
                "type": "best",
                "created_at": "...",
                "data": {...},
                "topics": [
                    {"id": topic_id, "topic": "Elon Musk", "memories_count": 5, ...},
                    ...
                ]
            },
            ...
        ]
    """
    trends_data = []

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Fetch all categories
            cur.execute("SELECT id, data FROM trends_categories ORDER BY id")
            categories = [(cat_id, data) for cat_id, data in cur.fetchall()]

            # Fetch all topics
            cur.execute("SELECT category_id, id, data FROM trends_topics ORDER BY category_id, id")
            topics_by_cat = defaultdict(list)
            for cat_id, topic_id, topic_data in cur.fetchall():
                topics_by_cat[cat_id].append({**topic_data, "id": topic_id})

    # Reconstruct with nested topics
    for cat_id, cat_data in categories:
        try:
            # Ensure category is in the valid list
            if cat_data.get('category') not in [
                'ceo',
                'company',
                'software_product',
                'hardware_product',
                'ai_product',
            ]:
                continue

            # Sort topics by memories_count descending
            topics = sorted(
                topics_by_cat[cat_id],
                key=lambda e: e.get('memories_count', 0),
                reverse=True
            )

            # Filter and clean topics
            cleaned_topics = []
            for topic in topics:
                if topic.get('topic') not in valid_items:
                    continue
                cleaned_topics.append(topic)

            result_dict = {**cat_data, "id": cat_id, "topics": cleaned_topics}
            trends_data.append(result_dict)
        except Exception as e:
            logger.error(f"Error processing category {cat_id}: {e}")
            continue

    return trends_data


def save_trends(memory_id: str, trends: List[Trend]):
    """Save trends to both trends_categories and trends_topics tables.

    For each Trend:
    - Upsert a category row (ON CONFLICT update if exists)
    - For each topic in the trend, upsert a topic row

    All writes are atomic via db.batch().
    """
    with db.batch() as conn:
        with conn.cursor() as cur:
            for trend in trends:
                category = trend.category.value
                topics = trend.topics
                trend_type = trend.type.value
                category_id = document_id_from_seed(category + trend_type)

                # Upsert category
                cur.execute(
                    """
                    INSERT INTO trends_categories (id, data)
                    VALUES (%s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET data = EXCLUDED.data
                    """,
                    (category_id, json.dumps({"category": category, "type": trend_type}))
                )

                # Upsert each topic
                for topic in topics:
                    topic_id = document_id_from_seed(topic)
                    cur.execute(
                        """
                        INSERT INTO trends_topics (category_id, id, data)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (category_id, id) DO UPDATE
                        SET data = data || %s
                        """,
                        (
                            category_id,
                            topic_id,
                            json.dumps({"topic": topic, "memories_count": 1}),
                            json.dumps({"memories_count": 1})
                        )
                    )
