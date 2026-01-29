"""
统一记忆管理器 - PostgreSQL + (Milvus | pgvector)

- PostgreSQL: 存储结构化数据（对话历史、任务记录、决策模式等）
- Milvus/pgvector: 存储向量数据，用于语义相似度搜索

通过环境变量切换向量后端：
- MEMORY_VECTOR_BACKEND=pgvector  （默认，使用 Postgres + pgvector 扩展）
- MEMORY_VECTOR_BACKEND=milvus    （使用 Milvus）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from app.config import settings
from app.db.session import engine as default_engine

try:
    from pymilvus import (  # type: ignore
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )
except Exception:  # pragma: no cover
    Collection = None
    CollectionSchema = None
    DataType = None
    FieldSchema = None
    connections = None
    utility = None

logger = logging.getLogger(__name__)


def _normalize_vector_backend(value: Any) -> str:
    v = str(value or "").strip().lower()
    if v in {"miluvs"}:
        return "milvus"
    if v in {"pg", "postgres", "postgresql"}:
        return "pgvector"
    return v or "pgvector"


class UnifiedMemoryManager:
    """
    统一记忆管理器

    - 结构化数据：PostgreSQL（通过 app/db/session.py 的 SQLAlchemy engine）
    - 向量数据：Milvus 或 pgvector（可选）
    """

    def __init__(self, engine: Optional[Engine] = None):
        self.engine: Engine = engine or default_engine

        self.vector_backend: str = _normalize_vector_backend(
            getattr(settings, "MEMORY_VECTOR_BACKEND", "pgvector")
        )
        self.vector_dim: int = int(getattr(settings, "MEMORY_VECTOR_DIM", 1536))

        # Milvus
        self.milvus_collection = None
        self._milvus_collection_name = "agent_memory_vectors"

        # pgvector
        self.pgvector_enabled: bool = False
        self._pgvector_table = "agent_memory_vectors"

    async def initialize(self):
        """初始化 Postgres 表结构 + 可选向量后端"""
        await asyncio.to_thread(self._ensure_postgres_schema)

        if self.vector_backend == "milvus":
            try:
                await asyncio.to_thread(self._init_milvus)
                logger.info("✅ Milvus 初始化成功")
            except Exception as e:
                logger.warning("⚠️ Milvus 初始化失败，将禁用语义搜索: %s", e)
        elif self.vector_backend == "pgvector":
            await asyncio.to_thread(self._init_pgvector)
        else:
            logger.info("ℹ️ 未启用向量后端：MEMORY_VECTOR_BACKEND=%s", self.vector_backend)

        logger.info("统一记忆管理器初始化完成 | vector_backend=%s", self.vector_backend)

    # -------------------------
    # Schema (PostgreSQL)
    # -------------------------

    def _ensure_postgres_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS agent_memories (
                id BIGSERIAL PRIMARY KEY,
                agent_id VARCHAR(100) NOT NULL,
                memory_type VARCHAR(50) NOT NULL,
                content TEXT,
                metadata JSONB,
                embedding_id VARCHAR(100),
                importance_score DOUBLE PRECISION DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_agent_memories_agent_type ON agent_memories (agent_id, memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_agent_memories_importance ON agent_memories (importance_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_agent_memories_embedding_id ON agent_memories (embedding_id)",
            # 纯 Postgres 内置 FTS（对中文不如 MySQL ngram；必要时可在应用层回退 ILIKE）
            """
            CREATE INDEX IF NOT EXISTS idx_agent_memories_content_fts
            ON agent_memories
            USING gin (to_tsvector('simple', coalesce(content, '')))
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_working_memory (
                agent_id VARCHAR(100) PRIMARY KEY,
                current_goal TEXT,
                current_plan JSONB,
                recent_actions JSONB,
                context_window JSONB,
                attention_focus TEXT,
                active_memory_ids JSONB,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_associations (
                id BIGSERIAL PRIMARY KEY,
                memory_id_1 BIGINT REFERENCES agent_memories(id) ON DELETE CASCADE,
                memory_id_2 BIGINT REFERENCES agent_memories(id) ON DELETE CASCADE,
                association_type VARCHAR(50),
                strength DOUBLE PRECISION DEFAULT 0.5,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (memory_id_1, memory_id_2, association_type)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_memory_associations_strength ON memory_associations (strength DESC)",
        ]

        with self.engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))

    # -------------------------
    # Vector backend: pgvector
    # -------------------------

    def _init_pgvector(self) -> None:
        try:
            statements = [
                "CREATE EXTENSION IF NOT EXISTS vector",
                f"""
                CREATE TABLE IF NOT EXISTS {self._pgvector_table} (
                    memory_id BIGINT PRIMARY KEY REFERENCES agent_memories(id) ON DELETE CASCADE,
                    agent_id VARCHAR(100) NOT NULL,
                    memory_type VARCHAR(50) NOT NULL,
                    embedding vector({self.vector_dim}) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """,
                f"CREATE INDEX IF NOT EXISTS idx_{self._pgvector_table}_agent_type ON {self._pgvector_table} (agent_id, memory_type)",
            ]
            with self.engine.begin() as conn:
                for stmt in statements:
                    conn.execute(text(stmt))
            self.pgvector_enabled = True
            logger.info("✅ pgvector 初始化成功")

            try:
                self._ensure_pgvector_embedding_index()
            except Exception as e:
                logger.warning("⚠️ pgvector 向量索引创建失败（可忽略）: %s", e)
        except Exception as e:
            self.pgvector_enabled = False
            logger.warning("⚠️ pgvector 初始化失败（将禁用语义搜索）: %s", e)

    def _ensure_pgvector_embedding_index(self) -> None:
        if not self.pgvector_enabled:
            return

        auto_index = (os.getenv("PGVECTOR_AUTO_INDEX") or "1").strip().lower()
        if auto_index in {"0", "false", "no", "off"}:
            return

        hnsw_m = int((os.getenv("PGVECTOR_HNSW_M") or "16").strip())
        hnsw_ef_construction = int((os.getenv("PGVECTOR_HNSW_EF_CONSTRUCTION") or "64").strip())
        ivfflat_lists = int(
            (
                os.getenv("PGVECTOR_IVFFLAT_LISTS_AGENT")
                or os.getenv("PGVECTOR_IVFFLAT_LISTS")
                or "100"
            ).strip()
        )

        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    "SELECT amname FROM pg_am WHERE amname IN ('hnsw','ivfflat') ORDER BY amname"
                )
            ).fetchall()
            ams = {r[0] for r in rows}

            if "hnsw" in ams:
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS idx_{self._pgvector_table}_embedding_hnsw "
                        f"ON {self._pgvector_table} USING hnsw (embedding vector_cosine_ops) "
                        f"WITH (m={hnsw_m}, ef_construction={hnsw_ef_construction})"
                    )
                )
                conn.execute(text(f"ANALYZE {self._pgvector_table}"))
                logger.info("✅ pgvector 向量索引已就绪（HNSW）")
                return

            if "ivfflat" in ams:
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS idx_{self._pgvector_table}_embedding_ivfflat "
                        f"ON {self._pgvector_table} USING ivfflat (embedding vector_cosine_ops) "
                        f"WITH (lists={ivfflat_lists})"
                    )
                )
                conn.execute(text(f"ANALYZE {self._pgvector_table}"))
                logger.info("✅ pgvector 向量索引已就绪（IVFFLAT）")
                return

        logger.info("ℹ️ pgvector 未检测到 hnsw/ivfflat，跳过向量索引创建")

    def _to_pgvector_literal(self, embedding: np.ndarray) -> str:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size != self.vector_dim:
            raise ValueError(f"embedding 维度不匹配：期望 {self.vector_dim}，实际 {vec.size}")
        return "[" + ",".join(repr(float(x)) for x in vec.tolist()) + "]"

    def _save_to_pgvector(self, *, memory_id: int, agent_id: str, memory_type: str, embedding: np.ndarray) -> None:
        if not self.pgvector_enabled:
            return
        vec_str = self._to_pgvector_literal(embedding)
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self._pgvector_table} (memory_id, agent_id, memory_type, embedding)
                    VALUES (:memory_id, :agent_id, :memory_type, (:embedding)::vector)
                    ON CONFLICT (memory_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding
                    """
                ),
                {
                    "memory_id": memory_id,
                    "agent_id": agent_id,
                    "memory_type": memory_type,
                    "embedding": vec_str,
                },
            )

    def _search_in_pgvector(
        self,
        *,
        agent_id: str,
        query_embedding: np.ndarray,
        memory_type: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not self.pgvector_enabled:
            return []
        qvec = self._to_pgvector_literal(query_embedding)

        where = ["v.agent_id = :agent_id"]
        params: Dict[str, Any] = {"agent_id": agent_id, "qvec": qvec, "limit": limit}
        if memory_type:
            where.append("v.memory_type = :memory_type")
            params["memory_type"] = memory_type
        where_sql = " AND ".join(where)

        sql = f"""
        WITH ranked AS (
            SELECT
                memory_id,
                (1 - (embedding <=> (:qvec)::vector)) AS similarity
            FROM {self._pgvector_table} v
            WHERE {where_sql}
            ORDER BY embedding <=> (:qvec)::vector
            LIMIT :limit
        )
        SELECT
            m.id,
            m.agent_id,
            m.memory_type,
            m.content,
            m.metadata,
            m.importance_score,
            m.access_count,
            ranked.similarity
        FROM ranked
        JOIN agent_memories m ON m.id = ranked.memory_id
        ORDER BY ranked.similarity DESC
        """

        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row._mapping)
            item["similarity"] = float(item.get("similarity") or 0.0)
            results.append(item)
        return results

    # -------------------------
    # Vector backend: Milvus
    # -------------------------

    def _init_milvus(self) -> None:
        if connections is None or Collection is None or utility is None:
            raise RuntimeError("pymilvus 未安装或不可用")

        connect_kwargs: Dict[str, Any] = {
            "alias": "default",
            "host": settings.MILVUS_HOST,
            "port": int(settings.MILVUS_PORT),
        }
        if settings.MILVUS_USER and settings.MILVUS_PASSWORD:
            connect_kwargs["user"] = settings.MILVUS_USER
            connect_kwargs["password"] = settings.MILVUS_PASSWORD

        connections.connect(**connect_kwargs)

        if not utility.has_collection(self._milvus_collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="agent_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="memory_id", dtype=DataType.INT64),
                FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
                FieldSchema(name="timestamp", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields=fields, description="Agent memory embeddings for semantic search")
            self.milvus_collection = Collection(name=self._milvus_collection_name, schema=schema)
            index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
            self.milvus_collection.create_index(field_name="embedding", index_params=index_params)
        else:
            self.milvus_collection = Collection(self._milvus_collection_name)

        self.milvus_collection.load()

    def _save_to_milvus(
        self,
        *,
        agent_id: str,
        memory_id: int,
        memory_type: str,
        embedding: np.ndarray,
    ) -> None:
        if not self.milvus_collection:
            return
        try:
            vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if vec.size != self.vector_dim:
                raise ValueError(f"embedding 维度不匹配：期望 {self.vector_dim}，实际 {vec.size}")

            data = [
                [agent_id],
                [memory_id],
                [memory_type],
                [vec.tolist()],
                [int(datetime.now().timestamp())],
            ]
            self.milvus_collection.insert(data)
            self.milvus_collection.flush()
        except Exception as e:
            logger.error("保存到 Milvus 失败: %s", e)

    def _search_in_milvus(
        self,
        *,
        agent_id: str,
        query_embedding: np.ndarray,
        memory_type: Optional[str],
        limit: int,
    ) -> List[Tuple[int, float]]:
        if not self.milvus_collection:
            return []
        try:
            expr = f'agent_id == "{agent_id}"'
            if memory_type:
                expr += f' and memory_type == "{memory_type}"'

            vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
            if vec.size != self.vector_dim:
                raise ValueError(f"embedding 维度不匹配：期望 {self.vector_dim}，实际 {vec.size}")

            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.milvus_collection.search(
                data=[vec.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=["memory_id"],
            )

            out: List[Tuple[int, float]] = []
            for hits in results:
                for hit in hits:
                    mid = int(hit.entity.get("memory_id"))
                    # L2 距离 -> 0..∞，简单映射到 (0,1]
                    sim = 1.0 / (1.0 + float(getattr(hit, "distance", 0.0) or 0.0))
                    out.append((mid, sim))

            out.sort(key=lambda x: x[1], reverse=True)
            return out
        except Exception as e:
            logger.error("Milvus 搜索失败: %s", e)
            return []

    # -------------------------
    # Core APIs
    # -------------------------

    async def save_memory(
        self,
        agent_id: str,
        memory_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        importance: float = 0.5,
    ) -> int:
        return await asyncio.to_thread(
            self._save_memory_sync,
            agent_id,
            memory_type,
            content,
            metadata or {},
            embedding,
            float(importance),
        )

    def _save_memory_sync(
        self,
        agent_id: str,
        memory_type: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[np.ndarray],
        importance: float,
    ) -> int:
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        # 先保证“结构化记忆”落库成功；向量写入为 best-effort（失败不影响主流程）。
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    INSERT INTO agent_memories (agent_id, memory_type, content, metadata, importance_score)
                    VALUES (:agent_id, :memory_type, :content, (:metadata)::jsonb, :importance)
                    RETURNING id
                    """
                ),
                {
                    "agent_id": agent_id,
                    "memory_type": memory_type,
                    "content": content,
                    "metadata": meta_json,
                    "importance": importance,
                },
            ).first()
            memory_id = int(row[0]) if row else 0

        if embedding is not None and memory_id:
            if self.vector_backend == "pgvector" and self.pgvector_enabled:
                try:
                    self._save_to_pgvector(
                        memory_id=memory_id,
                        agent_id=agent_id,
                        memory_type=memory_type,
                        embedding=embedding,
                    )
                    with self.engine.begin() as conn:
                        conn.execute(
                            text("UPDATE agent_memories SET embedding_id=:eid WHERE id=:id"),
                            {"eid": f"pgvector_{memory_id}", "id": memory_id},
                        )
                except Exception as e:
                    logger.warning("pgvector 写入失败（将忽略，不影响记忆落库）: %s", e)
            elif self.vector_backend == "milvus" and self.milvus_collection:
                try:
                    self._save_to_milvus(
                        agent_id=agent_id,
                        memory_id=memory_id,
                        memory_type=memory_type,
                        embedding=embedding,
                    )
                    with self.engine.begin() as conn:
                        conn.execute(
                            text("UPDATE agent_memories SET embedding_id=:eid WHERE id=:id"),
                            {"eid": f"milvus_{memory_id}", "id": memory_id},
                        )
                except Exception as e:
                    logger.warning("Milvus 写入失败（将忽略，不影响记忆落库）: %s", e)

        return memory_id

    async def search_memories(
        self,
        agent_id: str,
        query_embedding: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10,
        use_semantic: bool = True,
    ) -> List[Dict[str, Any]]:
        limit = max(1, int(limit))

        # 1) 语义搜索（向量）
        if query_embedding is not None and use_semantic:
            if self.vector_backend == "pgvector" and self.pgvector_enabled:
                results = await asyncio.to_thread(
                    self._search_in_pgvector,
                    agent_id=agent_id,
                    query_embedding=query_embedding,
                    memory_type=memory_type,
                    limit=limit,
                )
                await self._update_access_count([int(r["id"]) for r in results if r.get("id")])
                return results

            if self.vector_backend == "milvus" and self.milvus_collection:
                pairs = await asyncio.to_thread(
                    self._search_in_milvus,
                    agent_id=agent_id,
                    query_embedding=query_embedding,
                    memory_type=memory_type,
                    limit=limit,
                )
                ids = [mid for mid, _ in pairs]
                sims = {mid: sim for mid, sim in pairs}
                rows = await asyncio.to_thread(self._fetch_memories_by_ids, agent_id, ids, memory_type)
                for r in rows:
                    mid = int(r.get("id") or 0)
                    r["similarity"] = float(sims.get(mid, 0.0))
                await self._update_access_count([int(r["id"]) for r in rows if r.get("id")])
                return rows

        # 2) 文本搜索（Postgres FTS + ILIKE 回退）
        if query_text:
            results = await asyncio.to_thread(
                self._search_by_text,
                agent_id=agent_id,
                query_text=str(query_text),
                memory_type=memory_type,
                limit=limit,
            )
            await self._update_access_count([int(r["id"]) for r in results if r.get("id")])
            return results

        return []

    def _fetch_memories_by_ids(
        self, agent_id: str, ids: Sequence[int], memory_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        if not ids:
            return []
        stmt = text(
            """
            SELECT id, agent_id, memory_type, content, metadata, importance_score, access_count
            FROM agent_memories
            WHERE agent_id = :agent_id
              AND id IN :ids
            """
        ).bindparams(bindparam("ids", expanding=True))
        params: Dict[str, Any] = {"agent_id": agent_id, "ids": list(ids)}
        if memory_type:
            stmt = text(
                """
                SELECT id, agent_id, memory_type, content, metadata, importance_score, access_count
                FROM agent_memories
                WHERE agent_id = :agent_id
                  AND memory_type = :memory_type
                  AND id IN :ids
                """
            ).bindparams(bindparam("ids", expanding=True))
            params["memory_type"] = memory_type

        with self.engine.begin() as conn:
            rows = conn.execute(stmt, params).fetchall()
        items = [dict(r._mapping) for r in rows]
        order = {int(v): i for i, v in enumerate(ids)}
        items.sort(key=lambda x: order.get(int(x.get("id") or 0), 1_000_000))
        return items

    def _search_by_text(
        self, *, agent_id: str, query_text: str, memory_type: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        base_where = ["agent_id = :agent_id"]
        params: Dict[str, Any] = {"agent_id": agent_id, "q": query_text, "limit": limit}
        if memory_type:
            base_where.append("memory_type = :memory_type")
            params["memory_type"] = memory_type

        # 先尝试 FTS
        where_fts = " AND ".join(
            base_where + ["to_tsvector('simple', coalesce(content, '')) @@ plainto_tsquery('simple', :q)"]
        )
        sql_fts = f"""
        SELECT id, agent_id, memory_type, content, metadata, importance_score, access_count,
               ts_rank(to_tsvector('simple', coalesce(content, '')), plainto_tsquery('simple', :q)) AS rank
        FROM agent_memories
        WHERE {where_fts}
        ORDER BY rank DESC, importance_score DESC, created_at DESC
        LIMIT :limit
        """

        with self.engine.begin() as conn:
            rows = conn.execute(text(sql_fts), params).fetchall()

        if not rows:
            # FTS 对中文效果一般，回退 ILIKE
            where_like = " AND ".join(base_where + ["coalesce(content, '') ILIKE '%' || :q || '%'"])
            sql_like = f"""
            SELECT id, agent_id, memory_type, content, metadata, importance_score, access_count
            FROM agent_memories
            WHERE {where_like}
            ORDER BY importance_score DESC, created_at DESC
            LIMIT :limit
            """
            with self.engine.begin() as conn:
                rows = conn.execute(text(sql_like), params).fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            item = dict(r._mapping)
            item.setdefault("similarity", 0.0)
            results.append(item)
        return results

    async def _update_access_count(self, memory_ids: List[int]):
        await asyncio.to_thread(self._update_access_count_sync, memory_ids)

    def _update_access_count_sync(self, memory_ids: Sequence[int]) -> None:
        ids = [int(x) for x in memory_ids if int(x) > 0]
        if not ids:
            return
        stmt = text(
            """
            UPDATE agent_memories
            SET access_count = access_count + 1,
                last_accessed = NOW(),
                updated_at = NOW()
            WHERE id IN :ids
            """
        ).bindparams(bindparam("ids", expanding=True))
        with self.engine.begin() as conn:
            conn.execute(stmt, {"ids": ids})

    # -------------------------
    # Working memory
    # -------------------------

    async def update_working_memory(self, agent_id: str, working_memory: Dict[str, Any]):
        await asyncio.to_thread(self._update_working_memory_sync, agent_id, working_memory or {})

    def _update_working_memory_sync(self, agent_id: str, working_memory: Dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO agent_working_memory
                        (agent_id, current_goal, current_plan, recent_actions, context_window, attention_focus, active_memory_ids, updated_at)
                    VALUES
                        (:agent_id, :current_goal, (:current_plan)::jsonb, (:recent_actions)::jsonb, (:context_window)::jsonb, :attention_focus, (:active_memory_ids)::jsonb, NOW())
                    ON CONFLICT (agent_id) DO UPDATE SET
                        current_goal = EXCLUDED.current_goal,
                        current_plan = EXCLUDED.current_plan,
                        recent_actions = EXCLUDED.recent_actions,
                        context_window = EXCLUDED.context_window,
                        attention_focus = EXCLUDED.attention_focus,
                        active_memory_ids = EXCLUDED.active_memory_ids,
                        updated_at = NOW()
                    """
                ),
                {
                    "agent_id": agent_id,
                    "current_goal": working_memory.get("current_goal"),
                    "current_plan": json.dumps(working_memory.get("current_plan", {}), ensure_ascii=False),
                    "recent_actions": json.dumps(working_memory.get("recent_actions", []), ensure_ascii=False),
                    "context_window": json.dumps(working_memory.get("context_window", []), ensure_ascii=False),
                    "attention_focus": working_memory.get("attention_focus"),
                    "active_memory_ids": json.dumps(working_memory.get("active_memory_ids", []), ensure_ascii=False),
                },
            )

    # -------------------------
    # Maintenance / Stats
    # -------------------------

    async def consolidate_memories(self, agent_id: str, max_memories: int = 1000):
        await asyncio.to_thread(self._consolidate_memories_sync, agent_id, int(max_memories))

    def _consolidate_memories_sync(self, agent_id: str, max_memories: int) -> None:
        max_memories = max(1, max_memories)
        with self.engine.begin() as conn:
            total = conn.execute(
                text("SELECT COUNT(*) FROM agent_memories WHERE agent_id = :agent_id"),
                {"agent_id": agent_id},
            ).scalar_one()

            if int(total) <= max_memories:
                return

            delete_count = int(total) - max_memories
            conn.execute(
                text(
                    """
                    DELETE FROM agent_memories
                    WHERE id IN (
                        SELECT id
                        FROM agent_memories
                        WHERE agent_id = :agent_id
                          AND importance_score < 0.3
                          AND (last_accessed IS NULL OR last_accessed < NOW() - INTERVAL '30 days')
                        ORDER BY importance_score ASC, last_accessed ASC NULLS FIRST
                        LIMIT :delete_count
                    )
                    """
                ),
                {"agent_id": agent_id, "delete_count": delete_count},
            )
        logger.info("整理记忆: 删除了 %d 条低重要性记忆", delete_count)

    async def get_memory_stats(self, agent_id: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._get_memory_stats_sync, agent_id)

    def _get_memory_stats_sync(self, agent_id: str) -> Dict[str, Any]:
        with self.engine.begin() as conn:
            overall = conn.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS total_memories,
                        AVG(importance_score) AS avg_importance,
                        MAX(created_at) AS latest_memory,
                        COALESCE(SUM(access_count), 0) AS total_accesses
                    FROM agent_memories
                    WHERE agent_id = :agent_id
                    """
                ),
                {"agent_id": agent_id},
            ).mappings().first()

            by_type = conn.execute(
                text(
                    """
                    SELECT memory_type, COUNT(*) AS count
                    FROM agent_memories
                    WHERE agent_id = :agent_id
                    GROUP BY memory_type
                    """
                ),
                {"agent_id": agent_id},
            ).mappings().all()

            working = conn.execute(
                text("SELECT * FROM agent_working_memory WHERE agent_id = :agent_id"),
                {"agent_id": agent_id},
            ).mappings().first()

        has_vector = bool(self.milvus_collection) if self.vector_backend == "milvus" else bool(self.pgvector_enabled)
        return {
            "overall": dict(overall) if overall else {},
            "by_type": [dict(r) for r in by_type],
            "working_memory": dict(working) if working else None,
            "has_vector_search": has_vector,
            "vector_backend": self.vector_backend,
        }

    async def close(self):
        """关闭连接（Milvus需要断开；pgvector无需额外处理）"""
        if self.milvus_collection and connections is not None:
            try:
                connections.disconnect("default")
            except Exception:
                pass


# 全局实例
unified_memory = UnifiedMemoryManager()
