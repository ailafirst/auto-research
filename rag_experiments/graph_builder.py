"""Chunk 相似度图 —— 从 Qdrant 读向量 → 余弦矩阵 → Louvain 社区。

用法（被 run.py 调用）:
  graph = await build_graph(vector_store, task_id)
  if graph:
      comm_id = graph.key_to_community.get("src_0000#3")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities

logger = logging.getLogger(__name__)


@dataclass
class ChunkGraph:
    """任务级的 chunk 相似度图与社区结构。"""
    communities: list[dict]           # [{id, size, keys}]
    key_to_community: dict[str, int]  # "src_i#j" → community_id
    chunk_lookup: dict[str, dict]     # "src_i#j" → {text, url, title, ...}
    total_chunks: int
    threshold_used: float


async def build_graph(
    vector_store: Any,
    task_id: str,
    threshold: float = 0.70,
    min_community_size: int = 2,
) -> ChunkGraph | None:
    """从 Qdrant 读取 task_id 的全部向量，构建相似度图并检测社区。"""
    from qdrant_client import models

    # 1. Scroll 所有 points（含向量 + payload）
    all_points: list[Any] = []
    next_offset: Any = None
    while True:
        result = await vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="task_id", match=models.MatchValue(value=task_id),
                    )
                ],
            ),
            with_vectors=True,
            with_payload=True,
            limit=5000,
            offset=next_offset,
        )
        points, next_offset = result
        all_points.extend(points)
        if next_offset is None:
            break

    n = len(all_points)
    if n < 10:
        logger.info("GraphBuilder: 跳过（仅 %d chunks < 10）", n)
        return None

    # 2. 向量矩阵 + chunk key 映射
    vectors = np.array([p.vector for p in all_points], dtype=np.float32)
    keys: list[str] = []
    lookup: dict[str, dict] = {}
    for p in all_points:
        pl = p.payload
        key = f"{pl.get('source_id', '')}#{pl.get('chunk_index', 0)}"
        keys.append(key)
        lookup[key] = {
            "text": pl.get("text", ""),
            "url": pl.get("url", ""),
            "title": pl.get("title", ""),
            "source_id": pl.get("source_id", ""),
            "chunk_index": pl.get("chunk_index", 0),
        }

    # 3. 余弦相似度矩阵
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    sim = (vectors / np.clip(norm, 1e-8, None)) @ (
        vectors / np.clip(norm, 1e-8, None)
    ).T

    # 4. NetworkX 图
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = np.argwhere(sim > threshold)
    edges = edges[edges[:, 0] < edges[:, 1]]  # 上三角
    if len(edges) == 0:
        logger.info("GraphBuilder: 无边（threshold=%.2f 过高）", threshold)
        return None
    G.add_edges_from((int(i), int(j)) for i, j in edges)

    # 5. Louvain 社区检测
    comms = louvain_communities(G, seed=42)

    # 6. 格式化输出
    comm_list: list[dict] = []
    key_to_comm: dict[str, int] = {}
    for cid, members in enumerate(comms):
        member_keys = sorted(keys[i] for i in members)
        if len(member_keys) < min_community_size:
            continue
        for k in member_keys:
            key_to_comm[k] = cid
        comm_list.append({"id": cid, "size": len(member_keys), "keys": member_keys})

    if not comm_list:
        logger.info("GraphBuilder: 所有社区小于 min_community_size=%d", min_community_size)
        return None

    logger.info(
        "GraphBuilder: %d chunks → %d 社区（threshold=%.2f）",
        n, len(comm_list), threshold,
    )
    return ChunkGraph(
        communities=comm_list,
        key_to_community=key_to_comm,
        chunk_lookup=lookup,
        total_chunks=n,
        threshold_used=threshold,
    )
