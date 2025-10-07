# src/services/batch_builder.py

import dataclasses
import numpy as np
from sklearn.cluster import KMeans
from langchain_community.vectorstores.chroma import Chroma
from ..models import Chunk
import config


class BatchBuilder:
    """
    유사도 분석을 통해 코드 청크들을 의미있는 배치(그룹)로 묶는 역할을 담당합니다.
    '최초 실행' 모드와 '델타 업데이트' 모드를 지원합니다.
    """

    def __init__(self, vector_store: Chroma, all_chunks: list[Chunk]):
        self.vector_store = vector_store
        self.all_chunks = all_chunks
        self.chunk_map = {chunk.chunk_id: chunk for chunk in all_chunks}
        print(f"[BatchBuilder] Initialized with {len(all_chunks)} chunks.")

    # --- Public Method (Dispatcher) ---
    def build_batches(
        self,
        similarity_threshold: float,
        previous_batches: list[dict] | None = None,
        new_chunk_ids: set[str] | None = None,
    ) -> list[dict]:
        """상황에 따라 '최초 실행', '변경 없음', '델타 업데이트' 로직을 호출하는 지휘자 메서드."""

        # 시나리오 1: 최초 실행
        if not previous_batches:
            print("[BatchBuilder] No previous batches found. Starting full build.")
            return self._build_from_scratch(similarity_threshold)

        # 시나리오 2: 변경 없음
        if not new_chunk_ids:
            print(
                "[BatchBuilder] Delta update: No new chunks to process. Reusing previous batches."
            )
            return previous_batches

        # 시나리오 3: 변경 있음 (델타 업데이트)
        print("[BatchBuilder] Delta update mode started.")
        return self._update_incrementally(
            previous_batches, new_chunk_ids, similarity_threshold
        )

    # --- Private Main Methods ---
    def _build_from_scratch(self, similarity_threshold: float) -> list[dict]:
        """(최초 실행 로직) 2단계 클러스터링으로 처음부터 모든 배치를 생성합니다."""
        initial_clusters = self._phase1_discover_initial_clusters(similarity_threshold)
        if not initial_clusters:
            return []

        all_embeddings, chunk_id_to_index = self._get_all_embeddings()

        final_batches = self._phase2_refine_clusters_with_kmeans(
            initial_clusters, all_embeddings, chunk_id_to_index
        )
        if not final_batches:
            return []

        return self._format_output_batches(
            final_batches, all_embeddings, chunk_id_to_index
        )

    def _update_incrementally(
        self,
        previous_batches: list[dict],
        new_chunk_ids: set[str],
        similarity_threshold: float,
    ) -> list[dict]:
        """(델타 업데이트 로직) 변경된 청크만 기존 배치에 할당하거나 신규 배치로 생성합니다."""
        all_embeddings, chunk_id_to_index = self._get_all_embeddings()

        centroids = {b["batch_id"]: np.array(b["centroid"]) for b in previous_batches}
        batch_map = {b["batch_id"]: b for b in previous_batches}
        new_chunks = [
            self.chunk_map[cid] for cid in new_chunk_ids if cid in self.chunk_map
        ]

        distance_threshold = 1 - similarity_threshold

        for new_chunk in new_chunks:
            new_vector = all_embeddings[chunk_id_to_index[new_chunk.chunk_id]]

            closest_batch_id, min_distance = self._find_closest_centroid(
                new_vector, centroids
            )

            if closest_batch_id and min_distance <= distance_threshold:
                print(
                    f"  -> Assigning new chunk {new_chunk.chunk_id[:8]}... to existing batch {closest_batch_id}"
                )
                similarity = max(0, 1 - min_distance)
                batch_map[closest_batch_id]["chunks"].append(
                    {
                        "similarity_score": round(similarity, 4),
                        "chunk": dataclasses.asdict(new_chunk),
                    }
                )

        # TODO: 어떤 배치에도 속하지 못한 새로운 청크들끼리 새로운 클러스터를 형성하는 로직

        return list(batch_map.values())

    # --- Private Helper Methods ---
    def _phase1_discover_initial_clusters(
        self, similarity_threshold: float
    ) -> list[list[str]]:
        """(Phase 1) 시드 탐색을 통해 대략적인 클러스터를 찾습니다."""
        print("[BatchBuilder] Phase 1: Discovering initial seeds...")
        unassigned_chunk_ids = set(self.chunk_map.keys())
        initial_clusters = []
        distance_threshold = 1 - similarity_threshold

        while unassigned_chunk_ids:
            seed_id = unassigned_chunk_ids.pop()
            seed_chunk = self.chunk_map[seed_id]

            # TODO:config에서 K값을 설정하는 걸 고려.
            # k 값을 전체 개수로 잡음.
            # similar_docs_with_scores = self.vector_store.similarity_search_with_score(
            #     seed_chunk.chunk_content, k=len(self.all_chunks)
            # )

            # k 값을 전체 개수가 아닌, 합리적인 숫자로 변경 (예: 20)
            similar_docs_with_scores = self.vector_store.similarity_search_with_score(
                seed_chunk.chunk_content,
                k=50,
                filter={"chunk_method": config.CHUNKING_STRATEGY},
            )
            similar_chunk_ids = {
                doc.metadata["chunk_id"]
                for doc, score in similar_docs_with_scores
                if score <= distance_threshold
            }

            current_cluster_ids = unassigned_chunk_ids.intersection(similar_chunk_ids)
            if len(current_cluster_ids) > 1:
                initial_clusters.append(list(current_cluster_ids))

            unassigned_chunk_ids.difference_update(current_cluster_ids)

        print(
            f"[BatchBuilder] Phase 1 Done: Found {len(initial_clusters)} initial clusters."
        )
        return initial_clusters

    def _get_all_embeddings(self) -> tuple[np.ndarray, dict]:
        """DB에서 모든 임베딩 벡터와 ID-인덱스 맵을 가져옵니다."""
        all_data = self.vector_store.get(include=["embeddings"])
        chunk_id_to_index = {id_str: i for i, id_str in enumerate(all_data["ids"])}
        all_embeddings = np.array(all_data["embeddings"])
        return all_embeddings, chunk_id_to_index

    def _phase2_refine_clusters_with_kmeans(
        self,
        initial_clusters: list[list[str]],
        all_embeddings: np.ndarray,
        chunk_id_to_index: dict,
    ) -> list[list[Chunk]]:
        """(Phase 2) K-Means를 사용해 초기 클러스터를 정제합니다."""
        print("[BatchBuilder] Phase 2: Refining clusters with K-Means...")
        num_clusters = len(initial_clusters)

        initial_centroids = []
        for cluster_ids in initial_clusters:
            indices = [
                chunk_id_to_index[cid]
                for cid in cluster_ids
                if cid in chunk_id_to_index
            ]
            if indices:
                initial_centroids.append(all_embeddings[indices].mean(axis=0))

        if not initial_centroids:
            return []

        kmeans = KMeans(
            n_clusters=num_clusters,
            init=np.array(initial_centroids),
            n_init=1,
            random_state=42,
        )
        kmeans.fit(all_embeddings)

        final_batches = [[] for _ in range(num_clusters)]
        for chunk_id, index in chunk_id_to_index.items():
            if chunk_obj := self.chunk_map.get(chunk_id):
                cluster_label = kmeans.labels_[index]
                final_batches[cluster_label].append(chunk_obj)

        final_batches = [batch for batch in final_batches if len(batch) > 1]
        print(
            f"[BatchBuilder] Phase 2 Done: Refined to {len(final_batches)} final batches."
        )
        return final_batches

    def _format_output_batches(
        self,
        final_batches: list[list[Chunk]],
        all_embeddings: np.ndarray,
        chunk_id_to_index: dict,
    ) -> list[dict]:
        """최종 배치 목록에 유사도 점수와 중심점을 추가하여 포맷팅합니다."""
        print("[BatchBuilder] Formatting final batches with scores and centroids...")
        output_batches = []
        for i, batch_chunks in enumerate(final_batches):
            seed_chunk = batch_chunks[0]

            batch_indices = [chunk_id_to_index[c.chunk_id] for c in batch_chunks]
            batch_embeddings = all_embeddings[batch_indices]
            centroid = batch_embeddings.mean(axis=0)

            batch_info = {
                "batch_id": f"{seed_chunk.job_id}_batch_{i}",
                "status": "ready",  # (신규) 초기 상태를 'ready'로 설정
                "seed_chunk_id": seed_chunk.chunk_id,
                "centroid": centroid.tolist(),
                "chunks": [],
            }

            seed_vector = all_embeddings[chunk_id_to_index[seed_chunk.chunk_id]]
            for chunk in batch_chunks:
                chunk_vector = all_embeddings[chunk_id_to_index[chunk.chunk_id]]
                distance = np.linalg.norm(seed_vector - chunk_vector)
                similarity = max(0, 1 - distance)

                batch_info["chunks"].append(
                    {
                        "similarity_score": round(similarity, 4),
                        "chunk": dataclasses.asdict(chunk),
                    }
                )

            batch_info["chunks"].sort(key=lambda x: x["similarity_score"], reverse=True)
            output_batches.append(batch_info)

        return output_batches

    def _find_closest_centroid(
        self, vector: np.ndarray, centroids: dict
    ) -> tuple[str | None, float]:
        """주어진 벡터와 가장 가까운 중심점을 찾습니다."""
        closest_batch_id, min_distance = None, float("inf")
        for batch_id, centroid_vector in centroids.items():
            distance = np.linalg.norm(vector - centroid_vector)
            if distance < min_distance:
                min_distance = distance
                closest_batch_id = batch_id
        return closest_batch_id, min_distance
