from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import re
import uuid

# SFDPM에서 만든 계층 노드 타입 가져오기
try:
    from .sfdpm import HierarchicalNode, SFDPMResult
except ImportError:  # 직접 실행 시를 대비한 fallback
    from sfdpm import HierarchicalNode, SFDPMResult  # type: ignore


# ==========================
# 로깅 설정
# ==========================
logger = logging.getLogger("QFDPM")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[QFDPM] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ==========================
# 쿼리 관련 데이터 구조
# ==========================

@dataclass
class QueryItem:
    """
    단일 질의(쿼리)를 표현하는 단위
    - id: UUID 기반 전역 유일 식별자
    - text: 원본 질의 문장
    - tokens: 전처리된 토큰 리스트
    - metadata: 추가 메타데이터(채널, 출처 등 확장용)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    tokens: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"QueryItem(id={self.id[:8]}, tokens={self.tokens})"


@dataclass
class QueryMatch:
    """
    질의-노드 매칭 결과 구조
    - query: QueryItem
    - node: HierarchicalNode (SFDPM에서 만들어진 계층 노드)
    - score: 질의와 노드 간 유사도 점수 (0.0~1.0)
    """
    query: QueryItem
    node: HierarchicalNode
    score: float

    def __repr__(self) -> str:
        return (
            f"QueryMatch(q={self.query.id[:8]}, "
            f"node={self.node.id[:8]}, "
            f"score={self.score:.3f})"
        )


@dataclass
class QFDPMResult:
    """
    QFDPM 처리 결과 컨테이너
    - queries: 전처리된 QueryItem 목록
    - matches: QueryMatch 리스트
    """
    queries: List[QueryItem]
    matches: List[QueryMatch]


# ==========================
# 내부 유틸 함수
# ==========================

_TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")


def _simple_tokenize(text: str) -> List[str]:
    """
    간단한 토큰화 함수
    - 한글/영문/숫자 연속 문자열을 토큰으로 사용
    - 모두 소문자로 변환
    - 길이 1인 토큰은 제거 (조사/기호 수준)
    """
    tokens = [m.group(0).lower() for m in _TOKEN_PATTERN.finditer(text)]
    return [t for t in tokens if len(t) > 1]


def _token_overlap_score(
    q_tokens: List[str],
    node_tokens: List[str],
) -> float:
    """
    질의 토큰과 노드 토큰 간의 단순 중복 기반 점수
    - Jaccard 유사도와 유사한 아이디어
    - |교집합| / |합집합|
    """
    qs = set(q_tokens)
    ns = set(node_tokens)
    if not qs or not ns:
        return 0.0
    inter = qs.intersection(ns)
    union = qs.union(ns)
    return len(inter) / len(union)


# ==========================
# 외부 공개 API
# ==========================

def build_query_items(
    queries: List[str],
    metadata_list: Optional[List[Dict[str, Any]]] = None,
) -> List<QueryItem]:
    """
    질의 문자열 리스트를 QueryItem 리스트로 변환.
    - 공백/빈 문자열은 자동으로 제거.
    - metadata_list가 주어지면 각 질의와 1:1 매핑.
    """
    result: List[QueryItem] = []
    if metadata_list is None:
        metadata_list = [{} for _ in queries]

    for text, meta in zip(queries, metadata_list):
        text = (text or "").strip()
        if not text:
            continue
        qi = QueryItem(
            text=text,
            tokens=_simple_tokenize(text),
            metadata=meta or {},
        )
        result.append(qi)

    logger.info(f"총 {len(result)}개의 QueryItem 생성")
    return result


def match_queries_to_nodes(
    queries: List[QueryItem],
    nodes: List[HierarchicalNode],
    top_k: int = 5,
    min_score: float = 0.0,
) -> List[QueryMatch]:
    """
    질의-노드 간 토큰 중복 기반 매칭 수행.

    Parameters
    ----------
    queries : List[QueryItem]
        전처리된 질의 객체 리스트
    nodes : List[HierarchicalNode]
        SFDPM에서 생성한 계층 노드 리스트
    top_k : int
        질의당 상위 몇 개의 노드를 반환할지
    min_score : float
        이 값보다 작은 점수는 필터링

    Returns
    -------
    List[QueryMatch]
        모든 질의에 대한 매칭 결과 (정렬되어 있지는 않음)
    """
    matches: List[QueryMatch] = []

    # 노드 content도 토큰화해서 캐시해두면 여러 질의에 재사용 가능
    node_tokens_cache: Dict[str, List[str]] = {}
    for node in nodes:
        node_tokens_cache[node.id] = _simple_tokenize(node.content)

    for q in queries:
        scored: List[Tuple[float, HierarchicalNode]] = []
        for node in nodes:
            node_tokens = node_tokens_cache[node.id]
            score = _token_overlap_score(q.tokens, node_tokens)
            if score >= min_score:
                scored.append((score, node))

        # 점수 내림차순으로 정렬 후 상위 top_k만 사용
        scored.sort(key=lambda x: x[0], reverse=True)
        for score, node in scored[:top_k]:
            matches.append(QueryMatch(query=q, node=node, score=score))

    logger.info(
        f"총 {len(queries)}개 질의에 대해 "
        f"{len(matches)}개 질의-노드 매칭 결과 생성"
    )
    return matches


def run_qfdpm(
    sfdpm_result: SFDPMResult,
    query_texts: List[str],
    query_metadata: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
    min_score: float = 0.0,
) -> QFDPMResult:
    """
    SFDPM 결과와 질의 리스트를 입력으로 받아
    QFDPM 파이프라인 전체를 실행.

    1) 질의 전처리 및 QueryItem 생성
    2) 계층 노드(HierarchicalNode)와의 토큰 중복 기반 매칭
    3) QFDPMResult로 결과 반환
    """
    queries = build_query_items(query_texts, query_metadata)
    matches = match_queries_to_nodes(
        queries=queries,
        nodes=sfdpm_result.all_nodes,
        top_k=top_k,
        min_score=min_score,
    )
    return QFDPMResult(queries=queries, matches=matches)
