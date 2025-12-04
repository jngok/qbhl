"""
SFDPM (Structure Feature Detection & Preprocessing Module) 예시 템플릿
- 나중에 실제 QBHL용 전처리 로직을 여기에 채워 넣으면 됨
"""

from typing import List, Dict, Any


def run_sfdpm(documents: List[str], query: str = "") -> Dict[str, Any]:
    """
    SFDPM 메인 함수 (예시)

    Parameters
    ----------
    documents : List[str]
        레이블링/전처리 대상 문장 리스트
    query : str
        질의(예: '예방 단계', '대응 단계' 등). 아직은 로그 용도로만 사용.

    Returns
    -------
    Dict[str, Any]
        간단한 요약 정보(문장 수, 첫 문장, 질의 내용 등)를 담은 딕셔너리
    """
    n_docs = len(documents)

    return {
        "status": "ok",
        "n_docs": n_docs,
        "query": query,
        "first_doc": documents[0] if n_docs > 0 else None,
    }


def simple_summary(documents: List[str]) -> Dict[str, Any]:
    """
    문장 리스트에 대한 아주 간단한 통계 요약 함수 (예시)

    Parameters
    ----------
    documents : List[str]
        대상 문장 리스트

    Returns
    -------
    Dict[str, Any]
        문장 수, 전체 길이, 평균 길이 등을 담은 요약 정보
    """
    lengths = [len(d) for d in documents]
    n_docs = len(lengths)
    total_len = sum(lengths) if n_docs > 0 else 0
    avg_len = total_len / n_docs if n_docs > 0 else 0

    return {
        "n_docs": n_docs,
        "total_len": total_len,
        "avg_len": avg_len,
    }
