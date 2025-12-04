from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import uuid
import logging
import os

# ==========================
# 로깅 설정
# ==========================
logger = logging.getLogger("SFDPM")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[SFDPM] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ==========================
# 상수 정의
# ==========================
TEXT_FIELDS_PRIORITY: List[str] = ["조문내용", "content", "text", "내용"]
MIN_CONTENT_LENGTH: int = 20  # 20자 미만은 노드 생성 대상에서 제외


# ==========================
# 데이터 클래스 정의
# ==========================

@dataclass
class HierarchicalNode:
    """
    계층 구조 보존을 위한 기본 노드 단위
    - id: 전역 유일 식별자(UUID)
    - content: 계층 문맥이 포함된 최종 텍스트
    - data: 원본 JSON 조각 그대로 저장
    - level: 트리 내 계층 깊이 (0,1,2,...)
    - parent: 부모 노드
    - children: 자식 노드 리스트
    - embedding: 임베딩 벡터 (지연 평가)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    level: int = 0
    parent: Optional["HierarchicalNode"] = None
    children: List["HierarchicalNode"] = field(default_factory=list)
    embedding: Optional[Any] = None

    def add_child(self, child: "HierarchicalNode") -> None:
        child.parent = self
        self.children.append(child)

    def path_from_root(self) -> List["HierarchicalNode"]:
        """
        루트부터 현재 노드까지의 경로 반환
        """
        path: List[HierarchicalNode] = []
        node: Optional[HierarchicalNode] = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def text_path(self, sep: str = " > ") -> str:
        """
        루트부터 현재 노드까지의 content를 연결한 문자열
        """
        return sep.join([n.content for n in self.path_from_root() if n.content])

    def __repr__(self) -> str:
        return f"HierarchicalNode(id={self.id[:8]}, level={self.level}, content={self.content[:20]!r}...)"


@dataclass
class SFDPMResult:
    """
    SFDPM 파싱 결과 컨테이너
    - roots: 루트 노드들
    - all_nodes: 트리 전체 노드를 평탄화한 리스트
    - node_pool: 텍스트 기반 노드 캐시 (노드 풀링용)
    """
    roots: List[HierarchicalNode]
    all_nodes: List[HierarchicalNode]
    node_pool: Dict[str, HierarchicalNode]


# ==========================
# 내부 유틸 함수
# ==========================

def _read_json_utf8(path: str) -> Any:
    """
    JSON 파일을 UTF-8로 읽어서 파싱.
    인코딩 오류 및 JSON 파싱 오류는 로깅 후 예외 발생.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError as e:
        logger.error(f"UTF-8 인코딩 오류: {path} - {e}")
        raise

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {path} - {e}")
        raise


def _extract_text_from_dict(d: Dict[str, Any]) -> Optional[str]:
    """
    사전에서 우선순위에 따라 텍스트 필드를 추출.
    - '조문내용' > 'content' > 'text' > '내용'
    - 20자 미만 텍스트는 의미 없는 단편으로 보고 None 반환.
    """
    for key in TEXT_FIELDS_PRIORITY:
        if key in d and isinstance(d[key], str):
            candidate = d[key].strip()
            if len(candidate) >= MIN_CONTENT_LENGTH:
                return candidate
    return None


def _create_or_reuse_node(
    text: str,
    data: Dict[str, Any],
    level: int,
    parent: Optional[HierarchicalNode],
    node_pool: Dict[str, HierarchicalNode],
    context_prefix: str,
    enable_pooling: bool = True,
) -> HierarchicalNode:
    """
    노드 생성 + (옵션) 노드 풀링 + 계층 문맥 전파
    - context_prefix: 상위 장/절/조 제목을 접두사로 붙여 구조적 의미를 포함
    """
    # 계층 문맥 전파
    if context_prefix:
        full_text = f"{context_prefix} {text}".strip()
    else:
        full_text = text.strip()

    key = full_text

    # 노드 풀링: 동일 텍스트가 이미 있으면 기존 노드 재사용
    if enable_pooling and key in node_pool:
        node = node_pool[key]
        if parent is not None and node not in parent.children:
            parent.add_child(node)
        return node

    # 신규 노드 생성
    node = HierarchicalNode(
        content=full_text,
        data=data,
        level=level,
        parent=parent,
    )
    if parent is not None:
        parent.add_child(node)

    if enable_pooling:
        node_pool[key] = node

    return node


def _recursive_parse(
    obj: Any,
    level: int,
    parent: Optional[HierarchicalNode],
    path: str,
    context_prefix: str,
    roots: List[HierarchicalNode],
    all_nodes: List[HierarchicalNode],
    node_pool: Dict[str, HierarchicalNode],
    enable_pooling: bool = True,
) -> None:
    """
    재귀적 하강 파싱(recursive descent parsing)
    - dict: 텍스트 필드 있으면 노드 생성, 자식은 level+1
    - list: 같은 level에서 형제 노드로 처리 (level 유지)
    - 기타 타입: 무시
    - 오류 허용적: 예외 발생 시 경고만 출력하고 계속 진행
    """
    try:
        # dict 처리
        if isinstance(obj, dict):
            text = _extract_text_from_dict(obj)
            node: Optional[HierarchicalNode] = None

            if text:
                # 이 dict에서 의미 있는 텍스트가 발견되면 새로운 노드 생성
                node = _create_or_reuse_node(
                    text=text,
                    data=obj,
                    level=level,
                    parent=parent,
                    node_pool=node_pool,
                    context_prefix=context_prefix,
                    enable_pooling=enable_pooling,
                )
                all_nodes.append(node)

                # 루트 노드 등록
                if parent is None and node not in roots:
                    roots.append(node)

                # 자식에게 전달할 새로운 문맥
                new_context_prefix = node.content
            else:
                # 텍스트가 없으면 문맥 유지
                new_context_prefix = context_prefix

            # 하위 요소 재귀 처리
            for k, v in obj.items():
                child_path = f"{path}.{k}" if path else k
                # 텍스트 노드를 생성한 경우에만 level+1
                next_level = level + 1 if text else level
                _recursive_parse(
                    obj=v,
                    level=next_level,
                    parent=node or parent,
                    path=child_path,
                    context_prefix=new_context_prefix,
                    roots=roots,
                    all_nodes=all_nodes,
                    node_pool=node_pool,
                    enable_pooling=enable_pooling,
                )

        # list 처리: 같은 level의 형제 노드
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                child_path = f"{path}[{idx}]"
                _recursive_parse(
                    obj=item,
                    level=level,
                    parent=parent,
                    path=child_path,
                    context_prefix=context_prefix,
                    roots=roots,
                    all_nodes=all_nodes,
                    node_pool=node_pool,
                    enable_pooling=enable_pooling,
                )

        # 기타 타입(str/int/float 등)은 이미 텍스트 필드에서 처리되므로 무시
        else:
            return

    except Exception as e:
        # 오류 허용적: 에러 로그만 남기고 계속 진행
        logger.warning(f"파싱 중 오류: path={path}, error={e}")


# ==========================
# 외부 공개 API
# ==========================

def parse_json_to_tree(
    json_data: Any,
    enable_pooling: bool = True,
) -> SFDPMResult:
    """
    이미 파싱된 JSON 객체를 입력으로 받아
    구조 보존 계층 트리(HierarchicalNode)로 변환.

    Parameters
    ----------
    json_data : Any
        json.loads(...) 결과 객체
    enable_pooling : bool
        동일 텍스트 노드를 풀링할지 여부

    Returns
    -------
    SFDPMResult
    """
    roots: List[HierarchicalNode] = []
    all_nodes: List[HierarchicalNode] = []
    node_pool: Dict[str, HierarchicalNode] = {}

    _recursive_parse(
        obj=json_data,
        level=0,
        parent=None,
        path="",
        context_prefix="",
        roots=roots,
        all_nodes=all_nodes,
        node_pool=node_pool,
        enable_pooling=enable_pooling,
    )

    logger.info(f"파싱 완료: 루트 노드 {len(roots)}개, 전체 노드 {len(all_nodes)}개")
    return SFDPMResult(roots=roots, all_nodes=all_nodes, node_pool=node_pool)


def parse_json_file_to_tree(
    path: str,
    enable_pooling: bool = True,
) -> SFDPMResult:
    """
    단일 JSON 파일을 읽어 SFDPM 계층 트리로 변환.

    1) UTF-8 인코딩 검증 및 JSON 파싱
    2) 재귀적 구조 파싱
    3) HierarchicalNode 트리 구성
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    logger.info(f"JSON 파일 파싱 시작: {path}")
    data = _read_json_utf8(path)
    return parse_json_to_tree(data, enable_pooling=enable_pooling)


def parse_multiple_files(
    paths: List[str],
    enable_pooling: bool = True,
    progress_interval: int = 100,
) -> List[SFDPMResult]:
    """
    여러 JSON 파일을 순차적으로 처리.
    progress_interval 단위로 진행 상황 로그 출력.

    Returns
    -------
    List[SFDPMResult]
    """
    results: List[SFDPMResult] = []
    for idx, p in enumerate(paths, start=1):
        try:
            result = parse_json_file_to_tree(p, enable_pooling=enable_pooling)
            results.append(result)
        except Exception as e:
            logger.warning(f"파일 처리 중 오류: {p} - {e}")
            continue

        if idx % progress_interval == 0:
            logger.info(f"{idx}개 파일 처리 완료")

    logger.info(f"총 {len(results)}개 파일 파싱 완료")
    return results
