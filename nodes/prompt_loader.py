"""프롬프트 로더 - 프롬프트 파일을 읽어오는 유틸리티"""
from pathlib import Path
from typing import Dict

_prompt_cache: Dict[str, str] = {}


def load_prompt(prompt_path: str, **kwargs) -> str:
    """
    프롬프트 파일을 로드하고 변수를 치환
    
    Args:
        prompt_path: 프롬프트 파일 경로 (nodes/ 기준 상대 경로)
        **kwargs: 프롬프트 내 {변수}를 치환할 값들
    
    Returns:
        치환된 프롬프트 문자열
    
    Example:
        prompt = load_prompt(
            "sar/prompts/master_slave_check.txt",
            question="InSAR 해줘",
            files_info="[0] file1.zip\n[1] file2.zip"
        )
    """
    # 캐시 확인
    if prompt_path not in _prompt_cache:
        # 프롬프트 파일 읽기
        nodes_dir = Path(__file__).parent
        full_path = nodes_dir / prompt_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {full_path}")
        
        _prompt_cache[prompt_path] = full_path.read_text(encoding='utf-8')
    
    # 변수 치환
    prompt_template = _prompt_cache[prompt_path]
    
    if kwargs:
        return prompt_template.format(**kwargs)
    return prompt_template


def clear_cache():
    """프롬프트 캐시 초기화 (개발/테스트용)"""
    _prompt_cache.clear()
