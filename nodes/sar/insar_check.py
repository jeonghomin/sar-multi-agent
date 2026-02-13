"""InSAR Master/Slave 체크 노드 - LLM 기반"""
import re
from datetime import datetime
from langchain_core.messages import AIMessage
from core.llm_config import llm
from pydantic import BaseModel, Field
from ..prompt_loader import load_prompt


class MasterSlaveCheck(BaseModel):
    """Master/Slave 정보 추출 결과"""
    has_master: bool = Field(description="Master 정보가 명시되었는지")
    has_slave: bool = Field(description="Slave 정보가 명시되었는지")
    master_identifier: str = Field(description="Master 식별자 (파일명, 인덱스, 날짜 등)")
    slave_identifier: str = Field(description="Slave 식별자 (파일명, 인덱스, 날짜 등)")
    reasoning: str = Field(description="판단 근거")


# ===== 헬퍼 함수들 =====

def _parse_insar_parameters(question):
    """
    질문에서 InSAR 파라미터 파싱
    
    Returns:
        dict or None: 파싱된 파라미터 또는 None
    """
    question_lower = question.lower()
    
    # 기본값 체크
    if "기본" in question_lower or "default" in question_lower:
        return {
            "subswath": "IW3",
            "polarization": "VV",
            "first_burst": 1,
            "last_burst": 4
        }
    
    # 파라미터 파싱
    params = {}
    
    # IW 추출
    iw_match = re.search(r'(IW[123])', question, re.IGNORECASE)
    params["subswath"] = iw_match.group(1).upper() if iw_match else None
    
    # Polarization 추출
    pol_match = re.search(r'\b(VV|VH|HH|HV)\b', question, re.IGNORECASE)
    params["polarization"] = pol_match.group(1).upper() if pol_match else None
    
    # Burst 추출
    burst_match = re.search(r'burst\s*(\d+)\s*[-~]\s*(\d+)', question, re.IGNORECASE)
    if burst_match:
        params["first_burst"] = int(burst_match.group(1))
        params["last_burst"] = int(burst_match.group(2))
    else:
        # 단일 숫자 2개 찾기
        nums = re.findall(r'\b(\d+)\b', question)
        if len(nums) >= 2:
            params["first_burst"] = int(nums[0])
            params["last_burst"] = int(nums[1])
        else:
            params["first_burst"] = None
            params["last_burst"] = None
    
    # 완전히 파싱되었는지 확인
    if all([params.get("subswath"), 
           params.get("polarization"),
           params.get("first_burst") is not None,
           params.get("last_burst") is not None]):
        return params
    
    return None


def _build_ready_response(params=None):
    """InSAR 실행 준비 완료 응답"""
    response = {
        "insar_master_slave_ready": True,
        "sar_result": {
            "task": "insar",
            "status": "ready_for_execution",
            "message": "Master/Slave 및 파라미터 준비 완료"
        }
    }
    
    if params:
        response["insar_parameters"] = params
        response["awaiting_insar_parameters"] = False
    
    return response


def _build_param_request_response(master_path, slave_path, detailed=False):
    """파라미터 입력 요청 응답"""
    master_filename = master_path.split('/')[-1]
    slave_filename = slave_path.split('/')[-1]
    
    template_file = "sar/prompts/insar_param_request_detailed.txt" if detailed else "sar/prompts/insar_param_request.txt"
    message = load_prompt(
        template_file,
        master_filename=master_filename,
        slave_filename=slave_filename
    )
    
    return {
        "generation": message,
        "downloaded_sar_files": [master_path, slave_path],
        "insar_master_file": master_path,
        "insar_slave_file": slave_path,
        "awaiting_insar_parameters": True,
        "awaiting_master_slave_selection": False,
        "insar_master_slave_ready": False,
        "sar_result": {
            "task": "insar",
            "status": "awaiting_parameters",
            "message": "Master/Slave 선택 완료, 파라미터 입력 대기"
        },
        "messages": [AIMessage(content=message)]
    }


def _build_error_response(message):
    """에러 응답"""
    return {
        "generation": message,
        "sar_result": {
            "task": "insar",
            "status": "error",
            "message": message
        },
        "messages": [AIMessage(content=message)]
    }


def _collect_available_files(downloaded_sar_files, sar_search_results):
    """사용 가능한 SAR 파일 정보 수집"""
    available_files = []
    
    # 1. downloaded_sar_files
    if downloaded_sar_files and len(downloaded_sar_files) >= 2:
        print(f"✅ 다운로드한 파일: {len(downloaded_sar_files)}개")
        for i, f in enumerate(downloaded_sar_files[:2]):
            filename = f.split('/')[-1] if '/' in f else f
            date_match = re.search(r'(\d{8})', filename)
            date_str = date_match.group(1) if date_match else "unknown"
            available_files.append({
                'index': i,
                'filename': filename,
                'date': date_str,
                'path': f
            })
    
    # 2. sar_search_results
    elif sar_search_results and sar_search_results.get('products'):
        products = sar_search_results['products']
        print(f"✅ 검색 결과: {len(products)}개")
        for p in products[:2]:
            available_files.append({
                'index': p.get('display_index', p.get('index', 0)),
                'filename': p.get('filename', ''),
                'date': p.get('date', 'unknown'),
                'path': p.get('file_path', '')
            })
    
    return available_files


def _extract_file_index(identifier_str):
    """식별자 문자열에서 파일 인덱스 추출 (0 or 1)"""
    clean = identifier_str.strip()
    
    # "0" 또는 "1"이면 직접 변환
    if clean in ["0", "1"]:
        return int(clean)
    
    # 숫자 추출 시도
    match = re.search(r'\b([01])\b', clean)
    if match:
        return int(match.group(1))
    
    return None


def check_insar_master_slave(state):
    """
    InSAR 처리 전 Master/Slave 정보 체크 (LLM 기반)
    
    - Master/Slave가 명시되어 있으면 → run_insar_execute로 라우팅
    - 명시되어 있지 않으면 → 사용자에게 선택 요청 메시지 출력 → END
    """
    print("==== [INSAR CHECK - Master/Slave 확인] ====")
    
    question = state.get("question", "")
    downloaded_sar_files = state.get("downloaded_sar_files") or []
    sar_search_results = state.get("sar_search_results", {})
    
    # ⭐ 이미 state에 Master/Slave가 저장되어 있는지 확인
    existing_master = state.get("insar_master_file")
    existing_slave = state.get("insar_slave_file")
    
    # ⭐ 1. State에 이미 Master/Slave가 있는 경우
    if existing_master and existing_slave:
        print(f"✅ State에 저장된 Master/Slave 발견:")
        print(f"   Master: {existing_master}")
        print(f"   Slave: {existing_slave}")
        
        # 파라미터 확인
        insar_params = state.get("insar_parameters")
        if insar_params:
            print(f"✅ 파라미터도 준비됨: {insar_params}")
            return _build_ready_response()
        
        # 파라미터가 없으면 question에서 파싱 시도
        print("⚠️ 파라미터 없음 → question에서 파싱 시도")
        parsed_params = _parse_insar_parameters(question)
        
        if parsed_params:
            print(f"✅ 파라미터 파싱 완료: {parsed_params}")
            return _build_ready_response(params=parsed_params)
        
        # 파싱 실패 → 사용자 입력 요청
        print("⚠️ 파라미터 파싱 실패 → 사용자 입력 요청")
        return _build_param_request_response(existing_master, existing_slave)
    
    # ⭐ 2. 사용 가능한 파일 정보 수집
    available_files = _collect_available_files(downloaded_sar_files, sar_search_results)
    
    if len(available_files) < 2:
        error_msg = f"""❌ InSAR 처리를 위한 파일이 부족합니다.

현재 파일 개수: {len(available_files)}개 (필요: 2개)

InSAR 처리를 위해서는 2개의 SAR 이미지가 필요합니다."""
        return _build_error_response(error_msg)
    
    # 파일 정보 문자열 생성
    files_info = "\n".join([
        f"[{f['index']}] {f['date']} - {f['filename']}"
        for f in available_files
    ])
    
    # LLM에게 Master/Slave 정보 체크 요청
    prompt = load_prompt(
        "sar/prompts/master_slave_check.txt",
        question=question,
        files_info=files_info
    )
    
    try:
        # LLM 호출 (Structured Output)
        result = llm.with_structured_output(MasterSlaveCheck).invoke(prompt)
        
        print(f"[LLM 판단]")
        print(f"  has_master: {result.has_master}")
        print(f"  has_slave: {result.has_slave}")
        print(f"  master_identifier: {result.master_identifier}")
        print(f"  slave_identifier: {result.slave_identifier}")
        print(f"  reasoning: {result.reasoning}")
        
        # ⭐ 3. Master/Slave 모두 명시된 경우
        if result.has_master and result.has_slave:
            print("✅ Master/Slave 모두 명시됨 → InSAR 실행 진행")
            
            # 인덱스 추출
            master_idx = _extract_file_index(result.master_identifier)
            slave_idx = _extract_file_index(result.slave_identifier)
            
            # 인덱스 확인
            if master_idx is not None and slave_idx is not None:
                # 인덱스 범위 확인
                if master_idx >= len(available_files) or slave_idx >= len(available_files):
                    error_msg = f"""❌ 인덱스 범위 초과

선택한 인덱스가 유효하지 않습니다:
- Master: {master_idx} (최대: {len(available_files)-1})
- Slave: {slave_idx} (최대: {len(available_files)-1})"""
                    return _build_error_response(error_msg)
                
                # Master/Slave 파일 설정
                master_file = available_files[master_idx]
                slave_file = available_files[slave_idx]
                
                print(f"✅ Master: [{master_idx}] {master_file['filename']}")
                print(f"✅ Slave: [{slave_idx}] {slave_file['filename']}")
                
                # ⭐ State에 Master/Slave 파일 저장
                master_path = master_file['path']
                slave_path = slave_file['path']
                
                # InSAR 파라미터 확인
                insar_params = state.get("insar_parameters")
                
                if not insar_params:
                    print("⚠️ InSAR 파라미터 없음 → 사용자 입력 요청")
                    return _build_param_request_response(master_path, slave_path, detailed=True)
                
                # 파라미터 있음 → 실행 준비 완료
                return {
                    "downloaded_sar_files": [master_path, slave_path],
                    "insar_master_file": master_path,
                    "insar_slave_file": slave_path,
                    "insar_master_slave_ready": True,
                    "awaiting_master_slave_selection": False,  # 선택 대기 해제
                }
            else:
                # 인덱스 추출 실패 → 사용자에게 다시 요청
                print("⚠️ 인덱스 추출 실패 → 사용자에게 선택 요청")
                result.has_master = False
                result.has_slave = False
        
        # Master/Slave 중 하나라도 명시 안 된 경우
        if not result.has_master or not result.has_slave:
            print("⚠️ Master/Slave 명시 안 됨 → 사용자 선택 요청")
            
            # 선택 요청 메시지 생성
            selection_msg = f"""✅ InSAR 처리를 위한 **2개의 SAR 데이터**를 찾았습니다!

📂 **SAFE 파일 목록:**

"""
            for f in available_files:
                # 날짜 포맷팅
                date_str = f['date']
                if date_str != "unknown" and len(date_str) == 8:
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                else:
                    formatted_date = date_str
                
                selection_msg += f"[{f['index']}] {formatted_date} - {f['filename']}\n"
            
            selection_msg += """
🎯 **Master와 Slave를 선택해주세요:**

**다음 형식으로 입력:**
- "Master 0, Slave 1"
- "0번이 master, 1번이 slave"

💡 **선택 팁 (InSAR 지표변형 분석):**
- **Master**: 이벤트 **이전** 날짜 (기준 이미지, 변화 전)
- **Slave**: 이벤트 **이후** 날짜 (비교 이미지, 변화 후)
- 일반적으로 **시간 순서대로 Master → Slave** 순입니다.

❓ **왜 필요한가요?**
InSAR는 두 시점의 SAR 이미지를 비교하여 지표 변형을 측정합니다.
어떤 것을 기준(Master)으로, 어떤 것을 비교(Slave)로 사용할지 명확히 지정해야 합니다.
"""
            
            return {
                "generation": selection_msg,
                "sar_result": {
                    "task": "insar",
                    "status": "awaiting_selection",
                    "message": "Master/Slave 선택 대기"
                },
                "messages": [AIMessage(content=selection_msg)],
                "awaiting_master_slave_selection": True,  # 선택 대기 플래그
                "insar_master_slave_ready": False,
            }
    
    except Exception as e:
        print(f"❌ LLM 호출 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # LLM 실패 시 폴백: 사용자에게 선택 요청
        selection_msg = f"""✅ InSAR 처리를 위한 **2개의 SAR 데이터**를 찾았습니다!

📂 **SAFE 파일 목록:**

"""
        for f in available_files:
            date_str = f['date']
            if date_str != "unknown" and len(date_str) == 8:
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                formatted_date = date_str
            
            selection_msg += f"[{f['index']}] {formatted_date} - {f['filename']}\n"
        
        selection_msg += """
🎯 **Master와 Slave를 선택해주세요:**

**다음 형식으로 입력:**
- "Master 0, Slave 1"
- "0번이 master, 1번이 slave"
"""
        
        return {
            "generation": selection_msg,
            "sar_result": {
                "task": "insar",
                "status": "awaiting_selection",
                "message": "Master/Slave 선택 대기"
            },
            "messages": [AIMessage(content=selection_msg)],
            "awaiting_master_slave_selection": True,
            "insar_master_slave_ready": False,
        }
