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
    
    # 사용 가능한 파일 정보 수집
    available_files = []
    
    # 1. downloaded_sar_files (방금 다운로드한 파일)
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
    
    # 2. sar_search_results (검색 결과 또는 폴더 스캔 결과)
    elif sar_search_results and sar_search_results.get('products'):
        products = sar_search_results['products']
        print(f"✅ 검색 결과: {len(products)}개")
        for p in products[:2]:  # 처음 2개만
            available_files.append({
                'index': p.get('display_index', p.get('index', 0)),
                'filename': p.get('filename', ''),
                'date': p.get('date', 'unknown'),
                'path': p.get('file_path', '')
            })
    
    if len(available_files) < 2:
        # 파일이 부족한 경우 에러
        error_msg = f"""❌ InSAR 처리를 위한 파일이 부족합니다.
        
현재 파일 개수: {len(available_files)}개 (필요: 2개)

InSAR 처리를 위해서는 2개의 SAR 이미지가 필요합니다."""
        return {
            "generation": error_msg,
            "sar_result": {
                "task": "insar",
                "status": "error",
                "message": error_msg
            },
            "messages": [AIMessage(content=error_msg)]
        }
    
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
        
        # Master/Slave 모두 명시된 경우
        if result.has_master and result.has_slave:
            print("✅ Master/Slave 모두 명시됨 → InSAR 실행 진행")
            
            # Master/Slave 인덱스 추출
            master_idx = None
            slave_idx = None
            
            # 인덱스는 0 또는 1만 유효 (단일 숫자)
            master_clean = result.master_identifier.strip()
            slave_clean = result.slave_identifier.strip()
            
            # "0" 또는 "1"이면 직접 변환
            if master_clean in ["0", "1"]:
                master_idx = int(master_clean)
            else:
                # 숫자 추출 시도 (첫 번째 1자리 숫자만)
                master_match = re.search(r'\b([01])\b', master_clean)
                if master_match:
                    master_idx = int(master_match.group(1))
            
            if slave_clean in ["0", "1"]:
                slave_idx = int(slave_clean)
            else:
                slave_match = re.search(r'\b([01])\b', slave_clean)
                if slave_match:
                    slave_idx = int(slave_match.group(1))
            
            # 인덱스 확인
            if master_idx is not None and slave_idx is not None:
                # 인덱스 범위 확인
                if master_idx >= len(available_files) or slave_idx >= len(available_files):
                    error_msg = f"""❌ 인덱스 범위 초과

선택한 인덱스가 유효하지 않습니다:
- Master: {master_idx} (최대: {len(available_files)-1})
- Slave: {slave_idx} (최대: {len(available_files)-1})"""
                    return {
                        "generation": error_msg,
                        "sar_result": {
                            "task": "insar",
                            "status": "error",
                            "message": error_msg
                        },
                        "messages": [AIMessage(content=error_msg)]
                    }
                
                # Master/Slave 파일 설정
                master_file = available_files[master_idx]
                slave_file = available_files[slave_idx]
                
                print(f"✅ Master: [{master_idx}] {master_file['filename']}")
                print(f"✅ Slave: [{slave_idx}] {slave_file['filename']}")
                
                # InSAR 파라미터 확인 (IW, polarization, burst)
                insar_params = state.get("insar_parameters")
                
                if not insar_params:
                    # 파라미터가 없으면 사용자에게 물어보기
                    print("⚠️ InSAR 파라미터 없음 → 사용자 입력 요청")
                    param_msg = f"""✅ Master와 Slave를 선택했습니다!

🛰️ **선택된 파일**:
- Master: {master_file['filename']}
- Slave: {slave_file['filename']}

⚙️ **InSAR 처리 파라미터를 설정해주세요:**

**1. Subswath (IW)**
- IW1, IW2, IW3 중 선택
- 💡 추천: **IW3** (가장 넓은 범위)

**2. Polarization (편파)**
- VV, VH, HH, HV 중 선택
- 💡 추천: **VV** (일반적으로 사용)

**3. Burst (버스트 범위)**
- 첫 번째 burst와 마지막 burst 번호
- 💡 추천: **1-4** (표준 범위)

**입력 예시:**
- "IW3, VV, burst 1-4로 해줘"
- "기본값으로 해줘" (IW3, VV, 1-4)
- "IW2 사용해줘" (polarization과 burst는 기본값)

💡 **잘 모르시겠다면 "기본값"이라고 입력하세요!**
"""
                    return {
                        "generation": param_msg,
                        "sar_result": {
                            "task": "insar",
                            "status": "awaiting_parameters",
                            "message": "InSAR 파라미터 입력 대기"
                        },
                        "messages": [AIMessage(content=param_msg)],
                        "downloaded_sar_files": [master_file['path'], slave_file['path']],
                        "awaiting_insar_parameters": True,  # 파라미터 입력 대기
                        "awaiting_master_slave_selection": False,
                        "insar_master_slave_ready": False,  # 아직 준비 안 됨
                    }
                
                # 파라미터가 있으면 바로 실행 준비
                # downloaded_sar_files 업데이트 (순서 중요: Master → Slave)
                return {
                    "downloaded_sar_files": [master_file['path'], slave_file['path']],
                    "insar_master_slave_ready": True,  # 실행 준비 완료 플래그
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
