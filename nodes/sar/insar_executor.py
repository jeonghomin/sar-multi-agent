"""InSAR 실행 로직 - FastAPI 호출"""
import requests
import threading
from pathlib import Path
from langchain_core.messages import AIMessage


INSAR_API_URL = "http://localhost:8002"


def execute_insar_processing(
    safe_files: list, 
    location_name: str = None, 
    coordinates: dict = None,
    subswath: str = "IW3",
    polarization: str = "VV",
    first_burst: int = 1,
    last_burst: int = 4
):
    """
    2개의 SAFE 파일로 InSAR 처리 실행 (FastAPI 호출)
    
    Args:
        safe_files: 2개의 SAFE 파일 경로 리스트 (Path 객체)
        location_name: 지역명 (선택)
        coordinates: 좌표 정보 (선택)
        subswath: IW1/IW2/IW3 (기본값: IW3)
        polarization: VV/VH/HH/HV (기본값: VV)
        first_burst: 시작 burst (기본값: 1)
        last_burst: 끝 burst (기본값: 4)
    
    Returns:
        dict: sar_result와 messages를 포함한 결과
    """
    print(f"[Execute InSAR] InSAR API 호출 시작")
    
    # 파일 경로 확인
    master_file = str(safe_files[0].absolute())
    slave_file = str(safe_files[1].absolute())
    
    if not Path(master_file).exists():
        error_msg = f"❌ Master 파일을 찾을 수 없습니다: {master_file}"
        print(error_msg)
        return {
            "generation": error_msg,
            "sar_result": {"task": "insar", "status": "error", "message": error_msg},
            "messages": [AIMessage(content=error_msg)]
        }
    
    if not Path(slave_file).exists():
        error_msg = f"❌ Slave 파일을 찾을 수 없습니다: {slave_file}"
        print(error_msg)
        return {
            "generation": error_msg,
            "sar_result": {"task": "insar", "status": "error", "message": error_msg},
            "messages": [AIMessage(content=error_msg)]
        }
    
    # 작업 디렉토리 생성 (환경변수 경로 사용)
    try:
        from config import DEFAULT_SAR_PATH
        base_path = DEFAULT_SAR_PATH / "insar_output"
    except ImportError:
        # fallback: 첫 번째 파일의 부모 디렉토리 사용
        base_path = safe_files[0].parent / "insar_output"
    
    workdir = base_path
    workdir.mkdir(parents=True, exist_ok=True)
    
    print(f"🛰️ InSAR API 호출 준비")
    print(f"  - Master: {Path(master_file).name}")
    print(f"  - Slave: {Path(slave_file).name}")
    print(f"  - Subswath: {subswath}, Polarization: {polarization}")
    print(f"  - Burst: {first_burst}-{last_burst}")
    print(f"  - 작업 디렉토리: {workdir}")
    
    # 백그라운드 실행 함수
    def run_insar_background(payload):
        """InSAR API를 백그라운드에서 호출"""
        try:
            print(f"[Background] InSAR 처리 시작...")
            response = requests.post(
                f"{INSAR_API_URL}/insar",
                json=payload,
                timeout=7200  # 2시간
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"[Background] ✅ InSAR 완료: {result}")
            else:
                print(f"[Background] ❌ InSAR 실패 (HTTP {response.status_code}): {response.text}")
        except Exception as e:
            print(f"[Background] ❌ InSAR 오류: {e}")
    
    try:
        # InSAR API 호출 준비
        payload = {
            "master_file": master_file,
            "slave_file": slave_file,
            "subswath": subswath,
            "polarization": polarization,
            "first_burst": first_burst,
            "last_burst": last_burst,
            "workdir": str(workdir)
        }
        
        print(f"📡 POST {INSAR_API_URL}/insar (백그라운드)")
        
        # 백그라운드 스레드로 실행
        thread = threading.Thread(target=run_insar_background, args=(payload,), daemon=True)
        thread.start()
        
        location_str = location_name or (coordinates.get("location", "N/A") if coordinates else "N/A")
        
        # 즉시 시작 메시지 반환
        start_msg = f"""🚀 InSAR 처리를 시작했습니다!

📁 **작업 디렉토리**: `{workdir}`

🛰️ **입력 파일**:
- Master: {safe_files[0].name}
- Slave: {safe_files[1].name}

⚙️ **처리 파라미터**:
- Subswath: {subswath}
- Polarization: {polarization}
- Burst: {first_burst}-{last_burst}

⏱️ **예상 소요 시간**: 약 20-30분

📊 **처리 단계**:
1. TOPSAR Split (관심 영역 추출)
2. Apply Orbit File (궤도 정보 적용)
3. Back-Geocoding (영상 정합)
4. Enhanced Spectral Diversity (ESD 보정)
5. Interferogram 생성 (간섭무늬 계산)
6. TOPSAR Deburst (버스트 병합)
7. Topographic Phase Removal (지형 위상 제거)
8. Multilooking (해상도 조정)
9. Goldstein Phase Filtering (위상 필터링)
10. Terrain Correction (지형 보정)

🔔 **알림**: 백그라운드에서 처리가 진행됩니다. 
완료되면 결과 파일이 `{workdir}` 디렉토리에 저장됩니다.

💡 **다른 작업을 계속 진행하셔도 됩니다!**
"""
        print(start_msg)
        
        return {
            "generation": start_msg,
            "sar_result": {
                "task": "insar",
                "status": "processing",
                "location": location_str,
                "event": "SAR Interferometry",
                "file_path": str(workdir),
                "message": start_msg
            },
            "messages": [AIMessage(content=start_msg)]
        }
        
    except requests.exceptions.ConnectionError:
        error_msg = f"❌ InSAR API 서버에 연결할 수 없습니다 ({INSAR_API_URL}). 서버가 실행 중인지 확인하세요."
        print(error_msg)
        return {
            "generation": error_msg,
            "sar_result": {"task": "insar", "status": "error", "message": error_msg},
            "messages": [AIMessage(content=error_msg)]
        }
    except requests.exceptions.Timeout:
        error_msg = f"❌ InSAR 처리 시간 초과 (2시간). 데이터가 너무 크거나 서버에 문제가 있을 수 있습니다."
        print(error_msg)
        return {
            "generation": error_msg,
            "sar_result": {"task": "insar", "status": "error", "message": error_msg},
            "messages": [AIMessage(content=error_msg)]
        }
    except Exception as e:
        error_msg = f"❌ InSAR 처리 중 오류 발생: {str(e)}"
        print(error_msg)
        return {
            "generation": error_msg,
            "sar_result": {"task": "insar", "status": "error", "message": error_msg},
            "messages": [AIMessage(content=error_msg)]
        }
