"""
SAR Download API 테스트 스크립트
"""
import requests
import json

# API 서버 주소
API_URL = "http://localhost:8001"


def test_health_check():
    """헬스 체크 테스트"""
    print("\n=== 1. Health Check ===")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


def test_download():
    """다운로드 테스트"""
    print("\n=== 2. SAR Download Test (일반) ===")
    
    payload = {
        "latitude": 36.0,
        "longitude": 140.0,
        "location_name": "이바라키 테스트",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "buffer": 0.5,
        "max_results": 5,
        "select_insar_pair": True
    }
    
    print(f"Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\n다운로드 시작... (시간이 걸릴 수 있습니다)")
    
    try:
        response = requests.post(
            f"{API_URL}/download",
            json=payload,
            timeout=600  # 10분
        )
        
        print(f"\nStatus: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
    except requests.exceptions.Timeout:
        print("⚠️ 요청 타임아웃 (10분 초과)")
    except requests.exceptions.ConnectionError:
        print("❌ 서버 연결 실패 - 서버가 실행 중인지 확인하세요")
        print(f"서버 시작: python sar_download_api.py")


def test_download_with_event():
    """이벤트 날짜 기반 다운로드 테스트 (튀르키예 지진)"""
    print("\n=== 3. SAR Download Test (Event Date - 튀르키예 지진) ===")
    
    payload = {
        "latitude": 38.0,
        "longitude": 37.0,
        "location_name": "튀르키예 남부",
        "event_date": "2023-02-06",  # 지진 발생일
        # start_date, end_date는 자동 설정됨 (2022-02-06 ~ 2024-02-06)
        "buffer": 0.5,
        "max_results": 10,
        "select_insar_pair": True
    }
    
    print(f"Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\n🎯 Event Date 기반 다운로드:")
    print("  - Master: 2023-02-06 이후 중 가장 가까운 데이터")
    print("  - Slave: 2023-02-06 이전 중 가장 가까운 데이터")
    print("\n다운로드 시작... (시간이 걸릴 수 있습니다)")
    
    try:
        response = requests.post(
            f"{API_URL}/download",
            json=payload,
            timeout=600  # 10분
        )
        
        print(f"\nStatus: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
    except requests.exceptions.Timeout:
        print("⚠️ 요청 타임아웃 (10분 초과)")
    except requests.exceptions.ConnectionError:
        print("❌ 서버 연결 실패 - 서버가 실행 중인지 확인하세요")
        print(f"서버 시작: python sar_download_api.py")


if __name__ == "__main__":
    print("🧪 SAR Download API Test")
    print(f"API Server: {API_URL}")
    
    try:
        # 1. Health Check
        test_health_check()
        
        # 2. Download Test (일반)
        # test_download()  # 주석 처리 (시간이 오래 걸림)
        
        # 3. Download Test (Event Date)
        test_download_with_event()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ API 서버에 연결할 수 없습니다.")
        print("다음 명령어로 서버를 시작하세요:")
        print("  cd /home/mjh/Project/LLM/RAG/rag-study/agent_cv/sar_api")
        print("  python sar_download_api.py")
