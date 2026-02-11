"""InSAR 데이터셋 설정 파일

새로운 InSAR 데이터를 추가하려면 INSAR_DATASETS 딕셔너리에 항목을 추가하세요.
"""

INSAR_DATASETS = {
    # 터키 2023년 지진
    "turkey_2023_earthquake": {
        "keywords": ["터키", "turkey", "kahramanmaras", "카흐라만마라슈", "튀르키예"],
        "event_keywords": ["지진", "earthquake"],
        "location": "Turkey",
        "event": "2023 Turkey-Syria Earthquake",
        "date_range": "2023-01-29 to 2023-02-10",
        "file_path": "/home/mjh/Project/LLM/RAG/files/Phase_ifg_VV_29Jan2023_10Feb2023.kmz",
        "description": "2023년 터키-시리아 국경 지진 전후 InSAR 간섭무늬 분석 결과입니다. 지진 발생 전(2023-01-29)과 발생 후(2023-02-10) 시점의 SAR 이미지를 이용한 위상 간섭 무늬(Interferogram)를 보여줍니다.",
        "coordinates": {"lat": 37.23, "lon": 37.04}
    },
    
    # 추가 데이터셋 예시 (필요시 주석 해제 및 수정)
    # "japan_fuji_volcano": {
    #     "keywords": ["후지산", "fuji", "japan", "일본"],
    #     "event_keywords": ["화산", "volcano", "volcanic"],
    #     "location": "Japan",
    #     "event": "Mount Fuji Volcanic Activity Monitoring",
    #     "date_range": "2023-01-01 to 2023-12-31",
    #     "file_path": "/path/to/fuji_insar.kmz",
    #     "description": "후지산 화산 활동 모니터링을 위한 InSAR 간섭무늬 분석 결과입니다.",
    #     "coordinates": {"lat": 35.36, "lon": 138.73}
    # },
    
    # "california_sanandreas": {
    #     "keywords": ["california", "캘리포니아", "san andreas", "샌안드레아스"],
    #     "event_keywords": ["지진", "earthquake", "fault", "단층"],
    #     "location": "California, USA",
    #     "event": "San Andreas Fault Deformation Monitoring",
    #     "date_range": "2023-01-01 to 2023-12-31",
    #     "file_path": "/path/to/sanandreas_insar.kmz",
    #     "description": "샌안드레아스 단층 지표 변형 모니터링을 위한 InSAR 분석 결과입니다.",
    #     "coordinates": {"lat": 34.00, "lon": -118.25}
    # },
    
    # "iceland_volcano": {
    #     "keywords": ["iceland", "아이슬란드", "reykjanes", "레이캬네스"],
    #     "event_keywords": ["화산", "volcano", "volcanic", "eruption"],
    #     "location": "Iceland",
    #     "event": "Reykjanes Peninsula Volcanic Activity",
    #     "date_range": "2023-11-01 to 2024-01-31",
    #     "file_path": "/path/to/iceland_insar.kmz",
    #     "description": "아이슬란드 레이캬네스 반도 화산 활동 모니터링을 위한 InSAR 분석 결과입니다.",
    #     "coordinates": {"lat": 63.88, "lon": -22.45}
    # },
}


def get_available_datasets():
    """사용 가능한 데이터셋 목록 반환"""
    return [
        f"- {ds['location']}: {ds['event']} ({ds['date_range']})"
        for ds in INSAR_DATASETS.values()
    ]


def find_dataset_by_keywords(text: str, coordinates: dict = None):
    """
    키워드 또는 좌표로 데이터셋 검색
    
    Args:
        text: 검색할 텍스트 (질문, 대화 히스토리 등)
        coordinates: 좌표 딕셔너리 (lat, lon)
    
    Returns:
        매칭된 데이터셋 정보 또는 None
    """
    text_lower = text.lower()
    
    # 키워드 기반 매칭
    for dataset_id, dataset_info in INSAR_DATASETS.items():
        # 지역 키워드 체크
        location_match = any(kw in text_lower for kw in dataset_info["keywords"])
        
        # 이벤트 키워드 체크 (선택 사항)
        event_keywords = dataset_info.get("event_keywords", [])
        event_match = not event_keywords or any(kw in text_lower for kw in event_keywords)
        
        if location_match and event_match:
            return dataset_info
    
    # 좌표 기반 매칭
    if coordinates:
        user_lat = coordinates.get("latitude")
        user_lon = coordinates.get("longitude")
        
        if user_lat and user_lon:
            min_distance = float('inf')
            closest_dataset = None
            
            for dataset_info in INSAR_DATASETS.values():
                if "coordinates" in dataset_info:
                    ds_lat = dataset_info["coordinates"]["lat"]
                    ds_lon = dataset_info["coordinates"]["lon"]
                    # 간단한 유클리드 거리 계산
                    distance = ((user_lat - ds_lat)**2 + (user_lon - ds_lon)**2)**0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_dataset = dataset_info
            
            # 5도 이내면 매칭으로 간주
            if min_distance < 5.0:
                return closest_dataset
    
    return None
