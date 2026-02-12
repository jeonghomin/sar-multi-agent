"""DB 검색 및 문서 평가 노드"""
import json
import re
from math import radians, cos, sin, asin, sqrt
from evaluation.graders import retrieval_grader
import pdf_setup


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Haversine 공식으로 두 좌표 간 거리 계산 (km)
    """
    # 라디안으로 변환
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine 공식
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # 지구 반지름 (km)
    r = 6371
    
    return c * r


def extract_coordinates_from_chunk(page_content):
    """
    chunk에서 sceneCenterPointLla.coordinates 추출
    
    Returns:
        (latitude, longitude) or None
    """
    try:
        # sceneCenterPointLla 찾기
        if 'sceneCenterPointLla' not in page_content:
            return None
        
        # coordinates 배열 추출 (정규식)
        # "coordinates": [경도, 위도, 고도]
        pattern = r'"coordinates"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]'
        match = re.search(pattern, page_content)
        
        if match:
            lon = float(match.group(1))
            lat = float(match.group(2))
            # alt = float(match.group(3))  # 고도는 필요시 사용
            return (lat, lon)
    except Exception as e:
        print(f"좌표 추출 오류: {e}")
    
    return None


def extract_datetime_from_filename(source_path):
    """
    파일명에서 날짜/시간 추출
    
    예: 2024-04-13-12-43-27_UMBRA-05_METADATA.json
    → 2024-04-13 12:43:27 (UMBRA-05)
    
    Returns:
        str: 포맷팅된 날짜/시간 문자열 or None
    """
    try:
        import os
        filename = os.path.basename(source_path)
        
        # 패턴: YYYY-MM-DD-HH-MM-SS_SATELLITE_METADATA.json
        pattern = r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})_([A-Z0-9-]+)'
        match = re.search(pattern, filename)
        
        if match:
            year, month, day, hour, minute, second, satellite = match.groups()
            date_str = f"{year}-{month}-{day}"
            time_str = f"{hour}:{minute}:{second}"
            return f"{date_str} {time_str}", satellite
    except Exception as e:
        print(f"날짜 추출 오류: {e}")
    
    return None, None


def retrieve(state):
    """좌표 기반으로 SAR 이미지 메타데이터를 검색합니다."""
    print("==== [RETRIEVE SAR METADATA - 좌표 기반 필터링] ====")
    coordinates = state.get("coordinates")
    
    if pdf_setup is None or (pdf_setup.pdf_retriever is None and pdf_setup.vectorstore is None):
        print("경고: VectorStore가 초기화되지 않았습니다. RAG 기능이 비활성화됩니다.")
        return {"documents": []}
    
    if not coordinates:
        print("경고: 좌표 정보가 없어 검색할 수 없습니다.")
        return {"documents": []}
    
    target_lat = coordinates.get("latitude")
    target_lon = coordinates.get("longitude")
    location_name = coordinates.get("location", "")
    
    print(f"대상 좌표: 위도 {target_lat}, 경도 {target_lon} ({location_name})")
    
    # 거리 임계값 (km)
    DISTANCE_THRESHOLD = 50  # 50km 이내
    
    try:
        # Step 1: VectorStore에서 후보 문서 많이 가져오기 (k=100)
        search_query = f"sceneCenterPointLla coordinates {target_lat} {target_lon} {location_name}"
        print(f"1단계: VectorStore 검색 (k=100)...")
        
        if hasattr(pdf_setup, 'vectorstore') and pdf_setup.vectorstore is not None:
            candidate_docs = pdf_setup.vectorstore.similarity_search(search_query, k=100)
        else:
            candidate_docs = pdf_setup.pdf_retriever.invoke(search_query)
            candidate_docs = candidate_docs[:100]
        
        print(f"후보 문서: {len(candidate_docs)}개")
        
        # Step 2: 각 문서에서 좌표 추출 및 거리 계산
        print(f"2단계: 좌표 추출 및 거리 계산...")
        results = []
        
        for doc in candidate_docs:
            coords = extract_coordinates_from_chunk(doc.page_content)
            if coords:
                doc_lat, doc_lon = coords
                distance = haversine_distance(target_lat, target_lon, doc_lat, doc_lon)
                
                if distance <= DISTANCE_THRESHOLD:
                    # 날짜/시간 추출
                    source_path = doc.metadata.get("source", "unknown")
                    datetime_str, satellite = extract_datetime_from_filename(source_path)
                    
                    results.append({
                        "document": doc,
                        "latitude": doc_lat,
                        "longitude": doc_lon,
                        "distance_km": distance,
                        "source": source_path,
                        "datetime": datetime_str,
                        "satellite": satellite
                    })
        
        print(f"임계값({DISTANCE_THRESHOLD}km) 이내 문서: {len(results)}개")
        
        # Step 3: 거리순 정렬
        results.sort(key=lambda x: x["distance_km"])
        
        # 날짜 정보 확인
        date_range = state.get("date_range")
        
        if results:
            # 날짜 정보가 있으면 날짜 필터링
            if date_range:
                start_date = date_range.get("start_date")
                end_date = date_range.get("end_date")
                print(f"날짜 필터링: {start_date} ~ {end_date}")
                
                # 날짜 필터링 구현
                from datetime import datetime
                filtered_results = []
                
                for r in results:
                    datetime_str = r.get("datetime")
                    if datetime_str:
                        try:
                            # "2024-04-13 12:43:27" 형식에서 날짜 추출
                            file_date_str = datetime_str.split()[0]  # "2024-04-13"
                            file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                            start = datetime.strptime(start_date, "%Y-%m-%d")
                            end = datetime.strptime(end_date, "%Y-%m-%d")
                            
                            if start <= file_date <= end:
                                filtered_results.append(r)
                                print(f"  ✓ 날짜 매치: {datetime_str}")
                            else:
                                print(f"  ✗ 날짜 불일치: {datetime_str} (범위: {start_date}~{end_date})")
                        except Exception as e:
                            print(f"  ⚠️ 날짜 파싱 실패: {datetime_str} ({e})")
                            # 파싱 실패 시 결과에 포함
                            filtered_results.append(r)
                    else:
                        # 날짜 정보 없으면 결과에 포함
                        filtered_results.append(r)
                
                results = filtered_results
                print(f"날짜 필터링 후: {len(results)}개")
            
            # 가장 가까운 문서
            best_match = results[0]
            print(f"✓ 가장 가까운 SAR 이미지:")
            print(f"  - 거리: {best_match['distance_km']:.2f} km")
            print(f"  - 좌표: ({best_match['latitude']:.4f}, {best_match['longitude']:.4f})")
            print(f"  - 촬영일시: {best_match.get('datetime', 'N/A')} ({best_match.get('satellite', 'N/A')})")
            print(f"  - 파일: {best_match['source'][-80:]}")
            
            # 상위 5개 결과 요약
            if len(results) > 1:
                print(f"\n상위 {min(5, len(results))}개 결과:")
                for i, r in enumerate(results[:5], 1):
                    dt_info = f" [{r.get('datetime', 'N/A')}]" if r.get('datetime') else ""
                    print(f"  {i}. {r['distance_km']:.2f} km{dt_info} - {r['source'][-60:]}")
            
            return {
                "documents": [best_match["document"].page_content],
                "metadata": {
                    "source": best_match["source"],
                    "distance_km": best_match["distance_km"],
                    "coordinates": {
                        "latitude": best_match["latitude"],
                        "longitude": best_match["longitude"]
                    },
                    "datetime": best_match.get("datetime"),
                    "satellite": best_match.get("satellite"),
                    "all_results": [
                        {
                            "distance_km": r["distance_km"],
                            "latitude": r["latitude"],
                            "longitude": r["longitude"],
                            "source": r["source"],
                            "datetime": r.get("datetime"),
                            "satellite": r.get("satellite")
                        }
                        for r in results[:5]  # 상위 5개만
                    ]
                }
            }
        else:
            print(f"✗ {DISTANCE_THRESHOLD}km 이내에 SAR 데이터가 없습니다.")
            
            # 날짜 정보가 없으면 사용자에게 물어보기 위한 플래그 설정
            date_range = state.get("date_range")
            needs_date = date_range is None
            
            return {
                "documents": [],
                "needs_date_search": needs_date,  # ✅ 날짜 검색 필요 플래그
                "metadata": {
                    "error": f"{DISTANCE_THRESHOLD}km 이내에 데이터 없음",
                    "searched_area": f"({target_lat}, {target_lon}) ± {DISTANCE_THRESHOLD}km",
                    "needs_date": needs_date
                }
            }
            
    except Exception as e:
        print(f"검색 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"documents": []}


def grade_document(state):
    """문서 관련성 평가 및 다운로드 확인"""
    from langchain_core.messages import AIMessage
    
    question = state.get("question", "")
    documents = state.get("documents", [])
    metadata = state.get("metadata")
    location_name = state.get("location_name", "해당 지역")
    coordinates = state.get("coordinates") or {}  # None일 수도 있으므로 or {} 사용
    date_range = state.get("date_range") or {}    # None일 수도 있으므로 or {} 사용
    
    # 문서가 없는 경우: ASF 다운로드 제안
    if not documents or len(documents) == 0:
        print("==== [GRADE_DOCUMENT: 문서 없음 - ASF 다운로드 제안] ====")
        
        lat = coordinates.get("latitude", "N/A") if coordinates else "N/A"
        lon = coordinates.get("longitude", "N/A") if coordinates else "N/A"
        start_date = date_range.get("start_date", "N/A") if date_range else "N/A"
        end_date = date_range.get("end_date", "N/A") if date_range else "N/A"
        event_date = date_range.get("event_date", "N/A") if date_range else "N/A"
        
        message = f"""ℹ️ 로컬 데이터베이스에 {location_name}의 SAR 데이터가 없습니다.

📍 검색 위치: {location_name} ({lat}, {lon})
📅 검색 기간: {start_date} ~ {end_date}
🎯 이벤트 날짜: {event_date}

ASF (Alaska Satellite Facility)에서 Sentinel-1 데이터를 다운로드 받으시겠습니까?
"""
        
        return {
            "documents": [],
            "generation": message,
            "awaiting_download_confirmation": True,
            "messages": [AIMessage(content=message)]
        }
    
    # 문서가 있는 경우: 관련성 평가
    print(f"==== [GRADE_DOCUMENT: {len(documents)}개 문서 평가] ====")
    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke({
            "question": question,
            "document": doc.page_content if hasattr(doc, 'page_content') else str(doc),
        })
        if score.binary_score == "yes":
            filtered_docs.append(doc)
    
    print(f"✅ 관련성 평가 통과: {len(filtered_docs)}/{len(documents)}개")
    
    result = {"documents": filtered_docs, "awaiting_download_confirmation": False}
    if metadata:
        result["metadata"] = metadata
    return result
