"""SAR 데이터 다운로드 노드"""
import json
import requests
from core.llm_config import llm
from location_utils import location_to_coordinates
from langchain_core.messages import AIMessage

from .download_helpers import (
    extract_event_date,
    extract_location_from_question,
    auto_select_for_insar,
    parse_master_slave_selection,
    is_new_search_request,
    validate_indices,
    parse_single_selection
)
from .download_formatter import (
    filter_and_group_by_event,
    format_search_results_header,
    format_products_by_orbit,
    build_insar_selection_message,
    build_single_selection_message
)
from .download_executor import (
    execute_download_insar,
    execute_download_single
)

SAR_DOWNLOAD_API_URL = "http://localhost:8001"
SAR_DOWNLOAD_AVAILABLE = True


def download_sar(state):
    """ASF에서 Sentinel-1 데이터 다운로드"""
    print("==== [DOWNLOAD SAR] ====")
    question = state.get("question", "")
    coordinates = state.get("coordinates")
    location_name = state.get("location_name")
    
    auto_insar = state.get("auto_insar_after_download", False)
    needs_insar = state.get("needs_insar", False) or auto_insar
    insar_keywords = ["insar", "간섭무늬", "interferogram", "지표변형", "ground deformation", "master", "slave"]
    if not needs_insar:
        needs_insar = any(keyword in question.lower() for keyword in insar_keywords)
    
    # InSAR용: 2단계 플로우 (Master/Slave 선택)
    if needs_insar:
        awaiting_selection = state.get("awaiting_master_slave_selection", False)
        sar_search_results = state.get("sar_search_results")
        
        if awaiting_selection and sar_search_results:
            if is_new_search_request(question):
                awaiting_selection = False
            else:
                master_idx, slave_idx = parse_master_slave_selection(question)
                
                if master_idx is None or slave_idx is None:
                    msg = "❌ Master/Slave 인덱스를 파싱할 수 없습니다. 형식: 'Master 1, Slave 5' 또는 '1번과 5번'"
                    return {
                        "generation": msg,
                        "messages": [AIMessage(content=msg)],
                        "awaiting_master_slave_selection": True,
                    }
                products = sar_search_results.get('products', [])
                if products and 'file_path' in products[0]:
                    ok, max_idx, _ = validate_indices(products, [master_idx, slave_idx])
                    if not ok:
                        msg = f"❌ 인덱스 범위 초과 (Master: {master_idx}, Slave: {slave_idx}, 최대: {max_idx})"
                        return {
                            "generation": msg,
                            "messages": [AIMessage(content=msg)],
                            "awaiting_master_slave_selection": True,
                        }
                    # ⭐ display_index로 제품 찾기
                    master_product = next((p for p in products if p.get('display_index') == master_idx), None)
                    slave_product = next((p for p in products if p.get('display_index') == slave_idx), None)
                    
                    if not master_product or not slave_product:
                        msg = f"❌ 선택한 인덱스를 찾을 수 없습니다: Master[{master_idx}], Slave[{slave_idx}]"
                        return {
                            "generation": msg,
                            "messages": [AIMessage(content=msg)],
                            "awaiting_master_slave_selection": True,
                        }
                    
                    master_file = master_product['file_path']
                    slave_file = slave_product['file_path']
                    return {
                        "generation": f"✅ Master와 Slave를 선택했습니다. InSAR 처리를 시작합니다...",
                        "messages": [AIMessage(content="✅ Master와 Slave를 선택했습니다. InSAR 처리를 시작합니다...")],
                        "downloaded_sar_files": [master_file, slave_file],  # 순서 중요!
                        "awaiting_master_slave_selection": False,
                        "needs_insar": True,  # run_insar로 라우팅
                    }
                else:
                    # ASF 다운로드 필요
                    return execute_download_insar(
                        state,
                        sar_search_results,
                        master_idx,
                        slave_idx
                    )
    
    else:
        awaiting_selection = state.get("awaiting_single_sar_selection", False)
        sar_search_results = state.get("sar_search_results")
        
        if awaiting_selection and sar_search_results:
            if is_new_search_request(question):
                awaiting_selection = False
            else:
                print(f"[DOWNLOAD DEBUG] parse_single_selection 호출 중... 질문: {question}")
                selected_indices = parse_single_selection(question, llm)
                print(f"[DOWNLOAD DEBUG] parse_single_selection 완료! 결과: {selected_indices}")
                
                if selected_indices is None or len(selected_indices) == 0:
                    msg = "❌ 인덱스를 파싱할 수 없습니다. 형식: '1번' 또는 '1,2,3'"
                    return {
                        "generation": msg,
                        "messages": [AIMessage(content=msg)],
                        "awaiting_single_sar_selection": True,
                    }
                return execute_download_single(
                    state,
                    sar_search_results,
                    selected_indices
                )
    
    if not SAR_DOWNLOAD_AVAILABLE:
        msg = "❌ asf_search 미설치: pip install asf_search"
        return {
            "generation": msg,
            "messages": [AIMessage(content=msg)],
            "awaiting_download_confirmation": False
        }
    
    if not coordinates and location_name:
        coords = location_to_coordinates(location_name)
        if coords:
            try:
                coordinates = json.loads(coords) if isinstance(coords, str) else coords
            except Exception:
                coordinates = coords
    if not coordinates:
        location = extract_location_from_question(question, llm)
        if location:
            coords = location_to_coordinates(location)
            if coords:
                try:
                    coordinates = json.loads(coords) if isinstance(coords, str) else coords
                except:
                    coordinates = coords
    
    if not coordinates:
        msg = f"❌ 좌표 정보를 찾을 수 없습니다. 지역명을 포함해서 다시 요청해주세요. 예: \"이바라키 다운로드\""
        return {
            "generation": msg,
            "messages": [AIMessage(content=msg)],
            "awaiting_download_confirmation": False
        }
    
    lat = coordinates.get("latitude")
    lon = coordinates.get("longitude")
    location = coordinates.get("location", f"({lat}, {lon})")
    
    # 날짜 범위 추출 (이벤트 발생일 기준 ±3개월)
    start_date = None
    end_date = None
    event_date = None
    
    date_range = state.get("date_range")
    if date_range:
        event_date = date_range.get("event_date")
    if not event_date:
        event_date = extract_event_date(question, llm)
    if not event_date:
        summary = state.get("summary", "")
        if summary:
            event_date = extract_event_date(summary, llm)
    
    if event_date:
        from datetime import datetime, timedelta
        try:
            event_dt = datetime.strptime(event_date, "%Y-%m-%d")
            start_date = (event_dt - timedelta(days=730)).strftime("%Y-%m-%d")
            end_date = (event_dt + timedelta(days=730)).strftime("%Y-%m-%d")
        except Exception:
            start_date, end_date = "2022-01-01", "2024-12-31"
    else:
        start_date, end_date = "2022-01-01", "2024-12-31"
    
    try:
        search_payload = {
            "latitude": lat,
            "longitude": lon,
            "location_name": location,
            "start_date": start_date,
            "end_date": end_date,
            "buffer": 0.5,
            "max_results": 50
        }
        
        search_response = requests.post(
            f"{SAR_DOWNLOAD_API_URL}/search",
            json=search_payload,
            timeout=60
        )
        
        if search_response.status_code != 200:
            msg = f"❌ API 서버 오류 (HTTP {search_response.status_code})"
            return {
                "generation": msg,
                "messages": [AIMessage(content=msg)],
                "awaiting_download_confirmation": False
            }
        
        search_result = search_response.json()
        
        if not search_result['success'] or search_result['total'] == 0:
            msg = f"ℹ️ {location}에서 SAR 데이터를 찾을 수 없습니다. 기간: {start_date} ~ {end_date}"
            return {
                "generation": msg,
                "messages": [AIMessage(content=msg)],
                "awaiting_download_confirmation": False
            }
        
        products = search_result['products']
        total = search_result['total']
        actual_date_range = search_result.get('date_range', 'N/A')
        
        # 이벤트 날짜 기준으로 제품 필터링 및 그룹화
        products, event_info = filter_and_group_by_event(products, event_date, display_limit=10)
        display_limit = len(products)
        
        # 헤더 생성
        generation = format_search_results_header(
            location, total, start_date, end_date, actual_date_range, lat, lon, event_info
        )
        
        # 제품 리스트 포맷팅 (Orbit별 그룹화)
        generation += format_products_by_orbit(products, needs_insar=needs_insar)
        
        if total > display_limit:
            generation += f"\n... 외 {total - display_limit}개\n"
        
        filtered_search_result = {
            'success': search_result['success'],
            'total': len(products),
            'products': products,
            'date_range': search_result.get('date_range', ''),
            'location': search_result.get('location', location)
        }
        
        if needs_insar:
            if auto_insar and event_date:
                master_idx, slave_idx = auto_select_for_insar(products, event_date)
                
                if master_idx is None or slave_idx is None:
                    msg = f"❌ InSAR용 데이터를 자동 선택할 수 없습니다. 이벤트 날짜({event_date}) 기준 전/후 데이터 없음.\n\n{generation}\n\n수동 선택: 'Master 1, Slave 5'"
                    return {
                        "generation": msg,
                        "messages": [AIMessage(content=msg)],
                        "awaiting_master_slave_selection": True,
                        "awaiting_single_sar_selection": False,
                        "sar_search_results": filtered_search_result,
                        "auto_insar_after_download": False,
                    }
                
                download_result = execute_download_insar(
                    state,
                    filtered_search_result,
                    master_idx,
                    slave_idx
                )
                
                download_msg = download_result.get("generation", "")
                confirmation_msg = f"""{download_msg}

✅ **2개 데이터 다운로드 완료!**

🤖 InSAR 처리를 진행할까요?
- "네" 또는 "진행"을 입력하시면 InSAR 분석을 시작합니다.
- "취소"를 입력하시면 중단합니다.
"""
                
                return {
                    "generation": confirmation_msg,
                    "messages": [AIMessage(content=confirmation_msg)],
                    "downloaded_sar_files": download_result.get("downloaded_sar_files", []),
                    "sar_image_path": download_result.get("sar_image_path"),
                    "awaiting_insar_confirmation": True,
                    "awaiting_master_slave_selection": False,
                    "awaiting_single_sar_selection": False,
                    "auto_insar_after_download": False,
                }
            
            generation += build_insar_selection_message()
            return {
                "generation": generation,
                "messages": [AIMessage(content=generation)],
                "awaiting_master_slave_selection": True,
                "awaiting_single_sar_selection": False,
                "sar_search_results": filtered_search_result,
                "awaiting_download_confirmation": False,
                "needs_insar": True,
            }
        else:
            generation += build_single_selection_message()
            return {
                "generation": generation,
                "messages": [AIMessage(content=generation)],
                "awaiting_master_slave_selection": False,
                "awaiting_single_sar_selection": True,
                "sar_search_results": filtered_search_result,
                "awaiting_download_confirmation": False,
                "needs_insar": False,
            }
        
    except requests.exceptions.ConnectionError:
        msg = f"❌ SAR Download API 서버에 연결할 수 없습니다. 서버 시작: python sar_download_api.py ({SAR_DOWNLOAD_API_URL})"
        return {
            "generation": msg,
            "awaiting_download_confirmation": False,
            "messages": [AIMessage(content=msg)]
        }
    
    except Exception as e:
        msg = f"❌ 다운로드 오류: {str(e)}"
        return {
            "generation": msg,
            "awaiting_download_confirmation": False,
            "messages": [AIMessage(content=msg)]
        }


def _execute_download_insar(state, search_result, master_idx, slave_idx):
    """Master/Slave 선택 후 다운로드 실행 - InSAR용"""
    products = search_result.get('products', [])
    ok, max_idx, invalid = _validate_indices(products, [master_idx, slave_idx])
    if not ok:
        msg = f"❌ 인덱스 범위 초과 (Master: {master_idx}, Slave: {slave_idx}, 최대: {max_idx})"
        return {
            "generation": msg,
            "messages": [AIMessage(content=msg)],
            "awaiting_master_slave_selection": True,
        }
    master_product = next((p for p in products if p.get('display_index') == master_idx), None)
    slave_product = next((p for p in products if p.get('display_index') == slave_idx), None)
    if not master_product or not slave_product:
        msg = f"❌ 선택한 인덱스를 찾을 수 없습니다: Master[{master_idx}], Slave[{slave_idx}]"
        return {
            "generation": msg,
            "messages": [AIMessage(content=msg)],
            "awaiting_master_slave_selection": True,
        }
    master_original_idx = master_product.get('original_index', master_idx)
    slave_original_idx = slave_product.get('original_index', slave_idx)
    try:
        coordinates = state.get("coordinates", {})
        lat = coordinates.get("latitude", 0)
        lon = coordinates.get("longitude", 0)
        location = coordinates.get("location", "Unknown")
        start_date, end_date = _get_date_range(state)
        payload = {
            "latitude": lat,
            "longitude": lon,
            "location_name": location,
            "start_date": start_date,
            "end_date": end_date,
            "buffer": 0.5,
            "max_results": 50,
            "master_index": master_original_idx,
            "slave_index": slave_original_idx
        }
        response = requests.post(
            f"{SAR_DOWNLOAD_API_URL}/download",
            json=payload,
            timeout=1800
        )
        
        if response.status_code != 200:
            msg = f"❌ API 서버 오류 (HTTP {response.status_code})"
            return {
                "generation": msg,
                "messages": [AIMessage(content=msg)],
                "awaiting_master_slave_selection": False,
                "sar_search_results": None
            }
        
        result = response.json()
        
        if not result['success']:
            msg = f"❌ 다운로드 실패: {result.get('message', '알 수 없는 오류')}"
            return {
                "generation": msg,
                "messages": [AIMessage(content=msg)],
                "awaiting_master_slave_selection": False,
                "sar_search_results": None
            }
        
        # 성공 메시지
        dl = result.get('download_result', {})
        generation = f"""✅ **Sentinel-1 다운로드 완료!**

📍 **위치**: {location} ({lat}, {lon})
📅 **검색 기간**: {start_date} ~ {end_date}

🎯 **선택된 데이터**:
- **Master**: [{master_idx}] {master_product['date']} - {master_product['filename'][:60]}...
- **Slave**: [{slave_idx}] {slave_product['date']} - {slave_product['filename'][:60]}...

📊 **다운로드 결과**:
- 다운로드: {dl.get('downloaded', 0)}개
- 스킵 (이미 존재): {dl.get('skipped', 0)}개
- 실패: {dl.get('failed', 0)}개

📁 **저장 경로**: `{dl.get('save_path', 'N/A')}`

✅ InSAR 처리를 진행할 수 있습니다!
"""
        
        files = dl.get('files', [])
        if files:
            generation += "\n**다운로드된 파일**:\n"
            for f in files[:3]:
                generation += f"- {f}\n"
            if len(files) > 3:
                generation += f"... 외 {len(files) - 3}개\n"
        
        return {
            "generation": generation,
            "messages": [AIMessage(content=generation)],
            "awaiting_master_slave_selection": False,
            "sar_search_results": None,
            "awaiting_download_confirmation": False,
            "sar_image_path": dl.get('save_path'),
            "downloaded_sar_files": dl.get('files', [])
        }
        
    except requests.exceptions.ConnectionError:
        msg = "❌ SAR Download API 서버에 연결할 수 없습니다. python sar_download_api.py 실행 필요"
        return {
            "generation": msg,
            "messages": [AIMessage(content=msg)],
            "awaiting_master_slave_selection": False,
            "sar_search_results": None
        }
    except Exception as e:
        msg = f"❌ 다운로드 중 오류: {str(e)}"
        return {
            "generation": msg,
            "messages": [AIMessage(content=msg)],
            "awaiting_master_slave_selection": False,
            "sar_search_results": None
        }


def _execute_download_single(state, search_result, selected_indices):
    """SAR 데이터 선택 후 다운로드 실행 - 단일 또는 다중 지원"""
    products = search_result.get('products', [])
    if not isinstance(selected_indices, list):
        selected_indices = [selected_indices]
    ok, max_idx, invalid_indices = _validate_indices(products, selected_indices)
    if not ok:
        msg = f"❌ 인덱스 범위 초과 (잘못된 인덱스: {invalid_indices}, 최대: {max_idx})"
        return {
            "generation": msg,
            "messages": [AIMessage(content=msg)],
            "awaiting_single_sar_selection": True,
        }
    selected_products = []
    selected_original_indices = []
    
    for idx in selected_indices:
        product = next((p for p in products if p.get('display_index') == idx), None)
        
        if not product:
            continue
        original_idx = product.get('original_index', idx)
        selected_products.append(product)
        selected_original_indices.append(original_idx)
    
    try:
        coordinates = state.get("coordinates", {})
        lat = coordinates.get("latitude", 0)
        lon = coordinates.get("longitude", 0)
        location = coordinates.get("location", "Unknown")
        start_date, end_date = _get_date_range(state)
        download_results = []
        failed_downloads = []
        for i, original_idx in enumerate(selected_original_indices):
            product = selected_products[i]
            display_idx = selected_indices[i]
            payload = {
                "latitude": lat,
                "longitude": lon,
                "location_name": location,
                "start_date": start_date,
                "end_date": end_date,
                "buffer": 0.5,
                "max_results": 50,
                "master_index": original_idx,
                "slave_index": None
            }
            
            try:
                response = requests.post(
                    f"{SAR_DOWNLOAD_API_URL}/download",
                    json=payload,
                    timeout=1800  # 30분 (SAR 데이터는 7-8GB로 매우 큼)
                )
                
                if response.status_code != 200:
                    failed_downloads.append({
                        'index': display_idx,
                        'product': product,
                        'error': f"HTTP {response.status_code}"
                    })
                    continue
                
                result = response.json()
                
                if not result['success']:
                    failed_downloads.append({
                        'index': display_idx,
                        'product': product,
                        'error': result.get('message', '알 수 없는 오류')
                    })
                    continue
                
                dl = result.get('download_result', {})
                download_results.append({
                    'index': display_idx,
                    'product': product,
                    'result': dl
                })
            except Exception as e:
                failed_downloads.append({
                    'index': display_idx,
                    'product': product,
                    'error': str(e)
                })
        
        if len(download_results) == 0:
            msg = f"❌ 모든 다운로드 실패 ({location}, {len(failed_downloads)}개)\n"
            for fail in failed_downloads[:5]:
                msg += f"- [{fail['index']}] {fail['product']['date']}: {fail['error']}\n"
            if len(failed_downloads) > 5:
                msg += f"... 외 {len(failed_downloads) - 5}개\n"
            
            return {
                "generation": msg,
                "messages": [AIMessage(content=msg)],
                "awaiting_single_sar_selection": False,
                "sar_search_results": None
            }
        
        # 성공한 다운로드 통계
        total_downloaded = sum([r['result'].get('downloaded', 0) for r in download_results])
        total_skipped = sum([r['result'].get('skipped', 0) for r in download_results])
        total_failed = sum([r['result'].get('failed', 0) for r in download_results])
        
        # 저장 경로 (첫 번째 성공한 다운로드의 경로)
        save_path = download_results[0]['result'].get('save_path', 'N/A')
        
        # 다운로드된 파일 리스트 수집 (InSAR용)
        all_downloaded_files = []
        for r in download_results:
            files = r['result'].get('files', [])
            all_downloaded_files.extend(files)
        
        generation = f"""✅ **Sentinel-1 다운로드 완료!**

📍 **위치**: {location} ({lat}, {lon})
📅 **검색 기간**: {start_date} ~ {end_date}

🎯 **선택된 데이터** ({len(selected_indices)}개):
"""
        
        # 성공한 다운로드 목록
        for res in download_results[:10]:  # 최대 10개만 표시
            generation += f"- ✅ [{res['index']}] {res['product']['date']} - {res['product']['filename'][:50]}...\n"
        if len(download_results) > 10:
            generation += f"... 외 {len(download_results) - 10}개\n"
        
        # 실패한 다운로드 목록
        if failed_downloads:
            generation += f"\n❌ **실패한 데이터** ({len(failed_downloads)}개):\n"
            for fail in failed_downloads[:5]:
                generation += f"- [{fail['index']}] {fail['product']['date']}: {fail['error']}\n"
            if len(failed_downloads) > 5:
                generation += f"... 외 {len(failed_downloads) - 5}개\n"
        
        generation += f"""

📊 **다운로드 결과**:
- 다운로드: {total_downloaded}개
- 스킵 (이미 존재): {total_skipped}개
- 실패: {total_failed}개

📁 **저장 경로**: `{save_path}`

✅ SAR 데이터 분석을 진행할 수 있습니다!
"""
        
        return {
            "generation": generation,
            "messages": [AIMessage(content=generation)],
            "awaiting_single_sar_selection": False,
            "sar_search_results": None,
            "awaiting_download_confirmation": False,
            "sar_image_path": save_path,  # 분석을 위해 경로 저장
            "downloaded_sar_files": all_downloaded_files  # 방금 다운로드한 파일 리스트 (InSAR용)
        }
        
    except requests.exceptions.ConnectionError:
        msg = f"""❌ SAR Download API 서버에 연결할 수 없습니다.

🚀 서버를 시작해주세요:
```bash
cd /home/mjh/Project/LLM/RAG/rag-study/agent_cv/sar_api
python sar_download_api.py
```
"""
        return {
            "generation": msg,
            "messages": [AIMessage(content=msg)],
            "awaiting_single_sar_selection": False,
            "sar_search_results": None
        }
    
    except Exception as e:
        msg = f"❌ 다운로드 중 오류: {str(e)}"
        return {
            "generation": msg,
            "messages": [AIMessage(content=msg)],
            "awaiting_single_sar_selection": False,
            "sar_search_results": None
        }
