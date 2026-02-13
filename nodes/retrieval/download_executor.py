"""SAR 다운로드 실행 함수들"""
import requests
from langchain_core.messages import AIMessage
from .download_helpers import validate_indices, get_date_range

SAR_DOWNLOAD_API_URL = "http://localhost:8001"


def execute_download_insar(state, search_result, master_idx, slave_idx):
    """Master/Slave 선택 후 다운로드 실행 - InSAR용"""
    products = search_result.get('products', [])
    ok, max_idx, invalid = validate_indices(products, [master_idx, slave_idx])
    
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
        start_date, end_date = get_date_range(state)
        
        payload = {
            "latitude": lat,
            "longitude": lon,
            "location_name": location,
            "start_date": start_date,
            "end_date": end_date,
            "buffer": 0.5,
            "max_results": 500,  # ⭐ 클라이언트 검색과 동일하게 500으로 설정
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


def execute_download_single(state, search_result, selected_indices):
    """SAR 데이터 선택 후 다운로드 실행 - 단일 또는 다중 지원"""
    products = search_result.get('products', [])
    
    if not isinstance(selected_indices, list):
        selected_indices = [selected_indices]
    
    ok, max_idx, invalid_indices = validate_indices(products, selected_indices)
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
            print(f"[EXECUTOR DEBUG] display_index {idx}에 해당하는 제품을 찾을 수 없음!")
            continue
        original_idx = product.get('original_index', idx)
        print(f"[EXECUTOR DEBUG] display_index={idx} → original_index={original_idx}, 날짜={product.get('date')}, 파일명={product.get('filename', '')[:60]}")
        selected_products.append(product)
        selected_original_indices.append(original_idx)
    
    if len(selected_original_indices) == 0:
        msg = f"❌ 선택한 인덱스 {selected_indices}에 해당하는 제품을 찾을 수 없습니다."
        return {
            "generation": msg,
            "messages": [AIMessage(content=msg)],
            "awaiting_single_sar_selection": True,
        }
    
    try:
        coordinates = state.get("coordinates", {})
        lat = coordinates.get("latitude", 0)
        lon = coordinates.get("longitude", 0)
        location = coordinates.get("location", "Unknown")
        start_date, end_date = get_date_range(state)
        
        payload = {
            "latitude": lat,
            "longitude": lon,
            "location_name": location,
            "start_date": start_date,
            "end_date": end_date,
            "buffer": 0.5,
            "max_results": 500,  # ⭐ 클라이언트 검색과 동일하게 500으로 설정
            "selected_indices": selected_original_indices
        }
        
        print(f"[EXECUTOR DEBUG] API로 전달할 payload: selected_original_indices={selected_original_indices}")
        
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
                "awaiting_single_sar_selection": False,
                "sar_search_results": None
            }
        
        result = response.json()
        
        if not result['success']:
            msg = f"❌ 다운로드 실패: {result.get('message', '알 수 없는 오류')}"
            return {
                "generation": msg,
                "messages": [AIMessage(content=msg)],
                "awaiting_single_sar_selection": False,
                "sar_search_results": None
            }
        
        # 성공 메시지
        dl = result.get('download_result', {})
        generation = f"""✅ **Sentinel-1 다운로드 완료!**

📍 **위치**: {location} ({lat}, {lon})
📅 **검색 기간**: {start_date} ~ {end_date}

🎯 **선택된 데이터** ({len(selected_products)}개):
"""
        for idx, product in zip(selected_indices, selected_products):
            generation += f"- [{idx}] {product['date']} - {product['filename'][:50]}...\n"
        
        generation += f"""
📊 **다운로드 결과**:
- 다운로드: {dl.get('downloaded', 0)}개
- 스킵 (이미 존재): {dl.get('skipped', 0)}개
- 실패: {dl.get('failed', 0)}개

📁 **저장 경로**: `{dl.get('save_path', 'N/A')}`
"""
        
        files = dl.get('files', [])
        if files:
            generation += "\n**다운로드된 파일**:\n"
            for f in files[:5]:
                generation += f"- {f}\n"
            if len(files) > 5:
                generation += f"... 외 {len(files) - 5}개\n"
        
        return {
            "generation": generation,
            "messages": [AIMessage(content=generation)],
            "awaiting_single_sar_selection": False,
            "sar_search_results": None,
            "awaiting_download_confirmation": False,
            "sar_image_path": dl.get('save_path'),
            "downloaded_sar_files": dl.get('files', [])
        }
        
    except requests.exceptions.ConnectionError:
        msg = "❌ SAR Download API 서버에 연결할 수 없습니다"
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
