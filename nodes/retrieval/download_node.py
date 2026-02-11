"""SAR 데이터 다운로드 노드"""
import json
import requests
from core.llm_config import llm
from location_utils import location_to_coordinates
from langchain_core.messages import AIMessage

SAR_DOWNLOAD_API_URL = "http://localhost:8001"
SAR_DOWNLOAD_AVAILABLE = True


def _extract_event_date(question, llm):
    """질문에서 이벤트 발생 날짜 추출 (헬퍼 함수)"""
    prompt = f"""질문에서 지진/화산 등 이벤트 발생 날짜를 추출하세요:
질문: {question}

출력 형식: YYYY-MM-DD (날짜가 없으면 '없음')

예시:
질문: "터키 2023년 2월 6일 지진"
출력: 2023-02-06

질문: "2011년 일본 도호쿠 지진"
출력: 2011-03-11
"""
    try:
        response = llm.invoke(prompt)
        text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        for line in text.split('\n'):
            line = line.strip()
            if line and line.lower() not in ['없음', 'none', 'no', '']:
                if len(line) == 10 and line[4] == '-' and line[7] == '-':
                    return line
        
        return None
    except:
        return None


def _extract_location_from_question(question, llm):
    """질문에서 지역명 추출 (헬퍼 함수)"""
    prompt = f"질문에서 지역명 추출: {question}\n지역명만 출력 (없으면 '없음'):"
    try:
        response = llm.invoke(prompt)
        location = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        if location and location.lower() not in ["없음", "none", "no", ""]:
            return location
    except:
        pass
    return None


def _auto_select_for_insar(products, event_date):
    """InSAR용 자동 2개 선택 (이벤트 날짜 기준 전후)"""
    from datetime import datetime
    
    if not products or len(products) < 2:
        return None, None
    
    try:
        if '-' in event_date:
            event_dt = datetime.strptime(event_date, '%Y-%m-%d')
        else:
            event_dt = datetime.strptime(event_date, '%Y%m%d')
    except:
        return products[0].get('display_index', 0), products[1].get('display_index', 1)
    
    products_with_distance = []
    for p in products:
        try:
            p_date = p.get('date', '')
            if not p_date:
                continue
            if '-' in p_date:
                p_dt = datetime.strptime(p_date, '%Y-%m-%d')
            else:
                p_dt = datetime.strptime(p_date, '%Y%m%d')
            
            days_diff = (p_dt - event_dt).days
            products_with_distance.append({
                'product': p,
                'days_diff': days_diff,
                'abs_days_diff': abs(days_diff)
            })
        except:
            continue
    
    if len(products_with_distance) < 2:
        return products[0].get('display_index', 0), products[1].get('display_index', 1)
    
    products_with_distance.sort(key=lambda x: x['abs_days_diff'])
    before_products = [p for p in products_with_distance if p['days_diff'] < 0]
    after_products = [p for p in products_with_distance if p['days_diff'] >= 0]
    
    master_idx = None
    slave_idx = None
    
    if before_products and after_products:
        master_idx = before_products[0]['product'].get('display_index')
        slave_idx = after_products[0]['product'].get('display_index')
    else:
        master_idx = products_with_distance[0]['product'].get('display_index')
        slave_idx = products_with_distance[1]['product'].get('display_index')
    
    return master_idx, slave_idx


def _parse_master_slave_selection(question):
    """사용자 응답에서 Master/Slave 인덱스 추출 (InSAR용)"""
    import re
    
    master_match = re.search(r'[Mm]aster[\s:]*(\d+)', question)
    slave_match = re.search(r'[Ss]lave[\s:]*(\d+)', question)
    
    if master_match and slave_match:
        return int(master_match.group(1)), int(slave_match.group(1))
    
    numbers = re.findall(r'(\d+)번?', question)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    
    return None, None


def _is_new_search_request(question):
    """질문이 새로운 검색 요청인지 판단 (날짜/지역 정보 포함 여부)"""
    import re
    
    date_patterns = [r'\d{4}년', r'\d{1,2}월', r'\d{1,2}일', r'\d{4}[-/]\d{1,2}[-/]\d{1,2}']
    has_date = any(re.search(pattern, question) for pattern in date_patterns)
    location_keywords = ["지역", "지진", "위치", "어디", "where", "location", "데이터 가져", "데이터 받", "다운로드"]
    has_location = any(keyword in question.lower() for keyword in location_keywords)
    return has_date or has_location


def _get_date_range(state):
    """state에서 검색/다운로드용 날짜 범위 반환 (start_date, end_date)"""
    from datetime import datetime, timedelta
    date_range = state.get("date_range", {})
    event_date = date_range.get("event_date")
    if event_date:
        try:
            event_dt = datetime.strptime(event_date, "%Y-%m-%d")
            return (event_dt - timedelta(days=730)).strftime("%Y-%m-%d"), (event_dt + timedelta(days=730)).strftime("%Y-%m-%d")
        except Exception:
            pass
    return date_range.get("start_date", "2022-01-01"), date_range.get("end_date", "2024-12-31")


def _validate_indices(products, indices, max_idx_key='display_index'):
    """인덱스 범위 검증."""
    valid_indices = [p.get(max_idx_key) for p in products if max_idx_key in p]
    max_idx = max(valid_indices) if valid_indices else len(products) - 1
    idx_list = indices if isinstance(indices, list) else [indices]
    invalid = [i for i in idx_list if i > max_idx]
    return (len(invalid) == 0, max_idx, invalid)


def _parse_single_selection(question):
    """사용자 응답에서 인덱스 추출 (일반 SAR용) - 단일 또는 다중 선택 지원"""
    import re
    
    cleaned = question
    cleaned = re.sub(r'\d{4}[-/년]\d{1,2}[-/월]\d{1,2}일?', '', cleaned)
    cleaned = re.sub(r'\d{4}년', '', cleaned)
    cleaned = re.sub(r'\d{1,2}월', '', cleaned)
    cleaned = re.sub(r'\d{1,2}일', '', cleaned)
    number_with_marker = re.findall(r'(\d+)번', cleaned)
    if number_with_marker:
        indices = sorted(list(set([int(n) for n in number_with_marker])))
        return indices
    
    numbers = re.findall(r'(\d+)', cleaned)
    if numbers:
        indices = sorted(list(set([int(n) for n in numbers if not (1900 <= int(n) <= 2100)])))
        indices = [idx for idx in indices if idx <= 100]
        return indices if indices else None
    
    return None


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
            if _is_new_search_request(question):
                awaiting_selection = False
            else:
                master_idx, slave_idx = _parse_master_slave_selection(question)
                
                if master_idx is None or slave_idx is None:
                    msg = "❌ Master/Slave 인덱스를 파싱할 수 없습니다. 형식: 'Master 1, Slave 5' 또는 '1번과 5번'"
                    return {
                        "generation": msg,
                        "messages": [AIMessage(content=msg)],
                        "awaiting_master_slave_selection": True,
                    }
                products = sar_search_results.get('products', [])
                if products and 'file_path' in products[0]:
                    ok, max_idx, _ = _validate_indices(products, [master_idx, slave_idx])
                    if not ok:
                        msg = f"❌ 인덱스 범위 초과 (Master: {master_idx}, Slave: {slave_idx}, 최대: {max_idx})"
                        return {
                            "generation": msg,
                            "messages": [AIMessage(content=msg)],
                            "awaiting_master_slave_selection": True,
                        }
                    master_file = products[master_idx]['file_path']
                    slave_file = products[slave_idx]['file_path']
                    return {
                        "generation": f"✅ Master와 Slave를 선택했습니다. InSAR 처리를 시작합니다...",
                        "messages": [AIMessage(content="✅ Master와 Slave를 선택했습니다. InSAR 처리를 시작합니다...")],
                        "downloaded_sar_files": [master_file, slave_file],  # 순서 중요!
                        "awaiting_master_slave_selection": False,
                        "needs_insar": True,  # run_insar로 라우팅
                    }
                else:
                    # ASF 다운로드 필요
                    return _execute_download_insar(
                        state,
                        sar_search_results,
                        master_idx,
                        slave_idx
                    )
    
    else:
        awaiting_selection = state.get("awaiting_single_sar_selection", False)
        sar_search_results = state.get("sar_search_results")
        
        if awaiting_selection and sar_search_results:
            if _is_new_search_request(question):
                awaiting_selection = False
            else:
                selected_indices = _parse_single_selection(question)
                
                if selected_indices is None or len(selected_indices) == 0:
                    msg = "❌ 인덱스를 파싱할 수 없습니다. 형식: '1번' 또는 '1,2,3'"
                    return {
                        "generation": msg,
                        "messages": [AIMessage(content=msg)],
                        "awaiting_single_sar_selection": True,
                    }
                return _execute_download_single(
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
        location = _extract_location_from_question(question, llm)
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
        event_date = _extract_event_date(question, llm)
    if not event_date:
        summary = state.get("summary", "")
        if summary:
            event_date = _extract_event_date(summary, llm)
    
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
        if event_date:
            from datetime import datetime
            try:
                event_dt = datetime.strptime(event_date, "%Y-%m-%d")
                
                before_products = []
                after_products = []
                
                for product in products:
                    product_date_str = product['date']  # YYYYMMDD 형식
                    product_dt = datetime.strptime(product_date_str, "%Y%m%d")
                    time_diff_days = (product_dt - event_dt).days  # 부호 있는 차이
                    
                    product['time_diff_days'] = time_diff_days
                    product['product_dt'] = product_dt
                    
                    if time_diff_days < 0:  # 발생 이전
                        before_products.append(product)
                    else:  # 발생 이후 (동일 날짜 포함)
                        after_products.append(product)
                
                before_products.sort(key=lambda x: x['product_dt'], reverse=True)
                after_products.sort(key=lambda x: x['product_dt'])
                before_top = before_products[:5]
                after_top = after_products[:5]
                
                filtered_products = before_top + after_top
                for i, product in enumerate(filtered_products):
                    product['original_index'] = product['index']
                    product['display_index'] = i
                
                products = filtered_products
                display_limit = len(products)
            except Exception:
                display_limit = min(10, total)
                products = products[:display_limit]
                for i, product in enumerate(products):
                    product['original_index'] = product['index']
                    product['display_index'] = i
        else:
            display_limit = min(10, total)
            products = products[:display_limit]
            for i, product in enumerate(products):
                product['original_index'] = product['index']
                product['display_index'] = i
        
        actual_date_range = search_result.get('date_range', 'N/A')
        event_info = ""
        if event_date:
            from datetime import datetime
            try:
                event_dt = datetime.strptime(event_date, "%Y-%m-%d")
                before_count = sum(1 for p in products if 'time_diff_days' in p and p['time_diff_days'] < 0)
                after_count = sum(1 for p in products if 'time_diff_days' in p and p['time_diff_days'] >= 0)
                
                if before_count == 0:
                    event_info = f"\n⚠️ **이벤트 날짜({event_date}) 이전 데이터가 없습니다!** (발생 전 0개, 발생 후 {after_count}개)"
                elif after_count == 0:
                    event_info = f"\n⚠️ **이벤트 날짜({event_date}) 이후 데이터가 없습니다!** (발생 전 {before_count}개, 발생 후 0개)"
                else:
                    event_info = f"\n🎯 이벤트 날짜({event_date}) 기준 전/후 각 5개씩 (총 {display_limit}개) 표시 (발생 직전/직후 우선)"
            except:
                pass
        
        generation = f"""✅ **{location}**에서 **{total}개의 SAR 데이터**를 찾았습니다!

📅 **요청한 검색 범위**: {start_date} ~ {end_date}
📊 **실제 데이터 날짜 범위**: {actual_date_range}
📍 좌표: ({lat}, {lon}){event_info}

📊 **데이터 리스트** (상위 {display_limit}개):

"""
        
        date_groups = {}
        for product in products:
            date = product['date']
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append(product)
        
        sorted_dates = sorted(date_groups.keys())
        
        for date in sorted_dates:
            formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            timing_label = ""
            if event_date and date_groups[date]:
                first_product = date_groups[date][0]
                if 'time_diff_days' in first_product:
                    diff = first_product['time_diff_days']
                    if diff < 0:
                        timing_label = f" (📌 발생 {abs(diff)}일 전)"
                    elif diff > 0:
                        timing_label = f" (📌 발생 {diff}일 후)"
                    else:
                        timing_label = f" (📌 발생 당일)"
            
            for product in date_groups[date]:
                idx = product.get('display_index', product['index'])
                filename = product['filename']
                size_mb = product['size_mb']
                generation += f"[{idx}] {formatted_date}{timing_label}\n    {filename[:50]}... ({size_mb} MB)\n"
        
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
                master_idx, slave_idx = _auto_select_for_insar(products, event_date)
                
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
                
                download_result = _execute_download_insar(
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
            
            generation += f"""

🎯 **Master와 Slave를 선택해주세요 (InSAR용):**

다음 형식으로 입력:
- "Master 1, Slave 5"
- "1번과 5번"

💡 **선택 팁 (InSAR 지표변형 분석):**
- **Master**: 이벤트 **이전** 날짜 (기준 이미지, 변화 전)
- **Slave**: 이벤트 **이후** 날짜 (비교 이미지, 변화 후)
- 발생 시점에 **가장 가까운 전/후 데이터**를 선택하세요!
"""
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
            generation += f"""

🎯 **데이터를 선택해주세요:**

다음 형식으로 입력:
- 단일 선택: "1번" 또는 "5"
- 다중 선택: "1,2,3" 또는 "1 2 3"

💡 여러 개를 선택하면 모두 다운로드됩니다!
"""
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
