"""SAR 다운로드 헬퍼 함수들"""
import re
from datetime import datetime, timedelta


def extract_event_date(question, llm):
    """질문에서 이벤트 발생 날짜 추출"""
    prompt = f"""질문에서 지진/화산 등 이벤트 발생 날짜를 추출하세요:
질문: {question}

출력 형식: YYYY-MM-DD (날짜가 없으면 '없음')
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


def extract_location_from_question(question, llm):
    """질문에서 지역명 추출"""
    prompt = f"질문에서 지역명 추출: {question}\n지역명만 출력 (없으면 '없음'):"
    try:
        response = llm.invoke(prompt)
        location = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        if location and location.lower() not in ["없음", "none", "no", ""]:
            return location
    except:
        pass
    return None


def auto_select_for_insar(products, event_date):
    """InSAR용 자동 2개 선택 (이벤트 날짜 기준 전후, 같은 Orbit 내에서만)"""
    if not products or len(products) < 2:
        return None, None
    
    try:
        if '-' in event_date:
            event_dt = datetime.strptime(event_date, '%Y-%m-%d')
        else:
            event_dt = datetime.strptime(event_date, '%Y%m%d')
    except:
        return products[0].get('display_index', 0), products[1].get('display_index', 1)
    
    # Orbit별로 그룹화
    orbit_groups = {}
    for p in products:
        orbit = p.get('relative_orbit', p.get('path_number', 'N/A'))
        flight_dir = p.get('flight_direction', 'N/A')
        orbit_key = f"{orbit}_{flight_dir}"
        
        if orbit_key not in orbit_groups:
            orbit_groups[orbit_key] = []
        orbit_groups[orbit_key].append(p)
    
    # 가장 많은 데이터가 있는 Orbit 그룹 선택
    best_orbit_key = max(orbit_groups.keys(), key=lambda k: len(orbit_groups[k]))
    best_orbit_products = orbit_groups[best_orbit_key]
    
    print(f"  ✓ InSAR 자동 선택: Orbit {best_orbit_key} ({len(best_orbit_products)}개 중 선택)")
    
    # 해당 Orbit 내에서 이벤트 전후 선택
    products_with_distance = []
    for p in best_orbit_products:
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
        if len(best_orbit_products) >= 2:
            return (best_orbit_products[0].get('display_index'), 
                    best_orbit_products[1].get('display_index'))
        return None, None
    
    # 이벤트 전후로 가장 가까운 것 선택
    products_with_distance.sort(key=lambda x: x['abs_days_diff'])
    before_products = [p for p in products_with_distance if p['days_diff'] < 0]
    after_products = [p for p in products_with_distance if p['days_diff'] >= 0]
    
    if before_products and after_products:
        master_idx = before_products[0]['product'].get('display_index')
        slave_idx = after_products[0]['product'].get('display_index')
    else:
        master_idx = products_with_distance[0]['product'].get('display_index')
        slave_idx = products_with_distance[1]['product'].get('display_index')
    
    return master_idx, slave_idx


def parse_master_slave_selection(question):
    """사용자 응답에서 Master/Slave 인덱스 추출 (InSAR용)"""
    master_match = re.search(r'[Mm]aster[\s:]*(\d+)', question)
    slave_match = re.search(r'[Ss]lave[\s:]*(\d+)', question)
    
    if master_match and slave_match:
        return int(master_match.group(1)), int(slave_match.group(1))
    
    numbers = re.findall(r'(\d+)번?', question)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    
    return None, None


def is_new_search_request(question):
    """질문이 새로운 검색 요청인지 판단"""
    date_patterns = [r'\d{4}년', r'\d{1,2}월', r'\d{1,2}일', r'\d{4}[-/]\d{1,2}[-/]\d{1,2}']
    has_date = any(re.search(pattern, question) for pattern in date_patterns)
    location_keywords = ["지역", "지진", "위치", "어디", "where", "location", "데이터 가져", "데이터 받", "다운로드"]
    has_location = any(keyword in question.lower() for keyword in location_keywords)
    return has_date or has_location


def get_date_range(state):
    """state에서 검색/다운로드용 날짜 범위 반환 (start_date, end_date)"""
    date_range = state.get("date_range", {})
    event_date = date_range.get("event_date")
    if event_date:
        try:
            event_dt = datetime.strptime(event_date, "%Y-%m-%d")
            return (event_dt - timedelta(days=730)).strftime("%Y-%m-%d"), (event_dt + timedelta(days=730)).strftime("%Y-%m-%d")
        except Exception:
            pass
    return date_range.get("start_date", "2022-01-01"), date_range.get("end_date", "2024-12-31")


def validate_indices(products, indices, max_idx_key='display_index'):
    """인덱스 범위 검증"""
    valid_indices = [p.get(max_idx_key) for p in products if max_idx_key in p]
    max_idx = max(valid_indices) if valid_indices else len(products) - 1
    idx_list = indices if isinstance(indices, list) else [indices]
    invalid = [i for i in idx_list if i > max_idx]
    return (len(invalid) == 0, max_idx, invalid)


def parse_single_selection(question, llm):
    """
    ⭐ LLM을 사용하여 사용자 응답에서 인덱스 추출 (일반 SAR용)
    단일 또는 다중 선택 지원
    """
    prompt = f"""사용자가 SAR 데이터 리스트에서 다운로드할 번호를 선택했습니다.

사용자 입력: "{question}"

사용자가 선택한 모든 번호를 추출하세요.

예시:
- "1, 5번" → 1, 5
- "1번과 5번" → 1, 5  
- "3" → 3
- "2번 4번 7번" → 2, 4, 7
- "첫 번째와 다섯 번째" → 1, 5
- "1, 5번 다운" → 1, 5

출력 형식: 쉼표로 구분된 숫자만 출력 (예: 1, 5, 7)
선택한 번호가 없으면 "없음"이라고 출력하세요.
"""
    
    try:
        print(f"[PARSE_SINGLE DEBUG] LLM 호출 시작...")
        response = llm.invoke(prompt)
        print(f"[PARSE_SINGLE DEBUG] LLM 응답 받음!")
        text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        print(f"[PARSE_SINGLE DEBUG] 추출된 텍스트: {text}")
        
        if "없음" in text.lower() or "none" in text.lower():
            return None
        
        # 숫자 추출
        numbers = re.findall(r'\d+', text)
        if numbers:
            indices = sorted(list(set([int(n) for n in numbers if int(n) <= 100])))
            return indices if indices else None
        
        return None
    except Exception as e:
        print(f"LLM 파싱 실패: {e}")
        # Fallback: regex 기반
        numbers = re.findall(r'(\d+)', question)
        if numbers:
            indices = sorted(list(set([int(n) for n in numbers if not (1900 <= int(n) <= 2100) and int(n) <= 100])))
            return indices if indices else None
        return None
