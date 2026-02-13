"""SAR 검색 결과 포맷팅 함수들"""
from datetime import datetime


def filter_and_group_by_event(products, event_date, display_limit=10):
    """
    이벤트 날짜 기준으로 제품 필터링 및 그룹화
    전/후 각 5개씩 선택하여 반환
    """
    if not event_date:
        # 이벤트 날짜 없으면 상위 N개만
        display_limit = min(display_limit, len(products))
        filtered = products[:display_limit]
        for i, product in enumerate(filtered):
            product['original_index'] = product['index']
            product['display_index'] = i
        return filtered, None
    
    try:
        event_dt = datetime.strptime(event_date, "%Y-%m-%d")
        
        before_products = []
        after_products = []
        
        for product in products:
            product_date_str = product['date']  # YYYYMMDD
            product_dt = datetime.strptime(product_date_str, "%Y%m%d")
            time_diff_days = (product_dt - event_dt).days
            
            product['time_diff_days'] = time_diff_days
            product['product_dt'] = product_dt
            
            if time_diff_days < 0:
                before_products.append(product)
            else:
                after_products.append(product)
        
        # 발생 전/후 각 5개
        before_products.sort(key=lambda x: x['product_dt'], reverse=True)
        after_products.sort(key=lambda x: x['product_dt'])
        before_top = before_products[:5]
        after_top = after_products[:5]
        
        filtered = before_top + after_top
        for i, product in enumerate(filtered):
            product['original_index'] = product['index']
            product['display_index'] = i
        
        # 통계 정보
        before_count = len(before_top)
        after_count = len(after_top)
        event_info = {
            'before_count': before_count,
            'after_count': after_count,
            'warning': None
        }
        
        if before_count == 0:
            event_info['warning'] = f"⚠️ **이벤트 날짜({event_date}) 이전 데이터가 없습니다!** (발생 전 0개, 발생 후 {after_count}개)"
        elif after_count == 0:
            event_info['warning'] = f"⚠️ **이벤트 날짜({event_date}) 이후 데이터가 없습니다!** (발생 전 {before_count}개, 발생 후 0개)"
        else:
            event_info['info'] = f"🎯 이벤트 날짜({event_date}) 기준 전/후 각 5개씩 (총 {len(filtered)}개) 표시 (발생 직전/직후 우선)"
        
        return filtered, event_info
        
    except Exception as e:
        print(f"Event filtering error: {e}")
        display_limit = min(display_limit, len(products))
        filtered = products[:display_limit]
        for i, product in enumerate(filtered):
            product['original_index'] = product['index']
            product['display_index'] = i
        return filtered, None


def format_search_results_header(location, total, start_date, end_date, actual_date_range, lat, lon, event_info):
    """검색 결과 헤더 포맷팅"""
    header = f"""✅ **{location}**에서 **{total}개의 SAR 데이터**를 찾았습니다!

📅 **요청한 검색 범위**: {start_date} ~ {end_date}
📊 **실제 데이터 날짜 범위**: {actual_date_range}
📍 좌표: ({lat}, {lon})"""
    
    if event_info:
        if event_info.get('warning'):
            header += f"\n{event_info['warning']}"
        elif event_info.get('info'):
            header += f"\n{event_info['info']}"
    
    return header


def format_products_by_orbit(products, needs_insar=False):
    """
    제품 리스트를 Orbit별로 그룹화하여 포맷팅
    
    Returns:
        str: 포맷팅된 제품 리스트
    """
    result = ""
    
    # InSAR용 안내 메시지
    if needs_insar:
        result += """
⚠️ **InSAR 처리 안내**:
- InSAR를 위해서는 **같은 Orbit 번호**의 데이터를 선택해야 합니다
- 다른 Orbit을 선택하면 영상 정합(co-registration)이 실패합니다
- 아래에서 같은 Orbit 그룹 내의 데이터를 확인하세요

"""
    
    result += f"\n📊 **데이터 리스트** (Orbit별 그룹, 총 {len(products)}개):\n\n```"
    
    # Orbit별로 그룹화
    orbit_groups = {}
    for product in products:
        orbit = product.get('relative_orbit', product.get('path_number', 'N/A'))
        flight_dir = product.get('flight_direction', 'N/A')
        orbit_key = f"{orbit}_{flight_dir}"
        
        if orbit_key not in orbit_groups:
            orbit_groups[orbit_key] = []
        orbit_groups[orbit_key].append(product)
    
    # Orbit 그룹별로 표시
    for orbit_key in sorted(orbit_groups.keys()):
        orbit, flight_dir = orbit_key.split('_')
        flight_icon = "🔼" if flight_dir == "ASCENDING" else "🔽" if flight_dir == "DESCENDING" else "🛰️"
        
        orbit_products = orbit_groups[orbit_key]
        result += f"\n{flight_icon} **Orbit {orbit} ({flight_dir})** - {len(orbit_products)}개\n"
        result += f"{'─' * 80}\n"
        
        # 날짜별로 정렬
        orbit_products_sorted = sorted(orbit_products, key=lambda x: x['date'])
        
        for product in orbit_products_sorted:
            idx = product.get('display_index', product['index'])
            filename = product['filename']
            size_mb = product['size_mb']
            date = product['date']
            formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            
            # 타이밍 라벨
            timing_label = ""
            timing_icon = ""
            if 'time_diff_days' in product:
                diff = product['time_diff_days']
                if diff < 0:
                    timing_label = f"발생 {abs(diff)}일 전"
                    timing_icon = "⏪"
                elif diff > 0:
                    timing_label = f"발생 {diff}일 후"
                    timing_icon = "⏩"
                else:
                    timing_label = "발생 당일"
                    timing_icon = "⚡"
            
            # 크기 포맷 (GB 단위로 변환)
            if size_mb >= 1000:
                size_str = f"{size_mb/1000:.1f}GB"
            else:
                size_str = f"{size_mb:.0f}MB"
            
            # 파일명에서 날짜/시간 추출 (더 간결하게)
            # S1A_IW_SLC__1SDV_20230204T152607_... → S1A 20230204 15:26
            parts = filename.split('_')
            satellite = parts[0]  # S1A or S1B
            date_time = parts[4] if len(parts) > 4 else ""  # 20230204T152607
            time_str = ""
            if 'T' in date_time:
                time_part = date_time.split('T')[1][:4]  # 1526
                time_str = f"{time_part[:2]}:{time_part[2:]}"
            
            # 한 줄로 깔끔하게 표시
            if timing_label:
                result += f"  [{idx:2d}] 📅 {formatted_date} {time_str} | {timing_icon} {timing_label:15s} | 💾 {size_str:7s} | {satellite}\n"
            else:
                result += f"  [{idx:2d}] 📅 {formatted_date} {time_str} | 💾 {size_str:7s} | {satellite}\n"
        
        result += "\n"
    
    result += "```\n"
    return result


def build_insar_selection_message():
    """InSAR Master/Slave 선택 요청 메시지"""
    return """

🎯 **Master와 Slave를 선택해주세요 (InSAR용):**

다음 형식으로 입력:
- "Master 2, Slave 9"
- "2번과 9번"

⚠️ **중요! 같은 Orbit 번호를 선택해야 합니다:**
- 위 목록에서 같은 **Orbit** 그룹 내의 데이터를 선택하세요
- 다른 Orbit을 선택하면 InSAR 처리가 실패합니다

💡 **선택 팁 (InSAR 지표변형 분석):**
- **Master**: 이벤트 **이전** 날짜 (기준 이미지, 변화 전)
- **Slave**: 이벤트 **이후** 날짜 (비교 이미지, 변화 후)
- 발생 시점에 **가장 가까운 전/후 데이터**를 선택하세요!
- 🔼 (Ascending) 또는 🔽 (Descending)도 같아야 합니다
"""


def build_single_selection_message():
    """일반 SAR 데이터 선택 요청 메시지"""
    return """

🎯 **데이터를 선택해주세요:**

다음 형식으로 입력:
- "1번" (단일 선택)
- "1, 2, 3" (다중 선택)
- "1번과 2번과 3번" (다중 선택)

💡 **여러 개를 선택하면 모두 다운로드됩니다!**
"""
