"""
ASF (Alaska Satellite Facility)에서 Sentinel-1 SAR 데이터 다운로드 유틸리티
"""
import os
import time
import json
from pathlib import Path

try:
    import asf_search as asf
    ASF_AVAILABLE = True
except ImportError:
    ASF_AVAILABLE = False
    print("Warning: asf_search not installed. Install with: pip install asf_search")

# 환경변수에서 SAR 저장 경로 가져오기
def get_default_sar_path():
    """환경변수에서 기본 SAR 저장 경로를 가져옴"""
    paths_str = os.getenv("SAR_DATA_PATHS", "/mnt/sar,/home/mjh/sar_data,/data/sar")
    paths = [p.strip() for p in paths_str.split(",") if p.strip()]
    default_path = paths[0] if paths else "/mnt/sar"
    return f"{default_path}/S1A/Dataset"


class SARDownloader:
    """Sentinel-1 SAR 데이터 다운로드 클래스"""
    
    def __init__(self, save_path=None):
        """
        Args:
            save_path: 다운로드 저장 경로 (기본값: 환경변수 SAR_DATA_PATHS의 첫 번째 경로)
        """
        self.save_path = save_path or get_default_sar_path()
        
        # Earthdata 인증 정보
        self.username = os.environ.get('EARTHDATA_USERNAME', 'jeongho.min')
        self.password = os.environ.get('EARTHDATA_PASSWORD', '@Fnalfm80680866')
        
        # 환경변수 설정
        os.environ['EARTHDATA_USERNAME'] = self.username
        os.environ['EARTHDATA_PASSWORD'] = self.password
        
        self.session = None
        if ASF_AVAILABLE:
            self._init_session()
    
    def _init_session(self):
        """ASF 세션 초기화 및 인증"""
        try:
            self.session = asf.ASFSession()
            self.session.auth_with_creds(self.username, self.password)
            print("✓ ASF Authentication successful!")
            return True
        except Exception as e:
            print(f"⚠ Authentication warning: {str(e)}")
            print("Will use environment variables for authentication...")
            self.session = asf.ASFSession()
            return False
    
    def search_sentinel1(
        self,
        latitude,
        longitude,
        start_date="2024-01-01",
        end_date="2024-12-31",
        buffer=0.5,
        max_results=10,
        relative_orbit=None,
        flight_direction=None
    ):
        """
        Sentinel-1 데이터 검색
        
        Args:
            latitude: 위도
            longitude: 경도
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            buffer: 좌표 주변 버퍼 (도 단위, 기본값 0.5도)
            max_results: 최대 검색 결과 수
            relative_orbit: 상대 궤도 번호 (옵션)
            flight_direction: 비행 방향 (ASCENDING/DESCENDING, 옵션)
        
        Returns:
            ASFSearchResults or None
        """
        if not ASF_AVAILABLE:
            raise ImportError("asf_search module not available")
        
        # WKT 폴리곤 생성 (좌표 주변 박스)
        wkt_polygon = (
            f"POLYGON(("
            f"{longitude-buffer} {latitude-buffer}, "
            f"{longitude+buffer} {latitude-buffer}, "
            f"{longitude+buffer} {latitude+buffer}, "
            f"{longitude-buffer} {latitude+buffer}, "
            f"{longitude-buffer} {latitude-buffer}"
            f"))"
        )
        
        print(f"Searching Sentinel-1 products...")
        print(f"Location: ({latitude}, {longitude})")
        print(f"Date range: {start_date} to {end_date}")
        
        # 검색 파라미터 구성
        search_params = {
            'platform': asf.PLATFORM.SENTINEL1,
            'processingLevel': asf.PRODUCT_TYPE.SLC,
            'start': f'{start_date}T00:00:00Z',
            'end': f'{end_date}T23:59:59Z',
            'intersectsWith': wkt_polygon,
            # maxResults를 크게 설정하여 오래된 데이터도 포함
            'maxResults': max(max_results * 10, 500)  # 최소 500개
        }
        
        # 옵션 파라미터 추가
        if relative_orbit is not None:
            search_params['relativeOrbit'] = relative_orbit
        
        if flight_direction is not None:
            if flight_direction.upper() == 'ASCENDING':
                search_params['flightDirection'] = asf.FLIGHT_DIRECTION.ASCENDING
            elif flight_direction.upper() == 'DESCENDING':
                search_params['flightDirection'] = asf.FLIGHT_DIRECTION.DESCENDING
        
        # 검색 실행 (재시도 로직 포함)
        max_retries = 3
        retry_count = 0
        results = None
        
        while retry_count < max_retries:
            try:
                results = asf.search(**search_params)
                break
            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}")
                if retry_count < max_retries:
                    print(f"Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print("Max retries exceeded.")
                    raise
        
        if not results or len(results) == 0:
            print("No products found matching the search criteria.")
            return None
        
        print(f"Total Number of Searched Products: {len(results)}")
        return results
    
    def select_insar_pair(self, results):
        """
        InSAR 처리를 위한 Master/Slave 쌍 선택
        ⭐ 같은 궤도 번호(relative orbit) & 비행 방향(flight direction)끼리만 선택
        
        Args:
            results: ASF 검색 결과
        
        Returns:
            list: 선택된 제품 리스트 [master, slave]
        """
        if not results or len(results) == 0:
            return []
        
        # ⭐ 궤도별로 그룹화 (같은 궤도 & 비행 방향)
        orbit_groups = {}
        for product in results:
            props = product.properties
            filename = props['fileName']
            
            # 궤도 정보 추출
            relative_orbit = props.get('pathNumber') or props.get('relativeOrbit', 'unknown')
            flight_direction = props.get('flightDirection', 'unknown')
            
            # 날짜 추출
            date_str = filename.split('_')[5][:8]
            
            # 궤도+방향을 키로 사용
            orbit_key = f"{relative_orbit}_{flight_direction}"
            
            if orbit_key not in orbit_groups:
                orbit_groups[orbit_key] = {}
            
            if date_str not in orbit_groups[orbit_key]:
                orbit_groups[orbit_key][date_str] = []
            
            orbit_groups[orbit_key][date_str].append(product)
        
        print(f"Found {len(orbit_groups)} orbit groups:")
        for orbit_key, dates in orbit_groups.items():
            print(f"  - Orbit {orbit_key}: {len(dates)} unique dates")
        
        # 가장 많은 날짜를 가진 궤도 선택
        best_orbit_key = max(orbit_groups.keys(), key=lambda k: len(orbit_groups[k]))
        best_orbit = orbit_groups[best_orbit_key]
        
        print(f"✅ Selected orbit: {best_orbit_key} ({len(best_orbit)} dates)")
        
        # 날짜별로 정렬 (최신순)
        sorted_dates = sorted(best_orbit.keys(), reverse=True)
        
        # 각 날짜에서 첫 번째 제품만 선택
        unique_products = []
        for date_str in sorted_dates:
            unique_products.append(best_orbit[date_str][0])
        
        # Master와 Slave 선택 (가장 최근 두 개)
        if len(unique_products) >= 2:
            master = unique_products[0]
            slave = unique_products[1]
            
            master_date = master.properties['fileName'].split('_')[5][:8]
            slave_date = slave.properties['fileName'].split('_')[5][:8]
            
            master_orbit = master.properties.get('pathNumber') or master.properties.get('relativeOrbit', 'unknown')
            slave_orbit = slave.properties.get('pathNumber') or slave.properties.get('relativeOrbit', 'unknown')
            
            print(f"Master (most recent): {master.properties['fileName']}")
            print(f"  - Date: {master_date}, Orbit: {master_orbit}")
            print(f"Slave (second recent): {slave.properties['fileName']}")
            print(f"  - Date: {slave_date}, Orbit: {slave_orbit}")
            
            return [master, slave]
        elif len(unique_products) == 1:
            print(f"Warning: Only 1 product found. Need at least 2 products for InSAR.")
            return unique_products
        else:
            return []
    
    def download_products(self, products):
        """
        제품 다운로드
        
        Args:
            products: 다운로드할 제품 리스트
        
        Returns:
            dict: 다운로드 결과 정보
        """
        if not products or len(products) == 0:
            return {
                'success': False,
                'message': 'No products to download',
                'downloaded': 0,
                'skipped': 0,
                'failed': 0,
                'files': []
            }
        
        # 저장 경로 생성
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            print(f"Created save directory: {self.save_path}")
        
        download_count = 0
        skip_count = 0
        failed_count = 0
        downloaded_files = []
        failed_files = []
        
        print(f"\nDownloading {len(products)} products...")
        
        for idx, product in enumerate(products, 1):
            filename = product.properties['fileName']
            filepath = os.path.join(self.save_path, filename)
            
            # 이미 파일이 존재하는지 확인
            if os.path.exists(filepath):
                print(f"[{idx}/{len(products)}] Already exists: {filename}")
                skip_count += 1
                downloaded_files.append(filename)
                continue
            
            print(f"[{idx}/{len(products)}] Downloading: {filename}")
            
            try:
                # 다운로드 시도 (여러 방법)
                try:
                    product.download(path=self.save_path)
                except Exception as e1:
                    try:
                        asf.download([product], path=self.save_path)
                    except Exception as e2:
                        try:
                            product.download(path=self.save_path, session=self.session)
                        except TypeError:
                            raise e2
                
                download_count += 1
                downloaded_files.append(filename)
                print(f"  ✓ Download completed")
                
            except Exception as e:
                failed_count += 1
                failed_files.append(filename)
                print(f"  ✗ Download failed: {str(e)}")
        
        return {
            'success': download_count > 0 or skip_count > 0,
            'message': 'Download completed',
            'total': len(products),
            'downloaded': download_count,
            'skipped': skip_count,
            'failed': failed_count,
            'files': downloaded_files,
            'failed_files': failed_files,
            'save_path': self.save_path
        }
    
    def download_by_location(
        self,
        latitude,
        longitude,
        location_name=None,
        start_date="2024-01-01",
        end_date="2024-12-31",
        buffer=0.5,
        max_results=10,
        select_insar_pair=True
    ):
        """
        좌표 기반 Sentinel-1 데이터 검색 및 다운로드 (통합 함수)
        
        Args:
            latitude: 위도
            longitude: 경도
            location_name: 지역명 (옵션)
            start_date: 시작 날짜
            end_date: 종료 날짜
            buffer: 좌표 주변 버퍼
            max_results: 최대 검색 결과 수
            select_insar_pair: InSAR 쌍 선택 여부 (True: Master/Slave만, False: 모두)
        
        Returns:
            dict: 검색 및 다운로드 결과
        """
        if not ASF_AVAILABLE:
            return {
                'success': False,
                'error': 'asf_search module not installed',
                'message': 'Please install asf_search: pip install asf_search'
            }
        
        try:
            # 검색
            results = self.search_sentinel1(
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
                buffer=buffer,
                max_results=max_results
            )
            
            if not results:
                return {
                    'success': False,
                    'error': 'no_data',
                    'message': f'No Sentinel-1 data found for the specified location and date range',
                    'location': location_name or f"({latitude}, {longitude})",
                    'date_range': f"{start_date} ~ {end_date}"
                }
            
            # 제품 선택
            if select_insar_pair:
                selected_products = self.select_insar_pair(results)
                product_type = "InSAR Pair (Master + Slave)"
            else:
                selected_products = list(results)[:max_results]
                product_type = "All Products"
            
            if not selected_products:
                return {
                    'success': False,
                    'error': 'no_products',
                    'message': 'No suitable products found after selection'
                }
            
            # 다운로드
            download_result = self.download_products(selected_products)
            
            # 결과 통합
            return {
                'success': download_result['success'],
                'location': location_name or f"({latitude}, {longitude})",
                'coordinates': {'latitude': latitude, 'longitude': longitude},
                'date_range': f"{start_date} ~ {end_date}",
                'product_type': product_type,
                'search_results': len(results),
                'selected_products': len(selected_products),
                'download_result': download_result
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': 'exception',
                'message': str(e),
                'traceback': traceback.format_exc()
            }


def download_sentinel1_by_geojson(geojson_path, **kwargs):
    """
    GeoJSON 파일로부터 영역 정보를 읽어 Sentinel-1 다운로드
    
    Args:
        geojson_path: GeoJSON 파일 경로
        **kwargs: SARDownloader.search_sentinel1()에 전달할 추가 파라미터
    
    Returns:
        dict: 다운로드 결과
    """
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    # GeoJSON에서 폴리곤 좌표 추출
    coordinates = geojson_data['features'][0]['geometry']['coordinates'][0]
    
    # WKT 형식으로 변환
    wkt_polygon = "POLYGON((" + ", ".join([f"{lon} {lat}" for lon, lat in coordinates]) + "))"
    
    # TODO: GeoJSON 기반 검색 구현 (현재는 좌표 기반만 지원)
    raise NotImplementedError("GeoJSON-based download not yet implemented")
