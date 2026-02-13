"""
SAR 데이터 다운로드 전용 FastAPI 서버
"""
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# Import 처리 (직접 실행 vs 모듈 실행)
try:
    from .sar_download_utils import SARDownloader
except ImportError:
    from sar_download_utils import SARDownloader

app = FastAPI(
    title="SAR Download API",
    description="Sentinel-1 SAR 데이터 다운로드 API 서버",
    version="1.0.0"
)

# 전역 다운로더 인스턴스
downloader = SARDownloader()

# 다운로드 작업 상태 저장
download_jobs = {}


class SearchRequest(BaseModel):
    """SAR 검색 요청 모델 (다운로드 없이 리스트만 조회)"""
    latitude: float = Field(..., description="위도", example=36.0)
    longitude: float = Field(..., description="경도", example=140.0)
    location_name: Optional[str] = Field(None, description="지역명", example="이바라키")
    start_date: str = Field(..., description="검색 시작 날짜 (YYYY-MM-DD) - 이벤트 발생일 기준 ±1년", example="2022-02-06")
    end_date: str = Field(..., description="검색 종료 날짜 (YYYY-MM-DD) - 이벤트 발생일 기준 ±1년", example="2024-02-06")
    buffer: float = Field(0.5, description="좌표 주변 버퍼 (도 단위)", example=0.5)
    max_results: int = Field(50, description="최대 검색 결과 수", example=50)


class DownloadRequest(BaseModel):
    """SAR 다운로드 요청 모델"""
    latitude: float = Field(..., description="위도", example=36.0)
    longitude: float = Field(..., description="경도", example=140.0)
    location_name: Optional[str] = Field(None, description="지역명", example="이바라키")
    start_date: str = Field(..., description="검색 시작 날짜 (YYYY-MM-DD)", example="2022-02-06")
    end_date: str = Field(..., description="검색 종료 날짜 (YYYY-MM-DD)", example="2024-02-06")
    buffer: float = Field(0.5, description="좌표 주변 버퍼 (도 단위)", example=0.5)
    max_results: int = Field(10, description="최대 검색 결과 수", example=10)
    master_index: Optional[int] = Field(None, description="Master로 선택할 데이터 인덱스 (검색 결과에서)")
    slave_index: Optional[int] = Field(None, description="Slave로 선택할 데이터 인덱스 (검색 결과에서)")
    selected_indices: Optional[list] = Field(None, description="단일/다중 선택할 데이터 인덱스 리스트 (검색 결과에서)")


class DownloadResponse(BaseModel):
    """SAR 다운로드 응답 모델"""
    success: bool
    job_id: Optional[str] = None
    message: str
    location: Optional[str] = None
    coordinates: Optional[dict] = None
    date_range: Optional[str] = None
    search_results: Optional[int] = None
    download_result: Optional[dict] = None
    error: Optional[str] = None


def download_task(job_id: str, request: DownloadRequest):
    """백그라운드 다운로드 태스크"""
    try:
        download_jobs[job_id] = {"status": "processing", "message": "다운로드 진행 중..."}
        
        result = downloader.download_by_location(
            latitude=request.latitude,
            longitude=request.longitude,
            location_name=request.location_name,
            start_date=request.start_date,
            end_date=request.end_date,
            buffer=request.buffer,
            max_results=request.max_results,
            select_insar_pair=request.select_insar_pair
        )
        
        download_jobs[job_id] = {
            "status": "completed" if result['success'] else "failed",
            "result": result
        }
        
    except Exception as e:
        download_jobs[job_id] = {
            "status": "failed",
            "error": str(e)
        }


@app.post("/search")
async def search_sar(request: SearchRequest):
    """
    Sentinel-1 SAR 데이터 검색 (다운로드 없이 리스트만 조회)
    
    - **latitude**: 위도
    - **longitude**: 경도
    - **location_name**: 지역명 (옵션)
    - **start_date**: 검색 시작 날짜 (YYYY-MM-DD)
    - **end_date**: 검색 종료 날짜 (YYYY-MM-DD)
    - **buffer**: 좌표 주변 버퍼 (도 단위, 기본값 0.5도)
    - **max_results**: 최대 검색 결과 수 (기본값 50)
    """
    try:
        # 검색만 수행
        results = downloader.search_sentinel1(
            latitude=request.latitude,
            longitude=request.longitude,
            start_date=request.start_date,
            end_date=request.end_date,
            buffer=request.buffer,
            max_results=request.max_results
        )
        
        if not results:
            return {
                "success": False,
                "message": "검색 결과가 없습니다.",
                "total": 0,
                "products": []
            }
        
        # 날짜별로 그룹화 및 정렬
        date_products = {}
        for idx, product in enumerate(results):
            filename = product.properties['fileName']
            date_str = filename.split('_')[5][:8]  # YYYYMMDD
            
            if date_str not in date_products:
                date_products[date_str] = []
            
            date_products[date_str].append({
                "index": idx,
                "filename": filename,
                "date": date_str,
                "size_mb": round(float(product.properties.get('bytes', 0)) / (1024 * 1024), 2),
                "platform": product.properties.get('platform', 'N/A'),
                "polarization": product.properties.get('polarization', 'N/A'),
                "relative_orbit": product.properties.get('pathNumber') or product.properties.get('relativeOrbit', 'N/A'),
                "flight_direction": product.properties.get('flightDirection', 'N/A')
            })
        
        # 날짜순 정렬
        sorted_dates = sorted(date_products.keys())
        
        # 결과 포맷팅
        products_list = []
        for date_str in sorted_dates:
            for product in date_products[date_str]:
                products_list.append(product)
        
        return {
            "success": True,
            "message": f"{len(products_list)}개의 제품을 찾았습니다.",
            "total": len(products_list),
            "unique_dates": len(sorted_dates),
            "date_range": f"{sorted_dates[0]} ~ {sorted_dates[-1]}" if sorted_dates else "N/A",
            "products": products_list
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"검색 중 오류 발생: {str(e)}",
            "error": str(e),
            "total": 0,
            "products": []
        }


@app.post("/download", response_model=DownloadResponse)
async def download_sar(request: DownloadRequest, background_tasks: BackgroundTasks):
    """
    Sentinel-1 SAR 데이터 다운로드
    
    - **latitude**: 위도
    - **longitude**: 경도
    - **location_name**: 지역명 (옵션)
    - **start_date**: 검색 시작 날짜 (YYYY-MM-DD)
    - **end_date**: 검색 종료 날짜 (YYYY-MM-DD)
    - **buffer**: 좌표 주변 버퍼 (도 단위, 기본값 0.5도)
    - **max_results**: 최대 검색 결과 수 (기본값 10)
    - **master_index**: Master로 선택할 데이터 인덱스 (/search 결과에서)
    - **slave_index**: Slave로 선택할 데이터 인덱스 (/search 결과에서)
    """
    try:
        lat = request.latitude
        lon = request.longitude
        location = request.location_name or f"({lat}, {lon})"
        start_date = request.start_date
        end_date = request.end_date
        
        # 검색 수행
        results = downloader.search_sentinel1(
            latitude=lat,
            longitude=lon,
            start_date=start_date,
            end_date=end_date,
            buffer=request.buffer,
            max_results=request.max_results
        )
        
        if not results:
            return DownloadResponse(
                success=False,
                message="검색 결과가 없습니다.",
                error="no_data"
            )
        
        # Master/Slave 또는 단일 데이터 선택
        selected_products = []
        
        if request.master_index is not None:
            if request.slave_index is not None:
                # InSAR용: Master + Slave (2개)
                print(f"✅ 사용자 선택 (InSAR): Master[{request.master_index}], Slave[{request.slave_index}]")
                
                if request.master_index < len(results) and request.slave_index < len(results):
                    master = results[request.master_index]
                    slave = results[request.slave_index]
                    selected_products = [master, slave]
                    
                    master_name = master.properties['fileName']
                    slave_name = slave.properties['fileName']
                    print(f"Master: {master_name}")
                    print(f"Slave: {slave_name}")
                else:
                    return DownloadResponse(
                        success=False,
                        message=f"인덱스 범위 초과: 최대 {len(results)-1}",
                        error="index_out_of_range"
                    )
            else:
                # 일반 SAR용: 단일 데이터 (1개)
                print(f"✅ 사용자 선택 (단일): [{request.master_index}]")
                
                if request.master_index < len(results):
                    selected = results[request.master_index]
                    selected_products = [selected]
                    
                    selected_name = selected.properties['fileName']
                    print(f"Selected: {selected_name}")
                else:
                    return DownloadResponse(
                        success=False,
                        message=f"인덱스 범위 초과: 최대 {len(results)-1}",
                        error="index_out_of_range"
                    )
        elif request.selected_indices is not None and len(request.selected_indices) > 0:
            # 다중 선택 (selected_indices 사용)
            print(f"✅ 사용자 선택 (다중): {request.selected_indices}")
            for idx in request.selected_indices:
                if idx < len(results):
                    selected = results[idx]
                    selected_products.append(selected)
                    selected_name = selected.properties['fileName']
                    print(f"  [{idx}] Selected: {selected_name}")
                else:
                    return DownloadResponse(
                        success=False,
                        message=f"인덱스 범위 초과: 인덱스 {idx}, 최대 {len(results)-1}",
                        error="index_out_of_range"
                    )
        else:
            # 자동 선택 (가장 최근 2개)
            print("📌 자동 선택: 가장 최근 2개")
            date_products = {}
            for product in results:
                filename = product.properties['fileName']
                date_str = filename.split('_')[5][:8]
                
                if date_str not in date_products:
                    date_products[date_str] = []
                date_products[date_str].append(product)
            
            sorted_dates = sorted(date_products.keys(), reverse=True)
            unique_products = []
            for date_str in sorted_dates:
                unique_products.append(date_products[date_str][0])
            
            if len(unique_products) >= 2:
                selected_products = unique_products[:2]
            elif len(unique_products) == 1:
                selected_products = unique_products
            else:
                return DownloadResponse(
                    success=False,
                    message="선택 가능한 제품이 없습니다.",
                    error="no_products"
                )
        
        # 다운로드 실행
        download_result = downloader.download_products(selected_products)
        
        if not download_result['success']:
            return DownloadResponse(
                success=False,
                message=download_result.get('message', '다운로드 실패'),
                error=download_result.get('message')
            )
        
        return DownloadResponse(
            success=True,
            message='다운로드 완료',
            location=location,
            coordinates={'latitude': lat, 'longitude': lon},
            date_range=f"{start_date} ~ {end_date}",
            search_results=len(results),
            download_result=download_result,
            error=None
        )
        
    except Exception as e:
        return DownloadResponse(
            success=False,
            message=f"다운로드 중 오류 발생: {str(e)}",
            error=str(e)
        )


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "service": "SAR Download API",
        "version": "1.0.0"
    }


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """다운로드 작업 상태 조회 (향후 비동기 처리용)"""
    if job_id not in download_jobs:
        return {
            "success": False,
            "message": "작업을 찾을 수 없습니다.",
            "job_id": job_id
        }
    
    return {
        "success": True,
        "job_id": job_id,
        **download_jobs[job_id]
    }


if __name__ == "__main__":
    import sys
    import os
    # 부모 디렉토리를 sys.path에 추가 (agent_cv)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("🚀 Starting SAR Download API Server...")
    print("📡 Server will be available at: http://localhost:8001")
    print("📖 API docs: http://localhost:8001/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
