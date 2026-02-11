# SAR Download Service

ASF (Alaska Satellite Facility)에서 Sentinel-1 SAR 데이터를 검색하고 다운로드하는 API 서비스

## 📡 Port: 8001

## 🚀 시작 방법

```bash
bash start_sar_api.sh
```

## 📝 API 엔드포인트

### POST /search
SAR 데이터 검색

```json
{
  "latitude": 37.5,
  "longitude": 127.0,
  "location_name": "Seoul",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "buffer": 0.5,
  "max_results": 50
}
```

### POST /download
SAR 데이터 다운로드

```json
{
  "latitude": 37.5,
  "longitude": 127.0,
  "location_name": "Seoul",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "buffer": 0.5,
  "max_results": 50,
  "master_index": 0,
  "slave_index": 1
}
```

## 📦 Dependencies

- FastAPI
- uvicorn
- asf_search
- requests
