# InSAR Processing Service

SNAP을 사용한 InSAR (Interferometric SAR) 처리 API 서비스

## 📡 Port: 8002

## 🚀 시작 방법

```bash
# 1. esa_snappy 설치 (최초 1회)
bash INSTALL_ESA_SNAPPY.sh

# 2. 서비스 시작
bash start_insar_api.sh
```

## 📝 API 엔드포인트

### POST /insar
InSAR 처리 실행

```json
{
  "master_file": "/mnt/sar/S1A_...zip",
  "slave_file": "/mnt/sar/S1A_...zip",
  "subswath": "IW3",
  "polarization": "VV",
  "first_burst": 1,
  "last_burst": 4,
  "workdir": "/tmp/insar_output"
}
```

**응답**:
```json
{
  "success": true,
  "message": "InSAR processing completed",
  "output_dim": "/tmp/insar_output/ifg_ml_fit.dim",
  "output_tc_dim": "/tmp/insar_output/ifg_ml_fit_tc.dim",
  "phase_band": "Phase_ifg_VV",
  "workdir": "/tmp/insar_output"
}
```

## 🛰️ 처리 단계

1. TOPSAR Split (관심 영역 추출)
2. Apply Orbit File (궤도 정보 적용)
3. Back-Geocoding (영상 정합)
4. Enhanced Spectral Diversity (ESD 보정)
5. Interferogram 생성
6. TOPSAR Deburst
7. Topographic Phase Removal
8. Multilooking
9. Goldstein Phase Filtering
10. Terrain Correction

## ⏱️ 처리 시간

약 20-30분 소요 (데이터 크기와 시스템 성능에 따라 다름)

## 📦 Dependencies

- esa_snappy (SNAP Python API)
- FastAPI
- uvicorn
- Python 3.7+

## 🔧 Requirements

- SNAP Desktop 설치 필요 (`/home/mjh/esa-snap`)
- esa_snappy 설정 완료 (`INSTALL_ESA_SNAPPY.sh` 실행)
