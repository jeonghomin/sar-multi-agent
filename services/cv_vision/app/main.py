# FastAPI 엔드포인트

from fastapi import FastAPI, BackgroundTasks, Query, File, UploadFile, Form
from pydantic import BaseModel, Field
import time
from typing import Optional, Dict, Any
from enum import Enum
from app.paths import job_dir, NAS_ROOT, MODEL_WEIGHTS_DIR
from app.model import DualTaskWrapper
from app.model.integrated_model import MultiTaskWrapper
from app.model.copernicus_encoder import CopernicusFMMultiTaskModel, CopernicusFMEncoder
from app.model.segmentation_head import SegmentationDecoder
from fastapi.staticfiles import StaticFiles
from app.storage import save_mask_png, save_json
from app.load import load_s1_sar_image, preprocess_s1_sar_data, validate_s1_sar_data
from app.visualize import create_3channel_visualization, create_color_mask
from app.visualize_full import create_full_visualization
from app.lulc_analysis import create_lulc_analysis, create_analysis_result
import numpy as np
import os
import uuid
import json
import torch


MODEL : Optional[DualTaskWrapper] = None
SEGMENTATION_MODEL : Optional[tuple] = None  # (encoder, decoder) - S1 only



app = FastAPI(title="SARDIP AI-Service")

class Task(str, Enum):
    SEGMENTATION = "Segmentation" # LULC 
    DETECTION = "Detection" # Ship Detection 
    CLASSIFICATION = "Classification"
    # Change Detection
    # Trajectory
    
class RunJobIn(BaseModel):
    job_id: Optional[str] = Field(None, description="작업 식별자 (미입력시 자동 생성)")
    task: Task = Field(..., description='수행할 테스크')
    decoder: Optional[str] = Field(None, description="디코더")
    input_ref : str = Field(..., description="이미지 경로")
    params: Dict[str, Any] = Field(default_factory=dict, description="하이퍼파라미터")
    user_id: Optional[str] = Field(None, description="사용자 ID")
    request_id: Optional[str] = Field(None, description="요청 ID")
    sar_image_id: Optional[str] = Field(None, description="SAR 영상 ID")
    output_folder: Optional[str] = Field(None, description="출력 폴더 경로 (사용자ID/요청ID 구조)")
    output_filename: Optional[str] = Field(None, description="출력 파일명 (확장자 제외)")



def process_job(job_id: str, payload: RunJobIn, return_result: bool = False):
    """작업 처리 함수 (백그라운드 또는 동기 실행)
    
    Args:
        job_id: 작업 식별자
        payload: 작업 요청 데이터
        return_result: True면 분석 결과 JSON 반환, False면 None 반환 (기존 백그라운드 방식)
    """
    global MODEL, SEGMENTATION_MODEL

    out_path = None
    analysis_result = None  # 분석 결과 저장용

    if payload.task == Task.SEGMENTATION:
        # params에서 use_gt 옵션 확인 (Ground Truth 사용 여부)
        use_gt = payload.params.get("use_gt", False)
        
        if use_gt:
            # Ground Truth Label을 사용 (모델 추론 대신)
            print("[GT Mode] Using Ground Truth Label instead of model inference")
            
            # S1 파일 경로에서 대응하는 GT 파일 경로 생성
            import re
            s1_filename = os.path.basename(payload.input_ref)
            gt_filename = s1_filename.replace('_s1_', '_dfc_')
            
            # GT 파일 찾기 (여러 경로 시도)
            gt_path = None
            search_paths = []
            
            # 1. S1 파일과 같은 디렉토리
            s1_dir = os.path.dirname(payload.input_ref)
            path1 = os.path.join(s1_dir, gt_filename)
            search_paths.append(path1)
            if os.path.exists(path1):
                gt_path = path1
            
            # 2. 경로에 /s1/이 있으면 /dfc/로 변경
            if gt_path is None and '/s1/' in payload.input_ref:
                path2 = payload.input_ref.replace('/s1/', '/dfc/').replace(s1_filename, gt_filename)
                search_paths.append(path2)
                if os.path.exists(path2):
                    gt_path = path2
            
            # 3. DFC2020 원본 데이터셋 경로
            if gt_path is None:
                dfc_base = "/home/mjh/Project/Foundation/Copernicus-FM/Copernicus-Bench/data/copernicusbench/dfc2020_s1s2/dfc"
                path3 = os.path.join(dfc_base, gt_filename)
                search_paths.append(path3)
                if os.path.exists(path3):
                    gt_path = path3
            
            if gt_path is None:
                error_msg = f"Ground Truth 파일을 찾을 수 없습니다.\n"
                error_msg += f"찾는 파일: {gt_filename}\n"
                error_msg += f"검색한 경로들:\n"
                for p in search_paths:
                    error_msg += f"  - {p}\n"
                raise FileNotFoundError(error_msg)
            
            # GT Label 로드
            import rasterio
            with rasterio.open(gt_path) as src:
                predicted_mask = src.read(1)
            
            print(f"[GT Mode] Loaded Ground Truth from: {gt_path}")
            print(f"[GT Mode] GT shape: {predicted_mask.shape}, unique classes: {np.unique(predicted_mask)}")
            
            # S1 데이터 로드 (시각화용)
            image_array, file_type = load_s1_sar_image(payload.input_ref)
            if validate_s1_sar_data(image_array):
                image_tensor, vv_band, vh_band = preprocess_s1_sar_data(image_array, target_size=256)
                
                # 3채널 이미지를 NAS에 저장
                job_dir_path = job_dir(job_id)
                input_vis_path = os.path.join(job_dir_path, "input_3ch.png")
                create_3channel_visualization(vv_band, vh_band, input_vis_path)
        else:
            # 실제 모델 추론 사용
            if SEGMENTATION_MODEL is None:
                raise RuntimeError("Segmentation 모델이 로드되지 않았습니다.")
            
            encoder, decoder = SEGMENTATION_MODEL
            
            # S1 데이터 로드
            image_array, file_type = load_s1_sar_image(payload.input_ref)
            
            # 데이터 유효성 검사
            if not validate_s1_sar_data(image_array):
                raise ValueError("S1 SAR 데이터 유효성 검사 실패")
            
            # 데이터 전처리
            image_tensor, vv_band, vh_band = preprocess_s1_sar_data(image_array, target_size=256)
            print(f"[S1 SAR Mode] Input tensor shape: {image_tensor.shape}")
            
            # 3채널 이미지를 NAS에 저장
            job_dir_path = job_dir(job_id)
            input_vis_path = os.path.join(job_dir_path, "input_3ch.png")
            create_3channel_visualization(vv_band, vh_band, input_vis_path)
            
            # Segmentation 수행
            with torch.no_grad():
                global_features, multi_scale_features = encoder(image_tensor)
                seg_output = decoder(multi_scale_features, target_size=(256, 256))  # config.yaml에 맞춤
                
                # 결과를 numpy로 변환
                seg_probs = torch.softmax(seg_output, dim=1)
                predicted_mask = torch.argmax(seg_probs, dim=1).squeeze(0).cpu().numpy()
        
        result = {"mask": predicted_mask}
        out_path = save_mask_png(job_id, result["mask"], 
                                custom_name=payload.output_filename,
                                user_id=payload.user_id,
                                request_id=payload.request_id,
                                sar_image_id=payload.sar_image_id,
                                output_folder=payload.output_folder)
        
        # LULC 분석 결과 JSON 생성
        lulc_json = create_lulc_analysis(job_id, predicted_mask, payload.input_ref)
        analysis_result = lulc_json  # 동기 처리용 결과 저장
        json_path = save_json(job_id, lulc_json, "lulc_analysis", 
                             custom_name=payload.output_filename,
                             user_id=payload.user_id,
                             request_id=payload.request_id,
                             sar_image_id=payload.sar_image_id,
                             output_folder=payload.output_folder)
        
        # 컬러 시각화 마스크 생성 (PNG와 TIF 둘 다)
        job_dir_path = job_dir(job_id)
        
        # PNG 형식으로 저장
        vis_png_path = os.path.join(job_dir_path, "mask_vis.png")
        create_color_mask(predicted_mask, vis_png_path, "png")
        
        # TIF 형식으로 저장
        vis_tif_path = os.path.join(job_dir_path, "mask_vis.tif")
        create_color_mask(predicted_mask, vis_tif_path, "tif")
        
        # Full visualization 생성 (S1, S2, Prediction)
        try:
            full_vis_path = os.path.join(job_dir_path, "full_visualization.png")
            s2_path = payload.input_ref.replace('/s1/', '/s2/').replace('_s1_', '_s2_')
            create_full_visualization(
                s1_path=payload.input_ref,
                s2_path=s2_path if os.path.exists(s2_path) else None,
                mask_path=vis_png_path,
                output_path=full_vis_path,
                lulc_stats=lulc_json['result']['lulc_summary']
            )
            print(f"Full visualization saved: {full_vis_path}")
        except Exception as e:
            print(f"Full visualization 생성 실패: {e}")
        
        # 비교 이미지 생성 (S1, S2(있다면), 예측 결과)
        from app.visualize import create_comparison_visualization
        
        # S1 3채널 이미지 준비 (정규화된 RGB)
        s1_rgb = np.stack([vv_band, vh_band, (vv_band + vh_band) / 2], axis=-1)
        # 0-255 범위로 정규화
        s1_rgb_vis = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(3):
            ch_min, ch_max = s1_rgb[:, :, i].min(), s1_rgb[:, :, i].max()
            if ch_max > ch_min:
                s1_rgb_vis[:, :, i] = ((s1_rgb[:, :, i] - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
        
        # S2 이미지가 있는지 확인 (파일명에서 s1을 s2로 변경해서 확인)
        s2_image_vis = None
        s2_path = payload.input_ref.replace("_s1_", "_s2_")
        if os.path.exists(s2_path) and s2_path != payload.input_ref:
            try:
                from app.load import load_s2_optical_image, preprocess_s2_optical_data
                s2_array, _ = load_s2_optical_image(s2_path)
                s2_tensor, s2_rgb = preprocess_s2_optical_data(s2_array, target_size=256)
                s2_image_vis = s2_rgb
                print(f"S2 이미지 로드 성공: {s2_path}")
            except Exception as e:
                print(f"S2 이미지 로드 실패: {e}")
        
        # 비교 이미지 저장
        comparison_path = os.path.join(job_dir_path, "comparison_vis.png")
        create_comparison_visualization(s1_rgb_vis, predicted_mask, comparison_path, s2_image_vis)
        
        # 비교 이미지 경로를 out_path로 설정 (최종 출력)
        out_path = comparison_path
        
    elif payload.task == Task.DETECTION:
        if MODEL is None:
            raise RuntimeError("Detection 모델이 로드되지 않았습니다.")
        
        confidence_threshold = payload.params.get("confidence_threshold", 0.5)
        detections = MODEL.predict_detection(payload.input_ref, confidence_threshold)
        
        # Detection 전용 메타데이터
        detection_metadata = {
            "analysis_type": "Object Detection",
            "confidence_threshold": confidence_threshold,
            "num_detections": len(detections)
        }
        
        # Detection 결과 데이터
        detection_data = {
            "detections": detections,
            "num_detections": len(detections),
            "confidence_threshold": confidence_threshold
        }
        
        # 범용 분석 결과 생성
        result = create_analysis_result(job_id, "Detection", detection_data, payload.input_ref, detection_metadata)
        analysis_result = result  # 동기 처리용 결과 저장
        out_path = save_json(job_id, result, "detection_analysis", 
                             custom_name=payload.output_filename,
                             user_id=payload.user_id,
                             request_id=payload.request_id,
                             sar_image_id=payload.sar_image_id,
                             output_folder=payload.output_folder)
        
    elif payload.task == Task.CLASSIFICATION:
        if MODEL is None:
            raise RuntimeError("Classification 모델이 로드되지 않았습니다.")
        
        cls_id, confidence = MODEL.predict_classification(payload.input_ref)
        
        # Classification 전용 메타데이터
        classification_metadata = {
            "analysis_type": "Image Classification",
            "predicted_class_id": int(cls_id),
            "confidence_score": float(confidence)
        }
        
        # Classification 결과 데이터
        classification_data = {
            "class_id": int(cls_id),
            "confidence": float(confidence),
            "class_name": f"class_{cls_id}"  # 실제로는 클래스 이름 매핑 필요
        }
        
        # 범용 분석 결과 생성
        result = create_analysis_result(job_id, "Classification", classification_data, payload.input_ref, classification_metadata)
        analysis_result = result  # 동기 처리용 결과 저장
        out_path = save_json(job_id, result, "classification_analysis", 
                             custom_name=payload.output_filename,
                             user_id=payload.user_id,
                             request_id=payload.request_id,
                             sar_image_id=payload.sar_image_id,
                             output_folder=payload.output_folder)

    print({"event":"done", "job_id": job_id, "task": payload.task, "out": out_path})
    
    # 동기 처리 요청시 분석 결과 반환
    if return_result:
        return {
            "job_id": job_id,
            "task": payload.task.value,
            "status": "completed",
            "output_path": out_path,
            "analysis_result": analysis_result
        }

try:
    MODEL = DualTaskWrapper(
        weights_path=os.getenv("WEIGHTS_PATH"),
        img_size=224,
        num_classes_cls=1000,
        num_classes_det=80,
        num_queries=100
    )
    print("[startup] DualTask model loaded")
except Exception as e:
    print(f"[startup] model loading failed: {e}")
    MODEL = None


try:
    encoder_s1 = CopernicusFMEncoder(
        model_size="base",
        img_size=256,  # config.yaml에 맞춤
        in_chans=2,  # S1 SAR 데이터 (VV, VH)
        input_mode="spectral",
        band_wavelengths=[50000000, 50000000],  # config.yaml 값 사용
        band_bandwidths=[1e9, 1e9],  # config.yaml 값 사용 (1e9 = 1,000,000,000)
        pretrained_path="CopernicusFM_ViT_base_varlang_e100.pth"
    )
    decoder_s1 = SegmentationDecoder(
        embed_dim=768,
        num_classes=8,  # config.yaml에 맞춤 (S1 SAR 데이터, 8개 유효 클래스)
        channels=512,
        dropout_ratio=0.1,
        pretrained_path="s1_seg.ckpt"
    )
    encoder_s1.eval()
    decoder_s1.eval()
    SEGMENTATION_MODEL = (encoder_s1, decoder_s1)
    print("[startup] Copernicus-FM Segmentation model (S1 SAR) loaded")
except Exception as e:
    print(f"[startup] S1 segmentation model loading failed: {e}")
    SEGMENTATION_MODEL = None

# S2 모델은 학습되지 않았으므로 제거
# TODO: S2 segmentation decoder를 학습한 후 추가




@app.post("/run-job")
def run_job(body: RunJobIn, bg: BackgroundTasks):
    """
    작업 실행 API (POST 방식)
    
    지원하는 작업:
    - Segmentation: 이미지 분할
    - Detection: 객체 검출  
    - Classification: 이미지 분류
    
    예제 요청:
    {
        "job_id": "job_001",
        "task": "Detection",
        "input_ref": "/path/to/image.jpg",
        "params": {
            "confidence_threshold": 0.5,
            "img_size": 224,
            "num_classes_det": 80
        }
    }
    """
    # job_id가 없으면 자동 생성
    job_id = body.job_id or str(uuid.uuid4())
    
    # 응답은 바로 주되, 실제 처리는 백그라운드 테스크에서 진행
    bg.add_task(process_job, job_id, body)

    return {"job_id": job_id, "accepted": True, "task": body.task}


@app.post("/run-job-sync")
def run_job_sync(body: RunJobIn):
    """
    동기 작업 실행 API - 결과가 나올 때까지 대기 후 분석 결과 반환
    
    지원하는 작업:
    - Segmentation: LULC 분석 결과 (클래스별 면적, 비율 등)
    - Detection: 객체 검출 결과 (검출된 객체 목록, 개수, confidence)
    - Classification: 분류 결과 (class_id, confidence)
    
    예제 요청:
    {
        "task": "Segmentation",
        "input_ref": "/path/to/image.tif"
    }
    
    예제 응답:
    {
        "job_id": "job_001",
        "task": "Segmentation",
        "status": "completed",
        "output_path": "/path/to/output.png",
        "analysis_result": { ... LULC 분석 결과 ... }
    }
    """
    # job_id가 없으면 자동 생성
    job_id = body.job_id or str(uuid.uuid4())
    
    try:
        # 동기적으로 처리하고 결과 반환
        result = process_job(job_id, body, return_result=True)
        return result
    except Exception as e:
        return {
            "job_id": job_id,
            "task": body.task.value,
            "status": "failed",
            "error": str(e)
        }


@app.get("/run-job")
def run_job_get(
    job_id: str = Query(..., description="작업 식별자"),
    task: Task = Query(..., description="수행할 테스크"),
    input_ref: str = Query(..., description="이미지 경로"),
    confidence_threshold: Optional[float] = Query(0.5, description="신뢰도 임계값"),
    img_size: Optional[int] = Query(224, description="이미지 크기"),
    num_classes_cls: Optional[int] = Query(1000, description="분류 클래스 수"),
    num_classes_det: Optional[int] = Query(80, description="검출 클래스 수"),
    num_queries: Optional[int] = Query(100, description="검출 쿼리 수"),
    bg: BackgroundTasks = BackgroundTasks()
):
    # Query string 파라미터를 RunJobIn 형태로 변환
    params = {
        "confidence_threshold": confidence_threshold,
        "img_size": img_size,
        "num_classes_cls": num_classes_cls,
        "num_classes_det": num_classes_det,
        "num_queries": num_queries
    }
    
    payload = RunJobIn(
        task=task,
        input_ref=input_ref,
        params=params
    )
    
    # 백그라운드 작업으로 처리 (job_id와 payload 분리)
    bg.add_task(process_job, job_id, payload)
    
    return {
        "job_id": job_id, 
        "accepted": True, 
        "task": task.value,
        "message": "작업이 백그라운드에서 처리됩니다."
    }


@app.post("/run-job-with-file")
async def run_job_with_file(
    task: Task = Form(..., description="수행할 테스크"),
    file: UploadFile = File(..., description="업로드할 이미지 파일"),
    user_id: Optional[str] = Form(None, description="사용자 ID"),
    request_id: Optional[str] = Form(None, description="요청 ID"),
    sar_image_id: Optional[str] = Form(None, description="SAR 영상 ID"),
    output_folder: Optional[str] = Form(None, description="출력 폴더 경로 (사용자ID/요청ID 구조)"),
    output_filename: Optional[str] = Form(None, description="출력 파일명 (확장자 제외)"),
    confidence_threshold: Optional[float] = Form(0.5, description="신뢰도 임계값"),
    img_size: Optional[int] = Form(224, description="이미지 크기"),
    num_classes_cls: Optional[int] = Form(1000, description="분류 클래스 수"),
    num_classes_det: Optional[int] = Form(80, description="검출 클래스 수"),
    num_queries: Optional[int] = Form(100, description="검출 쿼리 수"),
    bg: BackgroundTasks = BackgroundTasks()
):
    """
    파일 업로드로 작업 실행 API
    
    지원하는 작업:
    - Segmentation: 이미지 분할
    - Detection: 객체 검출  
    - Classification: 이미지 분류
    
    사용법:
    curl -X POST "http://192.168.10.174:6600/run-job-with-file" \
         -F "task=Segmentation" \
         -F "file=@image.tif" \
         -F "user_id=user123" \
         -F "request_id=req456"
    """
    # job_id 생성
    job_id = str(uuid.uuid4())
    
    # 파일 저장
    output_dir = NAS_ROOT / "uploads"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = output_dir / f"{job_id}_input_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # 파라미터 구성
    params = {
        "confidence_threshold": confidence_threshold,
        "img_size": img_size,
        "num_classes_cls": num_classes_cls,
        "num_classes_det": num_classes_det,
        "num_queries": num_queries
    }
    
    # RunJobIn 형태로 변환 (job_id 제외)
    payload = RunJobIn(
        task=task,
        input_ref=str(file_path),  # PosixPath를 문자열로 변환
        params=params,
        user_id=user_id,
        request_id=request_id,
        sar_image_id=sar_image_id,
        output_folder=output_folder,
        output_filename=output_filename
    )
    
    # 백그라운드 작업으로 처리 (job_id와 payload 분리)
    bg.add_task(process_job, job_id, payload)
    
    return {
        "job_id": job_id,
        "accepted": True,
        "task": task.value,
        "message": "파일이 업로드되고 작업이 백그라운드에서 처리됩니다.",
        "file_path": file_path,
        "file_size": len(content),
        "content_type": file.content_type
    }
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.10.174", port=6600)