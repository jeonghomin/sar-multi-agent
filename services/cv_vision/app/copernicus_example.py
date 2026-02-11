# Copernicus-FM 사용 예제
import os
import sys

# 환경 변수 설정
os.environ["MODEL_WEIGHTS_DIR"] = "/home/mjh/Copernicus-FM/Copernicus-Bench/fm_weights"

def test_copernicus_fm():
    """Copernicus-FM 모델 테스트"""
    try:
        from app.model import MultiTaskWrapper, CopernicusFMMultiTaskModel
        
        print("=== Copernicus-FM 모델 테스트 ===")
        
        # 1. MultiTaskWrapper with Copernicus-FM
        print("\n1. MultiTaskWrapper with Copernicus-FM:")
        wrapper = MultiTaskWrapper(
            img_size=224,
            num_classes_cls=1000,
            num_classes_det=80,
            num_classes_seg=19,
            use_copernicus_fm=True,  # Copernicus-FM 사용
            vit_size="base",
            pretrained_path="copernicusfm_base_cls.pth",  # 실제 가중치 파일명
            language_embed="language_embeddings.pth",
            key="S2"
        )
        print("MultiTaskWrapper 생성 성공!")
        
        # 2. 직접 CopernicusFMMultiTaskModel 사용
        print("\n2. CopernicusFMMultiTaskModel 직접 사용:")
        model = CopernicusFMMultiTaskModel(
            model_size="base",
            img_size=224,
            num_classes_cls=1000,
            num_classes_seg=19,
            pretrained_path="copernicusfm_base_cls.pth",
            language_embed="language_embeddings.pth",
            key="S2"
        )
        print("CopernicusFMMultiTaskModel 생성 성공!")
        
        # 3. Copernicus-FM Segmentation 테스트
        print("\n3. Copernicus-FM Segmentation 테스트:")
        from app.model import CopernicusFMSegmentation, CopernicusFMSegmentationWrapper
        try:
            seg_model = CopernicusFMSegmentation(
                embed_dim=768,
                num_classes=19,
                channels=512
            )
            print("Copernicus-FM Segmentation 생성 성공!")
            
            seg_wrapper = CopernicusFMSegmentationWrapper(
                embed_dim=768,
                num_classes=19
            )
            print("Copernicus-FM Segmentation Wrapper 생성 성공!")
        except ImportError as e:
            print(f"MMSegmentation 사용 불가: {e}")
            print("간단한 UPerNet Decoder 사용...")
            from app.model import UPerNetDecoder
            upernet = UPerNetDecoder(
                embed_dim=768,
                num_classes=19,
                channels=512,
                use_mmseg=False
            )
            print("UPerNet Decoder 생성 성공!")
        
        print("\n모든 테스트 통과! 🎉")
        
    except ImportError as e:
        print(f"Import 오류: {e}")
        print("Copernicus-FM 모델을 사용할 수 없습니다.")
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def test_with_image():
    """실제 이미지로 테스트"""
    try:
        from app.model import MultiTaskWrapper
        
        print("\n=== 실제 이미지 테스트 ===")
        
        wrapper = MultiTaskWrapper(
            img_size=224,
            num_classes_cls=1000,
            num_classes_det=80,
            num_classes_seg=19,
            use_copernicus_fm=True,
            vit_size="base"
        )
        
        # 테스트 이미지 경로
        test_image = "test_image.jpg"
        
        if os.path.exists(test_image):
            print(f"이미지 테스트: {test_image}")
            
            # Classification
            cls_id, confidence = wrapper.predict_classification(test_image)
            print(f"Classification: 클래스 {cls_id}, 신뢰도 {confidence:.3f}")
            
            # Detection
            detections = wrapper.predict_detection(test_image)
            print(f"Detection: {len(detections)}개 객체 검출")
            
            # Segmentation
            mask = wrapper.predict_segmentation(test_image)
            print(f"Segmentation: 마스크 크기 {mask.shape}")
            
        else:
            print(f"테스트 이미지가 없습니다: {test_image}")
            
    except Exception as e:
        print(f"이미지 테스트 실패: {e}")

if __name__ == "__main__":
    test_copernicus_fm()
    test_with_image()
