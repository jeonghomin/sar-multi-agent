#!/usr/bin/env python3
"""
Agent CV에서 SAR Segmentation 테스트
"""

from graph import graph

# SAR 이미지 경로
sar_image = "/home/mjh/Project/LLM/RAG/files/test_folder/ROIs0000_test_s1_0_p1004.tif"

# 입력 state
inputs = {
    "question": "이 SAR 이미지의 토지 피복을 분석해줘",
    "image_path": sar_image,
    "use_gt": True  # Ground Truth 모드
}

print("=" * 80)
print("🎯 Agent CV SAR Segmentation 테스트")
print("=" * 80)
print(f"이미지: {sar_image}")
print(f"질문: {inputs['question']}")
print(f"GT 모드: {inputs['use_gt']}")
print("=" * 80)

# 그래프 실행
for output in graph.stream(inputs):
    for key, value in output.items():
        print(f"\n{'='*80}")
        print(f"🔧 NODE: {key}")
        print(f"{'='*80}")
        
        if "generation" in value:
            print("\n📝 최종 응답:")
            print(value["generation"])
        
        if "vision_result" in value:
            result = value["vision_result"]
            print("\n🖼️ Vision Result:")
            
            if "error" in result:
                print(f"❌ 에러: {result['error']}")
            elif "lulc_summary" in result:
                print(f"✅ SAR Segmentation 완료!")
                print(f"모드: {result.get('mode', 'Unknown')}")
                print(f"Full viz: {result.get('full_visualization', 'N/A')}")
                
                lulc_summary = result.get("lulc_summary", {})
                if lulc_summary:
                    print("\n📊 LULC 통계:")
                    for class_name, data in sorted(lulc_summary.items(), 
                                                   key=lambda x: -x[1].get('percentage', 0)):
                        label = data.get('label', class_name)
                        percentage = data.get('percentage', 0)
                        print(f"  {label}: {percentage:.2f}%")
            else:
                print(result)

print("\n" + "=" * 80)
print("✅ 테스트 완료!")
print("=" * 80)
