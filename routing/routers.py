"""모든 라우터 정의"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from core.llm_config import (
    structured_llm_router,
    structured_llm_retrieval,
    structured_llm_vision
)


# ===== Main Router =====
main_system = """
You are an expert at routing a user question to one of FOUR agents: casual_chat, retrieval, vision, or sar_processing.

CRITICAL: Analyze the user's INTENT and context, not just keywords!

**Available Agents:**

1. **casual_chat** - Casual conversation, greetings, emotions, small talk
   USE FOR:
   - Greetings: "안녕", "hi", "hello", "좋은 아침"
   - Thanks/emotions: "고마워", "감사", "수고했어", "잘했어"
   - Small talk: "잘 지내?", "기분 어때?", "오늘 뭐해?", "날씨 좋네"
   - Simple questions about yourself: "이름이 뭐야?", "뭘 할 수 있어?"

2. **retrieval** - Information queries, Q&A, web search, data requests
   USE FOR:
   - Information queries: news, events, general knowledge
     Examples: "2023년 지진 어디?", "군 복무 기간은?", "최근 뉴스"
   - SAR data requests: requesting/downloading SAR data
     Examples: "SAR 데이터 받아줘", "위성 데이터 다운로드"
   - Document-based Q&A: questions requiring document search
     Examples: "이 자료에 뭐라고 나와?", "찾아줘"

3. **vision** - Computer vision tasks on images (SAR, optical, any image type)
   USE FOR:
   - Segmentation/분할: land cover classification, terrain analysis
     Examples: "SAR 분할", "이미지 세그멘테이션", "LULC 분류"
   - Classification/분류: categorizing images
     Examples: "이미지 분류", "이게 뭐야?"
   - Detection/탐지: finding objects in images
     Examples: "배 찾아줘", "건물 탐지"

4. **sar_processing** - ONLY for specialized InSAR interferometry analysis
   USE FOR:
   - InSAR processing: interferogram generation, phase analysis
     Examples: "InSAR 처리", "간섭무늬 생성", "지표변형 분석"
   - Ground deformation: earthquake/volcano deformation measurement
     Examples: "지진으로 인한 지표변형", "화산 변위 측정"

**ROUTING RULES (STRICT PRIORITY ORDER):**

STEP 1: Check if CASUAL CONVERSATION
   - Greetings: "안녕", "hi", "hello", "좋은 아침", "반가워"
   - Thanks/Emotions: "고마워", "감사", "수고", "잘했어", "좋아"
   - Small talk: "잘 지내?", "기분 어때?", "뭐해?", "날씨", "이름"
   → Route to **casual_chat**
   
STEP 2: Check if VISION TASK
   - Keywords: segmentation, classification, detection, 분할, 분류, 탐지, LULC
   → Route to **vision**
   (NOTE: "SAR 분할" = vision, NOT sar_processing!)
   
STEP 3: Check if INSAR PROCESSING
   - Keywords: InSAR, interferogram, 간섭무늬, 지표변형, ground deformation
   → Route to **sar_processing**
   
STEP 4: Everything else
   - Information queries, web search, SAR data requests
   → Route to **retrieval**

**CRITICAL EXAMPLES:**
- "안녕하세요" → casual_chat ✓
- "고마워요" → casual_chat ✓
- "2023년 지진 어디?" → retrieval ✓
- "SAR 분할" → vision ✓
- "InSAR 처리" → sar_processing ✓

**DO NOT:**
- Route greetings to retrieval (WRONG!)
- Route "SAR 분할" to sar_processing (WRONG!)
"""

main_route_prompt = ChatPromptTemplate.from_messages([
    ("system", main_system),
    ("system", "Conversation summary (older context):\n{summary}"),  # ✅ Summary 추가
    MessagesPlaceholder(variable_name="messages", optional=True),
    ("human", "{question}"),
])

main_agent = main_route_prompt | structured_llm_router


# ===== Vision Router =====
vision_system = """
You are an expert at routing a user question to the appropriate computer vision task.

Available tasks:
- segmentation: Image segmentation including LULC, land cover, SAR segmentation, terrain analysis
- classification: Image classification to determine what category an image belongs to
- detection: Object detection to find and locate objects (ships, vehicles, buildings, etc.)

If the user asks about:
- Segmentation, 분할 (SAR 분할, LULC 분할, land cover, terrain analysis)
→ route to "segmentation"

If the user asks about:
- Classification, 분류 (what is in the image, image category)
→ route to "classification"

If the user asks about:
- Detection, 탐지, finding objects (ships, vehicles, buildings, etc.)
→ route to "detection"
"""

vision_route_prompt = ChatPromptTemplate.from_messages([
    ("system", vision_system),
    ("system", "Conversation summary (older context):\n{summary}"),  # ✅ Summary 추가
    MessagesPlaceholder(variable_name="messages", optional=True),
    ("human", "{question}"),
])

vision_router = vision_route_prompt | structured_llm_vision


# ===== Retrieval Router =====
retrieval_system = """
You are an expert at routing a user question to one of three destinations: vectorstore, web_search, or extract_coordinates.

IMPORTANT: The vectorstore ONLY contains **SAR (Synthetic Aperture Radar) image metadata** for specific coordinates:
- SAR 이미지 파일명, 좌표 정보 (위도/경도)
- SAR 촬영 날짜, 센서 정보
- 특정 좌표의 SAR 이미지 메타데이터

The vectorstore does NOT contain:
- General knowledge or information
- News or current events
- Location information or addresses
- Anything other than SAR metadata

ROUTING RULES (in priority order):

1. Route to "web_search" if:
   - The question is about general information, news, events
   - The question needs real-time or current data
   - The question is about finding location information (예: "최근 홍수난 지역이 어디야?")
   - The question does NOT mention SAR/InSAR/satellite data explicitly
   - DEFAULT: When in doubt and no SAR keywords, use web_search

2. Route to "extract_coordinates" if:
   - The question mentions a location name AND wants SAR data
   - The question has both: location (지역명, 도시명) + SAR keywords
   - Examples: "서울의 SAR 데이터", "Johnston Iowa Farm SAR 정보"
   - The question wants to convert location to coordinates

3. Route to "vectorstore" ONLY if:
   - The question provides SPECIFIC coordinates (lat/lon numbers)
   - The question asks about ALREADY retrieved SAR metadata
   - Examples: "위도 37.5 경도 127.0 SAR 이미지", "이 좌표의 SAR 데이터"

KEY POINT: If the question does NOT explicitly mention SAR/satellite data, route to "web_search".
"""

retrieval_route_prompt = ChatPromptTemplate.from_messages([
    ("system", retrieval_system),
    ("system", "Conversation summary (older context):\n{summary}"),  # ✅ Summary 추가
    MessagesPlaceholder(variable_name="messages", optional=True),
    ("human", "{question}"),
])

question_router = retrieval_route_prompt | structured_llm_retrieval


