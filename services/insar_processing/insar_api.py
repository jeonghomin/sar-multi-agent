"""
InSAR Processing API
Agent로부터 파라미터를 받아서 SNAP InSAR 처리를 수행
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import xml.etree.ElementTree as ET

app = FastAPI(title="InSAR Processing API")


class InSARRequest(BaseModel):
    """InSAR 처리 요청"""
    master_file: str = Field(..., description="Master SAR 파일 경로 (.SAFE.zip)")
    slave_file: str = Field(..., description="Slave SAR 파일 경로 (.SAFE.zip)")
    subswath: str = Field(default="IW3", description="IW1, IW2, or IW3")
    polarization: str = Field(default="VV", description="VV, VH, HH, or HV")
    first_burst: int = Field(default=1, description="시작 Burst 인덱스", ge=1)
    last_burst: int = Field(default=4, description="끝 Burst 인덱스", ge=1)
    workdir: str = Field(default="/tmp/insar_output", description="작업 디렉토리")
    graph_template: str = Field(
        default="/home/mjh/Project/SAR_Process/insar/snappy_InSAR/Turkey_2/graph1_TOPSAR_Coreg_Ifg_w_TC.xml",
        description="Graph1 템플릿 경로"
    )


class InSARResponse(BaseModel):
    """InSAR 처리 결과"""
    success: bool
    message: str
    output_dim: Optional[str] = None
    output_tc_dim: Optional[str] = None
    phase_band: Optional[str] = None
    workdir: Optional[str] = None


def update_graph_xml(
    template_path: str,
    master_file: str,
    slave_file: str,
    subswath: str,
    polarization: str,
    first_burst: int,
    last_burst: int,
    output_path: str,
    output_dim_path: str
) -> str:
    """
    Graph XML 템플릿을 업데이트
    
    Args:
        template_path: 원본 XML 템플릿 경로
        master_file: Master 파일 경로 (7번 줄)
        slave_file: Slave 파일 경로 (15번 줄)
        subswath: IW1/IW2/IW3
        polarization: VV/VH/HH/HV
        first_burst: 시작 burst
        last_burst: 끝 burst
        output_path: 수정된 XML 저장 경로
        output_dim_path: Graph1 출력 .dim 파일 경로
    
    Returns:
        저장된 XML 파일 경로
    """
    tree = ET.parse(template_path)
    root = tree.getroot()
    
    # 1. Master/Slave 파일 경로 업데이트
    nodes = root.findall('.//node')
    
    for node in nodes:
        node_id = node.get('id')
        
        # Read 노드 (Master)
        if node_id == 'Read':
            file_elem = node.find('.//file')
            if file_elem is not None:
                file_elem.text = master_file
        
        # Read(2) 노드 (Slave)
        elif node_id == 'Read(2)':
            file_elem = node.find('.//file')
            if file_elem is not None:
                file_elem.text = slave_file
        
        # TOPSAR-Split 노드들 (Master용)
        elif node_id in ['TOPSAR-Split']:
            params = node.find('.//parameters')
            if params is not None:
                for child in params:
                    if child.tag == 'subswath':
                        child.text = subswath
                    elif child.tag == 'selectedPolarisations':
                        child.text = polarization
                    elif child.tag == 'firstBurstIndex':
                        child.text = str(first_burst)
                    elif child.tag == 'lastBurstIndex':
                        child.text = str(last_burst)
        
        # TOPSAR-Split(2) 노드 (Slave용)
        elif node_id == 'TOPSAR-Split(2)':
            params = node.find('.//parameters')
            if params is not None:
                for child in params:
                    if child.tag == 'subswath':
                        child.text = subswath
                    elif child.tag == 'selectedPolarisations':
                        child.text = polarization
                    elif child.tag == 'firstBurstIndex':
                        child.text = str(first_burst)
                    elif child.tag == 'lastBurstIndex':
                        child.text = str(last_burst)
        
        # Write 노드 (출력 파일 경로)
        elif node_id == 'Write_Goldstein':
            file_elem = node.find('.//file')
            if file_elem is not None:
                file_elem.text = output_dim_path
    
    # XML 저장
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    return output_path


def execute_insar_processing(
    graph1_path: str,
    workdir: str
) -> dict:
    """
    InSAR 처리 실행 (esa_snappy 직접 사용)
    
    Returns:
        dict: {
            'success': bool,
            'output_dim': str,
            'output_tc_dim': str,
            'phase_band': str,
            'message': str
        }
    """
    from esa_snappy import jpy, ProductIO
    
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    
    output_dim = str(workdir_path / "ifg_ml_fit.dim")
    output_tc_dim = str(workdir_path / "ifg_ml_fit_tc.dim")
    
    try:
        # Step 1: Graph1 실행 (InSAR 처리)
        print("[1/3] Running Graph1 (InSAR processing)...")
        execute_graph(graph1_path)
        
        # Step 2: Phase band 찾기
        print("[2/3] Finding phase band...")
        phase_band = find_phase_band(output_dim)
        print(f"  Phase band: {phase_band}")
        
        # Step 3: Terrain Correction
        print("[3/3] Running Terrain Correction...")
        g2_xml = create_terrain_correction_xml(
            input_dim=output_dim,
            phase_band=phase_band,
            output_dim=output_tc_dim
        )
        
        g2_path = workdir_path / "graph2_tc_phase.xml"
        g2_path.write_text(g2_xml, encoding='utf-8')
        
        execute_graph(str(g2_path))
        
        return {
            'success': True,
            'output_dim': output_dim,
            'output_tc_dim': output_tc_dim,
            'phase_band': phase_band,
            'message': 'InSAR processing completed successfully'
        }
    
    except Exception as e:
        import traceback
        error_msg = f'InSAR processing failed: {str(e)}\n{traceback.format_exc()}'
        print(f"❌ {error_msg}")
        return {
            'success': False,
            'message': error_msg
        }


def find_phase_band(dim_path: str) -> str:
    """Phase band 자동 탐색"""
    from esa_snappy import ProductIO
    
    p = ProductIO.readProduct(dim_path)
    bands = [p.getBandAt(i).getName() for i in range(p.getNumBands())]
    
    # 1순위: Phase_ifg로 시작
    for b in bands:
        if b.startswith("Phase_ifg"):
            return b
    
    # 2순위: unit이 "phase"
    for i in range(p.getNumBands()):
        band = p.getBandAt(i)
        if (band.getUnit() or "").lower() == "phase":
            return band.getName()
    
    raise RuntimeError(f"Phase band not found. Bands: {bands}")


def execute_graph(xml_path: str):
    """SNAP Graph 실행"""
    from esa_snappy import jpy
    
    FileReader = jpy.get_type('java.io.FileReader')
    GraphIO = jpy.get_type('org.esa.snap.core.gpf.graph.GraphIO')
    GraphProcessor = jpy.get_type('org.esa.snap.core.gpf.graph.GraphProcessor')
    PrintPM = jpy.get_type('com.bc.ceres.core.PrintWriterProgressMonitor')
    System = jpy.get_type('java.lang.System')
    
    graphFile = FileReader(xml_path)
    graph = GraphIO.read(graphFile)
    processor = GraphProcessor()
    pm = PrintPM(System.out)
    processor.executeGraph(graph, pm)


def create_terrain_correction_xml(input_dim: str, phase_band: str, output_dim: str) -> str:
    """Terrain Correction XML 생성"""
    return f"""<graph id="G2">
  <version>1.0</version>

  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{input_dim}</file>
    </parameters>
  </node>

  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <sourceBands>{phase_band}</sourceBands>
      <demName>SRTM 3Sec</demName>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
      <outputComplex>false</outputComplex>
    </parameters>
  </node>

  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{output_dim}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>
"""


@app.post("/insar", response_model=InSARResponse)
async def process_insar(request: InSARRequest):
    """
    InSAR 처리 API
    
    Example:
        POST /insar
        {
            "master_file": "/mnt/sar/S1A_...zip",
            "slave_file": "/mnt/sar/S1A_...zip",
            "subswath": "IW3",
            "polarization": "VV",
            "first_burst": 1,
            "last_burst": 4,
            "workdir": "/tmp/insar_turkey"
        }
    """
    try:
        # 파일 존재 확인
        master_path = Path(request.master_file)
        slave_path = Path(request.slave_file)
        
        if not master_path.exists():
            raise HTTPException(status_code=400, detail=f"Master file not found: {request.master_file}")
        if not slave_path.exists():
            raise HTTPException(status_code=400, detail=f"Slave file not found: {request.slave_file}")
        
        # 작업 디렉토리 생성
        workdir = Path(request.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        
        # Graph1 XML 업데이트
        print(f"[API] Updating Graph XML...")
        updated_graph_path = str(workdir / "graph1_updated.xml")
        output_dim_path = str(workdir / "ifg_ml_fit")  # .dim 확장자 제외
        
        update_graph_xml(
            template_path=request.graph_template,
            master_file=request.master_file,
            slave_file=request.slave_file,
            subswath=request.subswath,
            polarization=request.polarization,
            first_burst=request.first_burst,
            last_burst=request.last_burst,
            output_path=updated_graph_path,
            output_dim_path=output_dim_path
        )
        print(f"[API] Updated Graph saved: {updated_graph_path}")
        print(f"[API] Output will be saved to: {output_dim_path}.dim")
        
        # InSAR 처리 실행
        print(f"[API] Starting InSAR processing...")
        result = execute_insar_processing(
            graph1_path=updated_graph_path,
            workdir=request.workdir
        )
        
        if result['success']:
            return InSARResponse(
                success=True,
                message=result['message'],
                output_dim=result.get('output_dim'),
                output_tc_dim=result.get('output_tc_dim'),
                phase_band=result.get('phase_band'),
                workdir=request.workdir
            )
        else:
            raise HTTPException(status_code=500, detail=result['message'])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"InSAR processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "ok", "service": "InSAR Processing API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
