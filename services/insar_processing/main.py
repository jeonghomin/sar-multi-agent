import argparse
from pathlib import Path
from esa_snappy import jpy, ProductIO

G2_TEMPLATE = """<graph id="G2">
  <version>1.0</version>

  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{INPUT_DIM}</file>
    </parameters>
  </node>

  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <sourceBands>{PHASE_BAND}</sourceBands>
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
      <file>{OUTPUT_DIM}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>
"""

def find_phase_band(dim_path: str) -> str:
    p = ProductIO.readProduct(dim_path)
    bands = [p.getBandAt(i).getName() for i in range(p.getNumBands())]

    
    for b in bands:
        if b.startswith("Phase_ifg"):
            return b

    #
    for i in range(p.getNumBands()):
        band = p.getBandAt(i)
        if (band.getUnit() or "").lower() == "phase":
            return band.getName()

    raise RuntimeError(f"Phase band not found. Bands were: {bands}")

def execute_graph(xml_path: str):
    """Execute SNAP graph using Python API"""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph1", default="/home/mjh/Project/SAR_Process/insar/snappy_InSAR/Turkey_2/graph1_TOPSAR_Coreg_Ifg_w_TC.xml", help="Graph1 XML path")
    ap.add_argument("--input_dim", default="/home/mjh/Project/SAR_Process/insar/snappy_InSAR/Turkey_2/ifg_ml_fit.dim", help="Graph1 output .dim path (ifg_ml_fit.dim)")
    ap.add_argument("--out", default="/home/mjh/Project/SAR_Process/insar/snappy_InSAR/Turkey_2/ifg_ml_fit_tc.dim", help="Final output .dim path (ifg_ml_fit_tc.dim)")
    ap.add_argument("--workdir", default="/home/mjh/Project/SAR_Process/insar/snappy_InSAR/Turkey_2")
    args = ap.parse_args()

    # 1) Graph1 실행
    print("[1/3] Running Graph1 (InSAR processing)...")
    execute_graph(args.graph1)

    # 2) Graph1 결과에서 phase 밴드명 자동 탐색
    print("[2/3] Finding phase band from Graph1 output...")
    phase_band = find_phase_band(args.input_dim)
    print("  Phase band:", phase_band)

    # 3) Graph2 XML 생성 + 실행 (Terrain-Correction)
    print("[3/3] Generating and running Graph2 (Terrain-Correction)...")
    g2_xml = G2_TEMPLATE.format(
        INPUT_DIM=args.input_dim,
        PHASE_BAND=phase_band,
        OUTPUT_DIM=args.out,
    
    )

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    g2_path = workdir / "graph2_tc_phase.xml"
    g2_path.write_text(g2_xml, encoding="utf-8")

    execute_graph(str(g2_path))
    print("Done:", args.out)

if __name__ == "__main__":
    main()