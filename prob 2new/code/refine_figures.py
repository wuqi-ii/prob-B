"""Redraw just the two requested figures, keeping original results/figures.

python code/refine_figures.py --name default --mesh-step 5
python code/refine_figures.py --all
"""
import argparse
from pathlib import Path
from bearing2.publication_plots import render_saved


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="候选区域与代价深浅图：读取已保存场景，加密重绘")
    parser.add_argument("--name",default="default",choices=["default","boundary"])
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--mesh-step",type=float,default=5.,help="加密求值点间距，单位米；不是插值像素大小")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    for name in (["default","boundary"] if args.all else [args.name]):
        result=render_saved(root,name,args.mesh_step)
        print(f"Figures: {result['directory']}",flush=True)
