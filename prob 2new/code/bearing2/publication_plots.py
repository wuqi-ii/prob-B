"""Readable, source-backed candidate-region and cost figures.

No optimisation is performed here and original solver outputs are never changed.
Dense display samples replay the SAVED scenarios and the same polygon as the
saved run. Each disconnected lobe is interpolated separately, masked by the
analytic constraints, with one shared colour scale. The dark end means low J.
"""
from pathlib import Path
import hashlib
import json
import math
import shutil
import time

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import brentq

from .config import Config
from .geometry import global_to_local
from .metrics import PointEvaluator
from .optimize import cost, COMP, DIST
from .pipeline import build_source_region
from .region import feasible_mask, lobe_bounds, margins, uniform_diameter_bound
from .scenarios import Scenarios

ROOT = Path(__file__).resolve().parents[2]
SURFACE_CACHE_VERSION = "upper-lobe-surface-v1"
INK = "#203440"
MUTED = "#647580"
TEAL = "#147D80"
FILL = "#D8EBE6"
ACCENT = "#CC583D"
BLUE = "#4679AB"
GOLD = "#AC8547"


def style():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    matplotlib.rcParams.update({
        "font.size": 11, "axes.labelsize": 11, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#BDCACF", "axes.linewidth": 0.7,
        "axes.unicode_minus": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "legend.fontsize": 10, "legend.frameon": False,
        "mathtext.fontset": "dejavusans", "path.simplify": False,
    })


def region_outline(config, side=1, count=600):
    """Sample the analytic boundary, including root-solved tip locations."""
    xs = np.linspace(0, config.guaranteed_reception_m, 10001)
    low, high = lobe_bounds(xs, config)
    gap = high - low
    inds = np.flatnonzero(np.isfinite(gap) & (gap >= 0))
    if len(inds) < 2:
        raise ValueError("No non-degenerate candidate lobe.")
    def f(x):
        lo, hi = lobe_bounds(np.array([x]), config)
        return float(hi[0] - lo[0])
    i, j = int(inds[0]), int(inds[-1])
    left = brentq(f, xs[max(0, i-1)], xs[i]) if i else xs[i]
    right = brentq(f, xs[j], xs[min(len(xs)-1,j+1)]) if j+1 < len(xs) else xs[j]
    x = np.linspace(left, right, count)
    lo, hi = lobe_bounds(x, config)
    return np.column_stack([np.r_[x, x[::-1]], side*np.r_[lo, hi[::-1]]])


def _axes(ax):
    from matplotlib.ticker import MaxNLocator
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(color="#ECF0F2", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.7, labelsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("局部 x / m", labelpad=8)
    ax.set_ylabel("局部 y / m", labelpad=8)


def _outline(ax, outline, fill=False):
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(outline, closed=True, facecolor=FILL if fill else "none",
                         edgecolor=TEAL, linewidth=1.6, zorder=4))


def _markers(ax, summary, side=None, annotate=False):
    p = np.array(summary["final"]["point_local_m"])
    q = np.array(summary["nearest_point_reference"]["point_local_m"])
    for point, symbol, color, size in ((q,"D",INK,58),(p,"*",ACCENT,240)):
        if side is None or point[1]*side > 0:
            ax.scatter(*point, marker=symbol, s=size, facecolors=color if symbol=="*" else "white",
                       edgecolors="white" if symbol=="*" else INK, linewidths=1.2, zorder=10)
    if annotate and (side is None or p[1]*side > 0):
        ax.annotate(r"$P^*$"+f"  ({p[0]:.0f}, {p[1]:.0f})",
                    p, xytext=(12,12), textcoords="offset points", fontsize=10.5,
                    color=ACCENT, zorder=11,
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.88,pad=2))


def _legend_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    return [Patch(facecolor=FILL,edgecolor=TEAL,label="固定候选区域"),
            Line2D([],[],marker="*",color="none",markerfacecolor=ACCENT,markeredgecolor="white",markersize=15,label="已保存选点 P*"),
            Line2D([],[],marker="D",color="none",markerfacecolor="white",markeredgecolor=INK,markersize=7,label="原网格最近点（对照）")]


def _save(fig, path):
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ("png","pdf","svg"):
        fig.savefig(path.with_suffix("."+suffix),dpi=260,bbox_inches="tight",pad_inches=.14)
    plt.close(fig)


def plot_constraint_region(path, config, summary, poly, poly_full):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, Rectangle
    style()
    fig = plt.figure(figsize=(14.0,7.8))
    gs = fig.add_gridspec(1,2,left=.065,right=.965,bottom=.23,top=.87,wspace=.25,width_ratios=[1.1,1])
    ax, zoom = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])
    final=np.array(summary["final"]["point_local_m"]); side=1 if final[1]>=0 else -1
    outlines={s:region_outline(config,s) for s in (1,-1)}
    src=np.asarray(poly_full)
    ax.add_patch(Polygon(src,facecolor="#C5D9E8",edgecolor=BLUE,linewidth=.8,zorder=3))
    ax.plot([0,config.far],[0,0],color=BLUE,lw=.9,ls="--",zorder=3)
    ax.scatter(0,0,s=48,c=INK,marker="s",zorder=7)
    ax.annotate(r"$S_1$",(0,0),xytext=(-10,12),textcoords="offset points",fontsize=12)
    mid=src.mean(axis=0)
    ax.annotate("源位置可行域 K",mid,xytext=(0,-32),textcoords="offset points",ha="center",color=BLUE,fontsize=10,
                arrowprops=dict(arrowstyle="-",color=BLUE,lw=.7))
    for s in (1,-1):
        _outline(ax,outlines[s],True)
    ax.plot([0,final[0]],[0,final[1]],color=ACCENT,ls=(0,(4,3)),lw=1.3,zorder=5)
    ax.text(final[0]*.46,final[1]*.46+65*side,
            f"移动 {summary['final']['distance_m']:.1f} m",color=ACCENT,fontsize=10,rotation=0)
    _markers(ax,summary)
    # The source map is not a movement constraint. Show its arc only if in view.
    centre=global_to_local(config.map_center_m,config)
    angles=np.linspace(0,2*math.pi,2000)
    arc=np.column_stack([centre[0]+config.map_radius_m*np.cos(angles),centre[1]+config.map_radius_m*np.sin(angles)])
    ax.plot(arc[:,0],arc[:,1],color="#AAB4BC",lw=.9,ls=":",zorder=1)
    ax.set_xlim(-90,max(config.far,config.guaranteed_reception_m)+65)
    ymax=float(abs(outlines[1][:,1]).max())+95
    ax.set_ylim(-ymax,ymax)
    _axes(ax)

    out=outlines[side];xmin,xmax=out[:,0].min(),out[:,0].max();ymin,ymax=out[:,1].min(),out[:,1].max()
    # Numerical geometry is enlarged without stretching the axes.
    zoom.set_xlim(xmin-32,xmax+32);zoom.set_ylim(ymin-34,ymax+34)
    ax.add_patch(Rectangle((xmin-12,ymin-12),xmax-xmin+24,ymax-ymin+24,fill=False,
                           edgecolor=MUTED,linestyle=(0,(3,3)),linewidth=.7,zorder=6))
    _outline(zoom,out,True)
    x=np.linspace(xmin-40,xmax+40,700);R=config.guaranteed_reception_m;L=config.far
    near=np.sqrt(np.maximum(0,R**2-x*x))
    far=np.sqrt(np.maximum(0,R**2-(x-L)**2))-config.h
    zoom.plot(x,side*near,color=BLUE,lw=1.25,ls="--",zorder=5)
    zoom.plot(x,side*far,color="#759DAB",lw=1.25,ls="--",zorder=5)
    zoom.plot(x,side*config.t*x,color=GOLD,lw=1.25,ls=(0,(5,2,1,2)),zorder=5)
    zoom.plot(x,side*(config.t*(L-x)+config.h),color="#BBA481",lw=1.25,ls=(0,(5,2,1,2)),zorder=5)
    _markers(zoom,summary,side,True)
    _axes(zoom)
    fig.text(.065,.935,"a  第一检测点、源区域与两侧候选区域",fontsize=12,weight="bold")
    fig.text(.555,.935,"b  已选侧候选区域放大",fontsize=12,weight="bold")
    # Compact legends outside geometry, no boxes covering the data.
    fig.legend(handles=_legend_handles(),loc="lower center",bbox_to_anchor=(.5,.128),ncol=3,columnspacing=2.4)
    lines=[Line2D([],[],color=BLUE,ls="--",label="近端接收边界"),
           Line2D([],[],color="#759DAB",ls="--",label="远端接收边界"),
           Line2D([],[],color=GOLD,ls="-.",label="近端角度边界"),
           Line2D([],[],color="#BBA481",ls="-.",label="远端角度边界")]
    fig.legend(handles=lines,loc="lower center",bbox_to_anchor=(.5,.073),ncol=4,columnspacing=1.8)
    fig.text(.5,.035,f"保证接收 ≤ {R:.0f} m   ·   测得交会角 ≥ {config.min_crossing_deg:.0f}°   ·   定位直径上界 {uniform_diameter_bound(config):.2f} m",
             ha="center",fontsize=10,color=MUTED)
    _save(fig,path)


def _fingerprint(root,name,step):
    # Deliberately decouple numerical cache invalidation from typography/layout
    # edits in this module. Bump SURFACE_CACHE_VERSION when evaluation changes.
    digest=hashlib.sha256(f"{SURFACE_CACHE_VERSION}:{step}".encode())
    paths=[Path(root)/"outputs"/name/k for k in ("summary.json","scenarios.npz","candidates.npy")]
    paths += [Path(__file__).with_name("metrics.py"),Path(__file__).with_name("geometry.py"),
              Path(__file__).with_name("region.py"),Path(__file__).with_name("scenarios.py")]
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_or_evaluate_surface(root,config,summary,step=5.,progress=print):
    """More display points, same saved distribution and same fixed lambda.

    This DOES NOT replace/reselect the saved optimal point. Minimum display
    sample and the original selected point are reported separately.
    """
    if not math.isfinite(step) or step<=0:
        raise ValueError("mesh step must be positive")
    root=Path(root);output=root/"outputs"/config.name
    cache=output/"visual_review";cache.mkdir(parents=True,exist_ok=True)
    stamp=_fingerprint(root,config.name,step)
    meta_path=cache/"cost_surface_meta.json";data_path=cache/"cost_surface_samples.npz"
    if meta_path.exists() and data_path.exists():
        meta=json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint")==stamp:
            with np.load(data_path) as z: return z["points"].copy(),z["cost_s"].copy(),meta
    with np.load(output/"scenarios.npz") as z:
        scenarios=Scenarios(z["sources"].copy(),z["errors"].copy(),z["weights"].copy(),int(summary["scenarios"]["seed"]),0,0)
    full,used,_=build_source_region(config)
    evaluator=PointEvaluator(used,scenarios,config)
    out=region_outline(config,1)
    x=np.arange(math.floor(out[:,0].min()/step)*step,out[:,0].max()+step,step)
    y=np.arange(math.floor(out[:,1].min()/step)*step,out[:,1].max()+step,step)
    gx,gy=np.meshgrid(x,y)
    p=np.column_stack([gx.ravel(),gy.ravel()]);p=p[feasible_mask(p,config)]
    pieces=[p,p*[1,-1],region_outline(config,1,80),region_outline(config,-1,80),
            np.array([summary['final']['point_local_m'],summary['nearest_point_reference']['point_local_m']])]
    p=np.unique(np.round(np.vstack(pieces),7),axis=0)
    p=p[feasible_mask(p,config,tol=1e-6)]
    lam=float(summary["lambda"]["base_weight"])
    values=np.empty(len(p));table=np.empty((len(p),7));tic=time.perf_counter();last=tic
    for i,point in enumerate(p):
        m=evaluator(point)
        if m.empty_updates:
            raise ValueError(f"{m.empty_updates} empty updates at {point}; cannot draw misleading low costs")
        values[i]=cost(m.distance_m,m.composite_m,lam,config)
        table[i]=[*point,m.distance_m,m.mean_diameter_m,m.cvar_diameter_m,m.composite_m,values[i]]
        if time.perf_counter()-last>=12:
            progress(f"{config.name}: cost surface {i+1}/{len(p)} exact scenario evaluations",flush=True)
            last=time.perf_counter()
    idx=int(values.argmin())
    meta={"fingerprint":stamp,"scenario_source":"../scenarios.npz","mesh_step_m":step,
          "points":len(p),"scenario_count":len(scenarios),"weight_lambda":lam,
          "evaluation":"saved scenario mean + CVaR; same polygon as original optimisation",
          "display_interpolation":"linear, independent per lobe, analytic feasibility mask; no cross-lobe extrapolation",
          "saved_point":summary['final']['point_local_m'],"saved_point_cost_s":summary['final']['cost_s'],
          "lowest_display_sample":p[idx].tolist(),"lowest_display_sample_cost_s":float(values[idx]),
          "display_min_is_not_a_new_optimisation":True,"runtime_s":time.perf_counter()-tic}
    np.savez_compressed(data_path,points=p,cost_s=values)
    np.savetxt(cache/"cost_surface_samples.csv",table,delimiter=",",comments="",
               header="x_m,y_m,distance_m,mean_diameter_m,cvar_diameter_m,composite_m,cost_s",fmt="%.10g")
    meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    progress(f"{config.name}: saved {len(p)} cost samples, {meta['runtime_s']:.1f}s",flush=True)
    return p,values,meta


def interpolate_lobe(points,values,config,side,resolution=300):
    """Never interpolate across disconnected lobes or paint outside C."""
    outline=region_outline(config,side)
    xx=np.linspace(outline[:,0].min()-5,outline[:,0].max()+5,resolution)
    yy=np.linspace(outline[:,1].min()-5,outline[:,1].max()+5,resolution)
    gx,gy=np.meshgrid(xx,yy);query=np.column_stack([gx.ravel(),gy.ravel()])
    selected=points[:,1]*side>0
    interp=LinearNDInterpolator(points[selected],values[selected])
    z=np.asarray(interp(query))
    z[~feasible_mask(query,config,tol=1e-8)]=np.nan
    return gx,gy,z.reshape(gx.shape)


def plot_cost_heatmap(path,config,summary,rows=None,root=ROOT,mesh_step=5.):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    from matplotlib.ticker import MaxNLocator,FormatStrFormatter
    style()
    points,values,meta=load_or_evaluate_surface(root,config,summary,mesh_step)
    # One focused panel is easier to read in the report.  The lower lobe remains
    # in the cached evaluations for auditing, but is intentionally not plotted.
    fig=plt.figure(figsize=(9.5,7.35))
    ax=fig.add_axes([.13,.255,.74,.58])
    cmap=LinearSegmentedColormap.from_list("low_cost_dark",["#123E5B","#226882","#65A6B6","#B9DBDE","#EFF5F1"])
    upper=points[:,1]>0
    vmin,vmax=float(values[upper].min()),float(values[upper].max())
    if vmax-vmin<1e-9: vmax=vmin+1e-6
    norm=Normalize(vmin,vmax)
    gx,gy,z=interpolate_lobe(points,values,config,1)
    ax.pcolormesh(gx,gy,np.ma.masked_invalid(z),cmap=cmap,norm=norm,
                  shading="auto",rasterized=True,zorder=2)
    localmin,localmax=float(np.nanmin(z)),float(np.nanmax(z))
    levels=MaxNLocator(5).tick_values(vmin,vmax)
    levels=levels[(levels>localmin+.03*(vmax-vmin))&(levels<localmax-.03*(vmax-vmin))]
    if len(levels):
        cs=ax.contour(gx,gy,np.ma.masked_invalid(z),levels=levels,
                      colors="#FAFBFB",linewidths=.8,alpha=.88,zorder=3)
        ax.clabel(cs,inline=True,fontsize=9,fmt="%.1f",inline_spacing=12)
    outline=region_outline(config,1)
    _outline(ax,outline)
    p=np.asarray(summary['final']['point_local_m'],dtype=float)
    if p[1]<0:
        p=p*np.array([1.,-1.])
    ax.scatter(*p,marker="*",s=270,facecolor=ACCENT,edgecolor="white",
               linewidth=1.4,zorder=10)
    ax.annotate(r"$P^*$"+f"  ({p[0]:.0f}, {p[1]:.0f}) m",p,
                xytext=(16,-22),textcoords="offset points",fontsize=9.8,
                color=INK,ha="left",va="center",zorder=11,
                arrowprops=dict(arrowstyle="-",color="white",lw=.9),
                bbox=dict(facecolor="white",edgecolor="none",alpha=.86,pad=2.2))
    ax.set_xlim(outline[:,0].min()-22,outline[:,0].max()+22)
    ax.set_ylim(outline[:,1].min()-24,outline[:,1].max()+24)
    _axes(ax)

    fig.text(.10,.942,"上侧候选区域的综合代价分布",fontsize=15.5,
             weight="semibold",ha="left",va="center",color=INK)
    fig.text(.10,.895,
             f"J = 移动时间 + 检测时间 + 定位代价   ·   λ = {float(summary['lambda']['base_weight']):.3f}   ·   深色表示代价更低",
             fontsize=9.7,ha="left",va="center",color=MUTED)
    cbax=fig.add_axes([.25,.132,.50,.025])
    sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    cb=fig.colorbar(sm,cax=cbax,orientation="horizontal")
    cb.ax.xaxis.set_major_locator(MaxNLocator(5));cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=9.5,length=2.5,pad=4)
    cb.set_label("综合代价  J / s（模型折算）",fontsize=10.5,labelpad=9,color=INK)
    fig.text(.5,.045,
             f"ρ = {config.tail_weight:g}，α = {config.cvar_alpha:g}；基于 {meta['scenario_count']} 个保存场景的加密求值，仅在可行域内插值。",
             ha="center",fontsize=8.8,color=MUTED)
    _save(fig,path)
    return meta


def render_saved(root,name,mesh_step=5.):
    root=Path(root);directory=root/"outputs"/name
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    cfg=Config(**summary["config"])
    full,poly,_=build_source_region(cfg)
    dest=root/"figures"/name/"refined"
    plot_constraint_region(dest/"01_candidate_region",cfg,summary,poly,full)
    meta=plot_cost_heatmap(dest/"02_cost_heatmap",cfg,summary,root=root,mesh_step=mesh_step)
    # Keep the conventional figure names in sync so users do not accidentally
    # open the legacy plots in the parent directory.
    aliases={"01_candidate_region":"fig1_constraints_region",
             "02_cost_heatmap":"fig2_cost_heatmap"}
    for source,target in aliases.items():
        for suffix in ("png","pdf","svg"):
            shutil.copy2(dest/f"{source}.{suffix}",dest.parent/f"{target}.{suffix}")
    return {"directory":str(dest),"cost_surface":meta}
