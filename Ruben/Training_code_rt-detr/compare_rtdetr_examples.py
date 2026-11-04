"""Create three-panel RT-DETR qualitative comparisons for TRANSSET.

Requires: pip install ultralytics matplotlib pillow pyyaml
Example:
 python compare_rtdetr_examples.py --weights "runs_rtdetr/rtdetr_1024/weights/best.pt" --data "C:/ai/Code/datasets/transset_v3/data.yaml" --split val --filenames file_02583.jpg
 python compare_rtdetr_examples.py --weights ".../best.pt" --data ".../data.yaml" --split test --count 6
 python compare_rtdetr_examples.py --weights ".../best.pt" --external "C:/images/road.jpg"
"""
import argparse
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import RTDETR

NAMES = ['car','hatchback','pickup truck','sedan','suv','trailer','truck','van']
COLORS = ['#1683d0','#eb641b','#19ad78','#d59c00','#d9598e','#117e17','#5b43a9','#cf3a44']
EXTS = {'.jpg','.jpeg','.png','.bmp','.webp'}

def read_config(path, split):
    cfg_path=Path(path).expanduser().resolve()
    cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8-sig'))
    root=Path(cfg.get('path') or cfg_path.parent)
    if not root.is_absolute(): root=(cfg_path.parent/root).resolve()
    val=cfg.get(split)
    if val is None: raise ValueError(f'Split {split} missing from {cfg_path}')
    paths=val if isinstance(val,list) else [val]
    images=[]
    for rel in paths:
        p=Path(rel)
        if not p.is_absolute(): p=root/p
        if p.is_file() and p.suffix.lower()=='.txt':
            for line in p.read_text().splitlines():
                s=line.strip()
                if s:
                    img=Path(s)
                    if not img.is_absolute(): img=(p.parent/img).resolve()
                    images.append(img)
        elif p.is_dir():
            images.extend(sorted(q for q in p.rglob('*') if q.suffix.lower() in EXTS))
        elif p.is_file() and p.suffix.lower() in EXTS: images.append(p)
        else: print(f'WARNING: split location not found: {p}')
    return images

def label_path(image):
    parts=list(image.parts)
    ix=next((i for i,x in enumerate(parts) if x.lower()=='images'),None)
    if ix is None: return None
    parts[ix]='labels'
    return Path(*parts).with_suffix('.txt')

def load_truth(image, width, height):
    p=label_path(image)
    if not p or not p.exists():
        print(f'WARNING: missing label file for {image.name}: {p}')
        return []
    boxes=[]
    for line in p.read_text(encoding='utf-8-sig').splitlines():
        v=line.split()
        if len(v)<5: continue
        try: cls=int(float(v[0])); x,y,w,h=map(float,v[1:5])
        except ValueError: continue
        if not 0<=cls<len(NAMES): continue
        boxes.append((cls,((x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height),None))
    return boxes

def box_iou(a,b):
    xx1=max(a[0],b[0]); yy1=max(a[1],b[1]); xx2=min(a[2],b[2]); yy2=min(a[3],b[3])
    inter=max(0,xx2-xx1)*max(0,yy2-yy1)
    sa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); sb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(sa+sb-inter) if sa+sb-inter>0 else 0

def compare(truth,pred,threshold=.5):
    pairs=sorted(((box_iou(t[1],p[1]),i,j) for i,t in enumerate(truth) for j,p in enumerate(pred)),reverse=True)
    used_t=set();used_p=set();right=0
    for score,i,j in pairs:
        if score<threshold: break
        if i in used_t or j in used_p: continue
        used_t.add(i);used_p.add(j);right+=truth[i][0]==pred[j][0]
    return len(used_t),right,len(truth)-len(used_t),len(pred)-len(used_p)


def analyze_duplicates(pred, threshold=0.7):
    """Find pairs of predictions with strongly overlapping boxes."""
    duplicates = []

    for i in range(len(pred)):
        cls_a, box_a, conf_a = pred[i]

        for j in range(i + 1, len(pred)):
            cls_b, box_b, conf_b = pred[j]

            iou = box_iou(box_a, box_b)

            if iou >= threshold:
                duplicates.append({
                    "box_a": i + 1,
                    "box_b": j + 1,
                    "iou": round(iou, 3),
                    "class_a": NAMES[cls_a],
                    "class_b": NAMES[cls_b],
                    "conf_a": round(conf_a, 3),
                    "conf_b": round(conf_b, 3)
                })

    print(f"\n--- Duplicate analysis (IoU >= {threshold}) ---")
    print(f"Total predictions: {len(pred)}")

    if not duplicates:
        print("No strongly overlapping prediction pairs found.")
    else:
        for d in sorted(duplicates, key=lambda x: x["iou"], reverse=True):
            print(
                f"Box {d['box_a']} ({d['class_a']}, {d['conf_a']}) "
                f"vs Box {d['box_b']} ({d['class_b']}, {d['conf_b']}) "
                f"| IoU = {d['iou']}"
            )

    return duplicates


def plot_panel(ax,im,boxes,title, show_ids=False):
    ax.imshow(im);ax.set_title(title,fontweight='bold',fontsize=10,loc='left');ax.axis('off')
    h,w=im.shape[:2]
    for idx, (cls, xyxy, confidence) in enumerate(boxes, start=1):
        x1,y1,x2,y2=xyxy
        x1=max(0,min(w,x1));x2=max(0,min(w,x2));y1=max(0,min(h,y1));y2=max(0,min(h,y2))
        color=COLORS[cls]
        ax.add_patch(Rectangle((x1,y1),max(0,x2-x1),max(0,y2-y1),fill=False,edgecolor=color,lw=1.7))
        label = NAMES[cls] + (
            f' {confidence:.2f}' if confidence is not None else ''
        )
        if show_ids:
            label = f'#{idx} ' + label
        ax.text(x1,max(9,y1-2),label,color='white',fontsize=6.8,fontweight='bold',va='bottom',
                bbox={'facecolor':color,'edgecolor':'none','pad':1.3,'alpha':.95})

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights',required=True,help='Path to RT-DETR best.pt')
    ap.add_argument('--data',help='Path to TRANSSET data.yaml')
    ap.add_argument('--split',choices=['train','val','test'],default='test')
    ap.add_argument('--filenames',nargs='*',help='Select images by filename, e.g. file_02583.jpg')
    ap.add_argument('--external',nargs='*',help='Image paths outside TRANSSET; no ground truth')
    ap.add_argument('--count',type=int,default=6,help='Number of dataset images to plot when no filenames given')
    ap.add_argument('--imgsz',type=int,default=640,help='Use 640 to match the YOLO example images, or 1024 for RT-DETR native training size')
    ap.add_argument('--conf',type=float,default=.45,help='Use .45 for same visual threshold as YOLO examples')
    ap.add_argument('--out',default='rtdetr_examples',help='Directory to save comparison figures')
    args=ap.parse_args()
    if not args.data and not args.external: ap.error('Specify --data for TRANSSET or --external for outside images')
    if not Path(args.weights).exists(): ap.error(f'weights file not found: {args.weights}')
    selected=[]
    if args.data:
        allimages=read_config(args.data,args.split)
        if args.filenames:
            wanted={s.lower() for s in args.filenames}
            selected=[p for p in allimages if p.name.lower() in wanted]
            missing=wanted-{p.name.lower() for p in selected}
            if missing: print('Not found in',args.split,'split:',', '.join(sorted(missing)))
        else: selected=allimages[:args.count]
    if args.external: selected.extend(Path(s) for s in args.external)
    if not selected: ap.error('No images were found; check the split or filenames')
    model=RTDETR(args.weights)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for image in selected:
        if not image.exists(): print('SKIPPING missing image:',image);continue
        im=np.asarray(Image.open(image).convert('RGB'))
        h,w=im.shape[:2]
        external=bool(args.external and str(image) in args.external)
        truth=None if external else load_truth(image,w,h)
        res=model.predict(source=str(image),imgsz=args.imgsz,conf=args.conf,device=0 if __import__('torch').cuda.is_available() else 'cpu',verbose=False)[0]
        pred=[]
        if res.boxes is not None:
            for coords,cls,score in zip(res.boxes.xyxy.cpu().numpy(),res.boxes.cls.int().cpu().tolist(),res.boxes.conf.cpu().tolist()):
                if 0<=cls<len(NAMES):pred.append((cls,tuple(float(x) for x in coords),float(score)))
        duplicates = analyze_duplicates(pred, threshold=0.7)
        fig,axs=plt.subplots(1,3,figsize=(16,6.1),dpi=160)
        plot_panel(axs[0],im,[],f'1  Original image')
        plot_panel(axs[1],im,truth or [],'2  True labels: '+(str(len(truth))+' vehicles' if truth is not None else 'not available'))
        plot_panel(axs[2],im,pred,f'3  RT-DETR predictions: {len(pred)} vehicles', show_ids=True)
        summary=f'{image.name} | RT-DETR-L at {args.imgsz}px | confidence ≥ {args.conf:.2f} | split: {"external" if external else args.split}'
        if truth is not None:
            found,right,miss,extra=compare(truth,pred)
            summary+=f'\nMatched at IoU ≥ 0.5: {found}/{len(truth)}; correct class: {right}; missed: {miss}; extra boxes: {extra}'
        else: summary+='\nNo ground-truth labels: predictions only; quantitative accuracy cannot be assessed.'
        fig.suptitle(summary,x=.02,y=.995,ha='left',fontsize=10)
        legend='   '.join(f'{NAMES[i]}: true {sum(t[0]==i for t in truth)}/pred {sum(t[0]==i for t in pred)}' if truth is not None else f'{NAMES[i]}: pred {sum(t[0]==i for t in pred)}' for i in range(len(NAMES)))
        fig.text(.02,.025,legend,fontsize=7,ha='left')
        fig.tight_layout(rect=[0,.06,1,.89],w_pad=.6)
        dest=out/(image.stem+'_RTDETR.png');fig.savefig(dest,bbox_inches='tight',facecolor='white');plt.close(fig)
        print('Saved:',dest)

if __name__=='__main__': main()
