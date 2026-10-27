#!/usr/bin/env python3
"""Construct a binary region-adjacency hierarchy from SAM integer labels.
This is a region-adjacency BPT baseline (color-based), not the original BPT algorithm.
"""
import argparse
import heapq
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--spatial-weight',type=float,default=0.0, help='optional centroid-distance factor')
    args=ap.parse_args()
    rgb=np.asarray(Image.open(args.image).convert('RGB'),dtype=np.float64)/255.0
    raw=np.load(args.labels)
    if raw.ndim != 2 or raw.shape != rgb.shape[:2]:
        raise ValueError(f'Expected 2D label map {rgb.shape[:2]}, got {raw.shape}')
    unique, flat = np.unique(raw,return_inverse=True)
    labels=flat.reshape(raw.shape)
    n=len(unique)
    counts=np.bincount(flat,minlength=n).astype(np.float64)
    sums=np.stack([np.bincount(flat,weights=rgb[:,:,c].ravel(),minlength=n) for c in range(3)],axis=1)
    yy,xx=np.indices(labels.shape)
    centers=np.stack([np.bincount(flat,weights=xx.ravel(),minlength=n),np.bincount(flat,weights=yy.ravel(),minlength=n)],axis=1)/counts[:,None]
    pairs=[]
    for a,b in ((labels[:,:-1],labels[:,1:]),(labels[:-1,:],labels[1:,:])):
        sel=(a!=b)
        if np.any(sel):
            edge=np.stack((np.minimum(a[sel],b[sel]),np.maximum(a[sel],b[sel])),axis=1)
            pairs.append(np.unique(edge,axis=0))
    neighbors={i:set() for i in range(n)}
    for a,b in np.unique(np.concatenate(pairs,axis=0),axis=0) if pairs else np.empty((0,2),dtype=int):
        a,b=int(a),int(b);neighbors[a].add(b); neighbors[b].add(a)
    active=set(range(n)); masses={i:float(counts[i]) for i in range(n)}; values={i:sums[i] for i in range(n)}; centroids={i:centers[i] for i in range(n)}
    nodes=[{'id':i,'children':[],'pixel_count':int(counts[i]),'original_label':int(unique[i])} for i in range(n)]
    heap=[]
    diagonal=np.hypot(*labels.shape)
    def weight(a,b):
        color=np.linalg.norm(values[a]/masses[a]-values[b]/masses[b])
        distance=np.linalg.norm(centroids[a]-centroids[b])/diagonal
        return float(color+args.spatial_weight*distance)
    def push(a,b):
        if a!=b and a in active and b in active:
            aa,bb=sorted((a,b));heapq.heappush(heap,(weight(aa,bb),aa,bb))
    for a in range(n):
        for b in neighbors[a]:
            if a<b:push(a,b)
    while len(active)>1:
        candidate=None
        while heap:
            _,a,b=heapq.heappop(heap)
            if a in active and b in active and b in neighbors[a]:
                candidate=(a,b); break
        if candidate is None:
            # Disconnected labels: merge closest-color pair, record this exception.
            ids=sorted(active)
            candidate=min(((a,b) for idx,a in enumerate(ids) for b in ids[idx+1:]),key=lambda pair:weight(*pair))
        a,b=candidate; k=len(nodes)
        m=masses[a]+masses[b]
        nodes.append({'id':k,'children':[a,b],'pixel_count':int(m),'merge_cost':weight(a,b)})
        values[k]=values[a]+values[b]; masses[k]=m
        centroids[k]=(centroids[a]*masses[a]+centroids[b]*masses[b])/m
        merged=(neighbors[a] | neighbors[b]) - {a,b}
        for x in merged:
            neighbors[x].discard(a); neighbors[x].discard(b);neighbors[x].add(k)
        neighbors[k]=merged
        active.remove(a);active.remove(b);active.add(k)
        for x in merged:push(k,x)
    root=next(iter(active))
    output={'format':'sam-region-bpt-v1','source_image':args.image,'source_labels':args.labels,'leaf_count':n,'node_count':len(nodes),'root':root,'nodes':nodes, 'method':'adjacent region agglomeration by mean RGB distance; fallback for disconnected islands'}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(output,indent=2))
    assert len(nodes)==2*n-1
    print(f'[success] {n} leaf regions -> {len(nodes)} nodes, root={root}; tree={path}')

if __name__=='__main__':main()
