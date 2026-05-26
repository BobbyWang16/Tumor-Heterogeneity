# ========== 公共工具导入 ==========
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.utils import (
    load_image, load_nifti, load_seg, save_nifti,
    match_image_mask_pairs, keep_largest_cc,
    locate_tumor_3d, clip_hu, crop_to_mask,
)
# ==================================

import argparse, csv, json, math
from pathlib import Path
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import SimpleITK as sitk             # 只负责 IO + 重采样

# ────────────────────────── 工具 ──────────────────────────
def load_nifti(path):
    return sitk.ReadImage(str(path))

def resample_isotropic(img, target_spacing=(1,1,1), is_label=False):
    orig_spacing = img.GetSpacing()
    if np.allclose(orig_spacing, target_spacing):     # 已经各向同性
        return img
    orig_size = np.array(img.GetSize(), dtype=float)
    new_size = np.round(orig_size * np.array(orig_spacing) / np.array(target_spacing)).astype(int)
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    return resampler.Execute(img)

def crop_to_mask(img_np, mask_np, margin=2):
    idx = np.where(mask_np)
    if len(idx[0]) == 0:                        # 空掩膜，直接返回原图
        return img_np, mask_np
    zmin,zmax = idx[0].min(), idx[0].max()
    ymin,ymax = idx[1].min(), idx[1].max()
    xmin,xmax = idx[2].min(), idx[2].max()
    zmin = max(zmin - margin, 0)
    ymin = max(ymin - margin, 0)
    xmin = max(xmin - margin, 0)
    zmax += margin + 1
    ymax += margin + 1
    xmax += margin + 1
    return (img_np[zmin:zmax, ymin:ymax, xmin:xmax],
            mask_np[zmin:zmax, ymin:ymax, xmin:xmax])

# ───── 分形维数 (盒计数) ─────
def box_count(mask, s, offsets):
    counts = []
    for off in offsets:
        sub = mask[off[0]:, off[1]:, off[2]:]
        # 截取到 s 的倍数尺寸后再 reshape
        crop_shape = (sub.shape[0]//s*s, sub.shape[1]//s*s, sub.shape[2]//s*s)
        if min(crop_shape) == 0:
            continue
        sub = sub[:crop_shape[0], :crop_shape[1], :crop_shape[2]]
        new_shape = (crop_shape[0]//s, s,
                     crop_shape[1]//s, s,
                     crop_shape[2]//s, s)
        sub = sub.reshape(new_shape)
        # 一个盒子里只要存在前景就算“盖住”
        cnt = (sub.sum(axis=(1,3,5)) > 0).sum()
        counts.append(cnt)
    return np.mean(counts)

def fractal_dimension(mask, min_box=4, max_box=None, n_sizes=6, n_offset=4):
    if max_box is None:
        max_box = min(mask.shape)//2
    sizes = np.unique(np.geomspace(min_box, max_box, num=n_sizes).astype(int))
    Bs, Ss = [], []
    rng = np.random.default_rng(0)
    for s in sizes:
        offsets = rng.integers(0, s, size=(n_offset, 3))
        Bs.append(box_count(mask, s, offsets))
        Ss.append(s)
    Bs, Ss = np.array(Bs), np.array(Ss)
    F = -np.polyfit(np.log(Ss), np.log(Bs), 1)[0]
    return F, Ss, Bs

# ───── 空隙度 (滑动盒) ─────
def lacunarity(mask, box_sizes=(3,5,7,9,11)):
    Γ, T = [], []
    for t in box_sizes:
        kernel = np.ones((t,t,t), dtype=np.float32)
        v_map = ndi.convolve(mask.astype(np.float32), kernel, mode='constant', cval=0)
        v = v_map.ravel()
        mean = v.mean()
        var = v.var()
        Γ.append(var / mean**2 + 1)
        T.append(t)
    return np.array(T), np.array(Γ)

# ───── 可视化 ─────
def plot_fd(Ss, Bs, FD, save_p):
    logS, logB = np.log10(Ss), np.log10(Bs)
    m, c = np.polyfit(logS, logB, 1)
    plt.figure(figsize=(3.6,3.6))
    plt.scatter(logS, logB)
    plt.plot(logS, m*logS + c, color='r', label=f'FD={-m:.3f}')
    plt.xlabel('log10(Box size)'); plt.ylabel('log10(Count)')
    plt.legend(); plt.tight_layout()
    plt.savefig(save_p, dpi=300); plt.close()

def plot_lacunarity(T, Γ, save_p):
    plt.figure(figsize=(3.6,3.6))
    plt.plot(T, Γ, 'o-')
    plt.xlabel('Box size t'); plt.ylabel('Lacunarity Γ(t)')
    plt.tight_layout(); plt.savefig(save_p, dpi=300); plt.close()

# ───── 单病例分析 ─────
def analyse_case(img_p, msk_p, args, out_root):
    case_id = img_p.stem
    # 1) 载入 & 重采样
    img = load_nifti(img_p)
    msk = load_nifti(msk_p)
    img = resample_isotropic(img, args.spacing)
    msk = resample_isotropic(msk, args.spacing, is_label=True)

    mask_np = sitk.GetArrayFromImage(msk) > 0          # (z,y,x)
    img_np  = sitk.GetArrayFromImage(img)

    # 2) 裁剪至最小包围盒
    img_np, mask_np = crop_to_mask(img_np, mask_np)

    # 3) FD & Lacunarity
    FD, Ss, Bs = fractal_dimension(mask_np,
                                   args.min_box, args.max_box,
                                   args.n_scales, args.n_offset)
    T, Γ = lacunarity(mask_np, args.lac_sizes)

    # 4) 保存
    case_dir = out_root / case_id
    case_dir.mkdir(exist_ok=True)
    plot_fd(Ss, Bs, FD, case_dir/'fd_plot.png')
    plot_lacunarity(T, Γ, case_dir/'lacunarity_plot.png')
    with open(case_dir/'result.json', 'w') as f:
        json.dump(dict(case_id=case_id,
                       fractal_dimension=FD,
                       box_sizes=Ss.tolist(),
                       box_counts=Bs.tolist(),
                       lac_box_sizes=T.tolist(),
                       lacunarity=Γ.tolist()), f, indent=2)
    return FD, Γ

# ───── 主流程 ─────
def main():
    pa = argparse.ArgumentParser("Batch FD & lacunarity （SimpleITK 版）")
    pa.add_argument("--root",   required=True, help="根目录，包含 image/ label/ 两个子文件夹")
    pa.add_argument("--outdir", required=True, help="输出目录")
    pa.add_argument("--spacing", type=float, nargs=3, default=(1,1,1),
                    help="目标各向同性体素间距，单位与原始像素一致")
    pa.add_argument("--min_box", type=int, default=4)
    pa.add_argument("--max_box", type=int, default=None)
    pa.add_argument("--n_scales", type=int, default=6)
    pa.add_argument("--n_offset", type=int, default=4)
    pa.add_argument("--lac_sizes", type=int, nargs='+', default=[3,5,7,9,11])
    args = pa.parse_args()

    root = Path(args.root)
    img_dir, msk_dir = root/'image', root/'label'
    out_root = Path(args.outdir); out_root.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob('*.nii*'))
    assert images, "image 文件夹为空！"

    csv_path = out_root/'summary.csv'
    head = ['case_id', 'FD'] + [f'Γ_t{t}' for t in args.lac_sizes]
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, head); writer.writeheader()
        for img_p in tqdm(images, desc='Analyse'):
            msk_p = msk_dir / img_p.name.replace("image", "seg")
            if not msk_p.exists():
                print(f"[警告] 掩膜缺失：{msk_p}，跳过")
                continue
            fd, Γ = analyse_case(img_p, msk_p, args, out_root)
            writer.writerow({'case_id': img_p.stem,
                             'FD': fd,
                             **{f'Γ_t{t}': g for t, g in zip(args.lac_sizes, Γ)}})

    print(f"\n全部完成！汇总结果见 {csv_path}")

if __name__ == "__main__":
    main()