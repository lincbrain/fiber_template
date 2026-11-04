"""Stage 3: the stacks made ready for the 3D brain mask and for the reconstruction (Methods, "Blockface reconstruction", steps 2 to 4).

You need
  the jump-corrected working stacks (_work/blockface_raw.ome.zarr, _work/brainmask.ome.zarr: jumps.py) and the brain's `missing`
  sections (configs/subjects/<ID>.yaml).
What it does
    remove_background     the 2D masks hole-filled on every real section, emptied on black (missing) frames
    interpolate_missing   a first fill of the missing sections (a full-field blend; label 2 marks them)
    filter_components_3d  3D connected components smaller than 10 % of the largest dropped from the mask (wax, tray, card)
    crop_and_mask         cropped to the tissue (caudal cutoff applied)
    blockface_bg          the same stack with its background kept, at L2 (the 3D brain-mask network's input)
    correct_shading       a per-section degree-2 illumination field, divided out
    harmonize_intensity   one gain per section, to a common brightness
    export_stage3         the NIfTI exports (zarr2nii), the mask gently smoothed, the L4 copies written
You get
  02_blockface.ome.zarr and 02_blockface_mask.ome.zarr (also read by the histology steps), and nii/02_blockface_{,bg_,mask_}L2 and
  their _L4 copies.

Ported unchanged from the analysis project's code/recon.py (and the export block of its code/run.py stage 3).
"""
import os

import cv2
import numpy as np
import zarr

from linc.blockface import external, zarrio


def _last_slice(cfg, nz):
    """Caudal cutoff: the index of the LAST slice to keep (0..last inclusive); nz-1 = keep all.
    In some stacks the cerebellum tapers out a few sections before the hemispheres end, so only the
    slices up to `last` have all three parts (L/R/cere). The cutoff is manual (read off 02_blockface_L2
    in freeview) and lives in the xlsx `missing` column as a trailing (n, -1) entry (second value < 0),
    symmetric to the leading (-1, k) drop - so ALL slice exclusion stays spreadsheet-driven. Slices
    last+1..nz-1 are dropped from the reconstruction (see crop_and_mask). nii slice index == zarr slice
    index (no reversal), so the freeview number is used verbatim."""
    cuts = [b for b, a in cfg.missing if a < 0]
    return min(cuts) if cuts else nz - 1


def remove_background(cfg):
    """Finalise the brain MASK (does NOT touch the raw): hole-fill the brain mask on every real slice and
    force it EMPTY on every black/missing slice (a network can predict a spurious ~full mask on pure black). The
    raw stack is deliberately LEFT UNMASKED - crop_and_mask masks the final blockface itself (bf * m), and
    the unmasked jump-corrected raw is what `blockface_bg` crops for the background-intact editing view.
    Runs after jump correction (masking commutes with the rigid warp)."""
    from scipy.ndimage import binary_fill_holes
    rg = zarr.open_group(cfg.raw_stack, mode="r"); rz = rg["0"]      # (3, rows, slices, cols) - read only
    mg = zarr.open_group(cfg.mask_stack, mode="r+"); mz = mg["0"]    # (1, rows, slices, cols)
    nz = rz.shape[2]
    blanked = 0
    for s in range(nz):
        if int(np.asarray(rz[:, :, s, :]).max()) == 0:             # missing (black-filled) slice:
            m = np.zeros(rz.shape[1::2], bool)                      # force empty mask (drop the spurious one)
            blanked += 1
        else:
            m = binary_fill_holes(np.asarray(mz[0, :, s, :]) > 0)
        mz[0, :, s, :] = m.astype(np.uint8)
    # Rebuild ONLY the (1-channel, cheap) mask pyramid: after this step the mask is the reliable present-
    # signal (spurious full masks on black slices are gone), and interpolate_missing reads mask L2 for its
    # present-check. The raw is untouched (still jump-corrected + unmasked) - crop_and_mask reads raw L0.
    zarrio.rebuild_pyramid(mg, "stride")
    print(f"finalised brain mask on {nz} slices ({blanked} black/missing -> empty mask; raw left unmasked)")


def filter_components_3d(cfg, frac=0.10, slab=64):
    """Drop 3D connected components smaller than `frac` x the largest, in the brain-mask stack.

    WHY THIS EXISTS. Disconnected false-positive components - wax, tray, label card - are the
    dominant error mode of every 2D mask model measured on this cohort, the incumbent included.
    An oracle that deletes them is worth +0.0063 median Dice cohort-wide and **+0.0550 on MR273**,
    which is larger than anything four rounds of retraining achieved. Nothing in the pipeline
    addressed them: normalize_masks and remove_background only hole-fill and zero black slices.

    WHY 3D AND NOT 2D. Per-slice largest-component filtering was measured and is destructive: at
    coronal levels where the hemispheres separate in-plane it deletes one of them (MR243 p05
    0.9530 -> 0.6404). In 3D that objection disappears - the hemispheres are joined through the
    callosum and the caudal slices, so the brain is ONE 3D component, while a wax blob on a single
    slice touches nothing above or below it. Persistence across slices is the signal that separates
    them, and it exists only in 3D.

    WHY frac=0.10. Swept over all 17 curated subjects (code/qc_cc3d_sweep.py). frac=0.10 is
    indistinguishable from strict top-1 (both recover 91.5% of the oracle head-room) but leaves a
    genuinely detached fragment >=10% of brain volume able to survive, which top-1 would delete -
    free safety. Dropping to 1% collects only 10.1%, because MR273's spurious mass sits BETWEEN 1%
    and 10% of brain volume, so the threshold cannot be lowered "to be safe" without losing the
    benefit. Worst harm across 17/17 subjects: -0.0000.

    CROP BOX, stated carefully. `_crop_geometry` takes the bounding box of the union over slices - a
    max-statistic that ONE blob on ONE slice sets for the whole subject - so removing blobs can
    shrink it, invisibly to any per-slice metric. BUT the large shrinks reported by the earlier
    unguarded sweep (qc_cc3d_sweep.py: MR256 0.747x, MF282 0.893x) are partly an ARTEFACT of exactly
    the damage the guard above prevents: emptying the outermost slices shrinks a union box, which
    looks like the filter working and is actually brain being deleted. Measured on MR257: 0.714x
    unguarded vs 1.000x guarded - the entire apparent gain was the artefact. Treat crop-box shrink
    as a possible bonus, not as a justification; the justification is the Dice head-room
    (+0.0063 cohort, +0.0550 on MR273) and those were measured per-slice, where the artefact
    cannot flatter them.

    Runs between `interpolate_missing` and `crop_and_mask`: after interpolation so the filled slices
    are filtered too, and immediately before the crop geometry it protects. The interpolated marker
    (label 2) is preserved per slice, exactly as edit_niftis does after its morphology.

    MEMORY (2026-09-26): the in-memory version (_fc3d_inmem, kept as the reference) labels the whole full-resolution mask at
    once - int32 labels for ~1.8e10 voxels - and peaked at 237 GB on MR263, exhausting a shared host's RAM and swap. This is
    the SAME algorithm, slab by slab (_fc3d_chunked): identical kept voxels (verified against _fc3d_inmem), ~9 GB per slab.
    """
    mg = zarr.open_group(cfg.mask_stack, mode="r+"); mz = mg["0"]     # (1, rows, slices, cols)
    nz = mz.shape[2]

    def read(s0, s1):
        return np.asarray(mz[0, :, s0:s1, :])                          # labels: 1 real, 2 interpolated

    def write(s0, s1, out):
        mz[0, :, s0:s1, :] = out

    st = _fc3d_chunked(read, write, nz, frac=frac, slab=slab)
    if st.get("empty"):
        print("filter_components_3d: empty mask - nothing to do"); return
    if st["n"] <= 1:
        print(f"filter_components_3d: {st['n']} component, nothing to drop"); return
    zarrio.rebuild_pyramid(mg, "stride")
    b0, b1 = st["box_before"], st["box_after"]
    a0, a1 = max(b0[0] * b0[1], 1), max(b1[0] * b1[1], 1)
    print(f"filter_components_3d: {st['n']} components -> kept {st['kept']} "
          f"({100 * st['kept_frac']:.2f}% of volume); "
          f"dropped {st['dropped_vox']:,} vox; rescued {st['rescued']} slice(s) that would have been "
          f"emptied; crop box {b0[0]}x{b0[1]} -> {b1[0]}x{b1[1]} ({a1 / a0:.3f}x area) [chunked, slab {slab}]")


def _box2(pr):
    """(rows, cols) extent of a (rows, cols) any-projection."""
    r, c = np.where(pr.any(1))[0], np.where(pr.any(0))[0]
    return (int(r[-1] - r[0] + 1), int(c[-1] - c[0] + 1)) if len(r) and len(c) else (0, 0)


def _fc3d_rescue(M, K):
    """The guard: never EMPTY a slice that had brain - fall back to that slice's largest 2D component (in place on K)."""
    from scipy.ndimage import label as cclabel2d
    had, has = M.any(axis=(0, 2)), K.any(axis=(0, 2))
    rescued = np.where(had & ~has)[0]
    for sl in rescued:
        l2, n2c = cclabel2d(M[:, sl, :])
        if n2c:
            sz2 = np.bincount(l2.ravel())[1:]
            K[:, sl, :] = (l2 == (1 + int(np.argmax(sz2))))
    return len(rescued)


def _fc3d_chunked(read, write, nz, frac=0.10, slab=64):
    """Memory-bounded, IDENTICAL filter_components_3d. With 6-connectivity (scipy's default 3D structure) the only edges
    between two slabs cut along the slice axis join a voxel to the SAME (row, col) in the next slice, so: pass 1 labels each
    slab, records each label's size and the (last slice of slab k, first slice of slab k+1) foreground overlaps as label pairs;
    graph connected components over those pairs = the true 3D components and their sizes; the keep rule on them gives a LUT
    over the slab-local labels. Pass 2 re-labels each slab (deterministic), applies the LUT, the per-slice rescue and the
    label-2 restore, and writes. read(s0, s1) -> (rows, s1 - s0, cols) labels; write(s0, s1, uint8)."""
    from scipy.ndimage import label as cclabel
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    starts = list(range(0, nz, slab))
    bases, sizes, pairs = [], [], []
    base, prev_last, box_m = 0, None, None
    total_m = 0
    for s0 in starts:
        s1 = min(nz, s0 + slab)
        M = read(s0, s1) > 0
        total_m += int(M.sum())
        pr = M.any(axis=1); box_m = pr if box_m is None else (box_m | pr)
        cc, n = cclabel(M)
        del M
        bases.append(base)
        sizes.append(np.bincount(cc.ravel(), minlength=n + 1)[1:].astype(np.int64))
        first = np.where(cc[:, 0, :] > 0, cc[:, 0, :].astype(np.int64) + base, 0)
        last = np.where(cc[:, -1, :] > 0, cc[:, -1, :].astype(np.int64) + base, 0)
        if prev_last is not None:
            both = (prev_last > 0) & (first > 0)
            if both.any():
                pairs.append(np.unique(np.stack([prev_last[both], first[both]], 1), axis=0))
        prev_last = last
        base += n
        del cc
    N = base
    if N == 0:
        return dict(empty=True, n=0)
    size_all = np.concatenate(sizes)                                   # label i (1..N) -> size_all[i - 1]
    P = np.concatenate(pairs) if pairs else np.zeros((0, 2), np.int64)
    g = coo_matrix((np.ones(len(P), np.int8), (P[:, 0], P[:, 1])), shape=(N + 1, N + 1))
    _, comp = connected_components(g, directed=False)
    uniq, inv = np.unique(comp[1:], return_inverse=True)               # the 3D components of labels 1..N
    n3d = len(uniq)
    if n3d <= 1:
        return dict(n=int(n3d))
    csize = np.bincount(inv, weights=size_all.astype(np.float64))     # exact (integers < 2**53)
    keep_c = csize >= frac * csize.max()
    lut = np.zeros(N + 1, bool)
    lut[1:] = keep_c[inv]
    box_k, rescued, total_k = None, 0, 0
    for k, s0 in enumerate(starts):
        s1 = min(nz, s0 + slab)
        lab = read(s0, s1)
        M = lab > 0
        cc, _ = cclabel(M)
        cc[cc > 0] += bases[k]
        K = lut[cc]
        del cc
        rescued += _fc3d_rescue(M, K)
        out = K.astype(np.uint8)
        interp = np.where((lab == 2).any(axis=(0, 2)))[0]
        out[:, interp, :] = out[:, interp, :] * 2
        total_k += int(K.sum())
        pr = K.any(axis=1); box_k = pr if box_k is None else (box_k | pr)
        write(s0, s1, out)
    return dict(n=int(n3d), kept=int(keep_c.sum()), kept_frac=float(csize[keep_c].sum() / csize.sum()), rescued=int(rescued),
                dropped_vox=int(total_m - total_k), box_before=_box2(box_m), box_after=_box2(box_k))


def _sdt(m):
    """Signed distance to the mask boundary: >0 outside, <0 inside. Interpolating this between two
    slices morphs one shape into the other (smoother than blending the binary masks)."""
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~m) - distance_transform_edt(m)


def interpolate_missing(cfg):
    """Fill the missing interior slices ANNOTATED in the spreadsheet (cfg.missing): each (before, after)
    is a bracket of present slices, so slices before+1..after-1 were dropped and are reconstructed from
    the nearest PRESENT slice on either side - image by a distance-weighted FULL-FOV blend (background
    intact, so bg has a real tray on interpolated slices), mask label by signed-distance morph (_sdt).
    Trusts the manual annotation: it does NOT invent gaps from the data. A raw-
    black scan cross-checks and only WARNS about any interior all-black run cfg.missing didn't cover (so
    a data/annotation issue surfaces rather than being silently guessed). No-op when cfg.missing is empty
    (e.g. MR243). 'present' = has a brain mask: remove_background just ran, so it force-zeroed the mask on
    every black/missing slice (removing a network's spurious full mask on black), leaving the mask L2 the
    reliable present-signal - and the one working pyramid still rebuilt (raw L2 is left stale, unread)."""
    rg = zarr.open_group(cfg.raw_stack, mode="r+"); rz = rg["0"]
    mg = zarr.open_group(cfg.mask_stack, mode="r+"); mz = mg["0"]
    mzc = mg["2"]                                     # coarse mask (fresh from remove_background) -> present-check
    nz = rz.shape[2]
    present = np.array([s for s in range(nz) if int(np.asarray(mzc[:, :, s, :]).max()) > 0])
    if len(present) < 2:
        print("too few present slices to interpolate"); return
    pset = set(present.tolist())

    def nearest(s, before):                          # nearest present slice on the given side of s
        side = present[present < s] if before else present[present > s]
        return int(side.max()) if before and len(side) else (int(side.min()) if len(side) else None)

    gaps = []                                        # (dropped slices, bracket a, bracket b) per cfg.missing run
    for b0, b1 in sorted(cfg.missing):
        if b0 < 0:                                   # leading-drop entry (-1, k): non-brain frames excluded at
            continue                                 # ingest (name card/grid) - not an interior gap to fill
        inner = [s for s in range(b0 + 1, b1) if s not in pset]
        if inner:
            gaps.append((inner, nearest(b0 + 1, True), nearest(b1 - 1, False)))
    annotated = set(s for inner, _, _ in gaps for s in inner)
    unannot = [s for s in range(int(present[0]) + 1, int(present[-1])) if s not in pset and s not in annotated]
    # Interior black slices the annotation missed (e.g. a corrupt raw photo cfg.missing bracketed
    # off-by-one). Fill SHORT isolated runs (<=3) from the nearest present neighbours - a black slice
    # bracketed by brain is unambiguously a gap - and WARN about longer runs (a real data issue to
    # surface, never silently invent a large stretch of anatomy).
    long_black, run = [], []
    for s in unannot + [None]:                            # sentinel flushes the last run
        if run and s == run[-1] + 1:
            run.append(s); continue
        if run:
            (gaps.append((run, nearest(run[0], True), nearest(run[-1], False)))
             if len(run) <= 3 else long_black.extend(run))
        run = [] if s is None else [s]
    if long_black:
        print(f"WARNING: {len(long_black)} interior black slice(s) in long unannotated run(s), left black: "
              f"{long_black[:12]}{'...' if len(long_black) > 12 else ''}")
    total = 0
    for inner, a, b in gaps:
        if a is None or b is None:
            print(f"  skip gap {inner[0]}..{inner[-1]} (no present bracket on one side)"); continue
        ma, mb = np.asarray(mz[0, :, a, :]) > 0, np.asarray(mz[0, :, b, :]) > 0
        sa, sb = _sdt(ma), _sdt(mb)
        # blend the FULL (unmasked) bracket slices across the whole FOV, so interpolated slices carry a real
        # background like every other slice. A zeroed background (the old brain-only fill) left them out-of-
        # distribution for a segmentation model and depended on the imperfect mask; the background twin (bg)
        # reads this raw unmasked, so it now gets a proper tray. crop_and_mask still re-masks 02_blockface.
        ra = np.asarray(rz[:, :, a, :]).astype(np.float32)
        rb = np.asarray(rz[:, :, b, :]).astype(np.float32)
        for s in inner:
            t = (s - a) / (b - a)
            mi = ((1 - t) * sa + t * sb) < 0          # morphed brain shape at this depth (drives the mask label)
            rz[:, :, s, :] = ((1 - t) * ra + t * rb).astype(np.uint8)   # full-FOV blend, background intact
            mz[0, :, s, :] = mi.astype(np.uint8) * 2  # LABEL 2 = interpolated (real brain = 1) -> identifiable
            total += 1
    if not total:
        print("no annotated interior missing slices to fill"); return
    # L0-only edit (no pyramid rebuild): this is the last in-place step, and crop_and_mask consumes L0
    # directly. The stale working L1/L2 (never read again) are dropped with _work; the final blockface
    # builds a fresh pyramid from the filled L0. Removes the last two full-stack rebuilds of stage 3.
    print(f"interpolated {total} missing slice(s) across {len(gaps)} cfg.missing gap(s)")


def _crop_geometry(cfg):
    """The crop box shared by crop_and_mask and blockface_bg so the masked blockface, its mask and the
    background-intact view sit on IDENTICAL grids: the tissue bounding box over the KEPT slices (caudal
    cutoff applied), centre-padded to a multiple of CROP_ALIGN. Returns a dict of box params + the kept
    label volume mlbl (1=real brain, 2=interpolated)."""
    mlbl = np.asarray(zarr.open_group(cfg.mask_stack, mode="r")["0"])[0]  # (rows, slices, cols) labels
    H, nz0, W = mlbl.shape
    keep = _last_slice(cfg, nz0) + 1                                     # caudal cutoff: drop trailing slices
    mlbl = mlbl[:, :keep, :]
    proj = (mlbl > 0).any(axis=1)                                        # tissue footprint over the kept slices
    rows, cols = np.where(proj.any(1))[0], np.where(proj.any(0))[0]
    t0, b0, l0, r0 = int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)
    m = int(round(zarrio.CROP_MARGIN_MM * 1000.0 / cfg.vx))              # tolerance for mask under-segmentation
    t0, b0, l0, r0 = max(0, t0 - m), min(H, b0 + m), max(0, l0 - m), min(W, r0 + m)
    h, w = b0 - t0, r0 - l0
    Hp = zarrio.ceildiv(h, zarrio.CROP_ALIGN) * zarrio.CROP_ALIGN
    Wp = zarrio.ceildiv(w, zarrio.CROP_ALIGN) * zarrio.CROP_ALIGN
    oy, ox = (Hp - h) // 2, (Wp - w) // 2
    # cropped voxel I <- raw voxel t0-oy+I: the same world x/z as the uncropped stacks so all niftis overlay.
    g = dict(keep=keep, nz0=nz0, t0=t0, b0=b0, l0=l0, r0=r0, h=h, w=w, Hp=Hp, Wp=Wp,
             oy=oy, ox=ox, z_off=int(t0 - oy), x_off=int(l0 - ox))
    return g, mlbl


def _src_y_anterior0(cfg):
    """World y (mm) of slice 0 in this subject's source stack - what a derived store must inherit so its A-P
    direction matches its parent (see zarrio._slice_axis). None if the source predates the 2026-09-19 fix."""
    for p in (cfg.raw_stack, cfg.blockface, cfg.blockface_bg):
        try:
            _, A = zarrio.read_level(p, 2)
        except Exception:
            continue
        if A[1, 1] < 0:
            return float(A[1, 3])
    return None


def crop_and_mask(cfg):
    """Crop the (jump-corrected, gap-filled) raw stack to the tissue bounding box, apply the hole-filled
    mask, and centre-pad to a multiple of CROP_ALIGN -> 02_blockface + 02_blockface_mask. Pure geometry;
    cross-slice intensity is a SEPARATE step (harmonize_intensity)."""
    from scipy.ndimage import binary_fill_holes
    bf = zarr.open_group(cfg.raw_stack, mode="r")["0"]                    # (3, rows, slices, cols), unmasked
    g, mlbl = _crop_geometry(cfg)
    t0, b0, l0, r0, oy, ox, h, w = g["t0"], g["b0"], g["l0"], g["r0"], g["oy"], g["ox"], g["h"], g["w"]

    bz = zarrio.new_group(cfg.blockface)
    a0 = zarrio.create_level(bz, 0, [3, g["Hp"], g["keep"], g["Wp"]])
    mz = zarrio.new_group(cfg.blockface_mask)
    m0 = zarrio.create_level(mz, 0, [1, g["Hp"], g["keep"], g["Wp"]])
    for s in range(g["keep"]):
        lbl = mlbl[t0:b0, s, l0:r0]
        m = binary_fill_holes(lbl > 0)                                  # fill interior holes before masking
        label = 2 if (lbl == 2).any() else 1                           # carry the interpolated marker (2) forward
        a0[:, oy:oy + h, s, ox:ox + w] = (np.asarray(bf[:, t0:b0, s, l0:r0]) * m[None]).astype(np.uint8)
        m0[0, oy:oy + h, s, ox:ox + w] = (m * label).astype(np.uint8)
    zarrio.add_pyramid(mz, "stride")
    y0 = _src_y_anterior0(cfg)                     # A-P: inherit the source stack's slice-0 world y
    zarrio.write_metadata(mz, cfg.vx, cfg.vz, "stride", z_off=g["z_off"], x_off=g["x_off"], y_anterior0=y0)
    zarrio.add_pyramid(bz, cfg.reduce)
    zarrio.write_metadata(bz, cfg.vx, cfg.vz, cfg.reduce, z_off=g["z_off"], x_off=g["x_off"], y_anterior0=y0)
    cut = f", caudal cutoff at {g['keep'] - 1} (dropped {g['nz0'] - g['keep']})" if g["keep"] < g["nz0"] else ""
    print(f"cropped+masked blockface: {h}x{w} -> padded {g['Hp']}x{g['Wp']} "
          f"(offset z={g['z_off']} x={g['x_off']}), {g['keep']} slices{cut}")


def blockface_bg(cfg, level=2):
    """Background-INTACT twin of 02_blockface (`02_blockface_bg`): the SAME cropped, jump-corrected grid
    but WITHOUT masking, so the aligned brain sits in its visible tray. It's the easy space to hand-edit
    02_blockface_mask (brain aligned + homogeneous intensity, vs the discontinuous native photos). Built
    DIRECTLY at L2 (in-plane /4) as a SINGLE array - it's only ever viewed/edited at L2/L4, so never worth
    a full-res copy (that L0 store, shaded+harmonised at L0, was the batch-killing I/O). Reads the unmasked
    jump-corrected raw (remove_background no longer zeroes it); correct_shading + harmonize_intensity then
    process this single level with the SAME per-slice fit as the masked blockface (its array == the masked
    L2, so the brain matches exactly, only the background differs). Transient - dropped after export."""
    bf = zarr.open_group(cfg.raw_stack, mode="r")["0"]
    g, _ = _crop_geometry(cfg)
    t0, b0, l0, r0, oy, ox, h, w, Hp, Wp, keep = (g["t0"], g["b0"], g["l0"], g["r0"], g["oy"], g["ox"],
                                                  g["h"], g["w"], g["Hp"], g["Wp"], g["keep"])
    f = 2 ** level                                                   # L<level> = /2**level in-plane
    # Always crop from L0 (raw_stack['0']) and block-mean by f, so any level shares the SAME padded-crop
    # lattice the curated mask was built on (reading raw_stack['level'] could shift an odd t0 half a voxel).
    bz = zarrio.new_group(cfg.blockface_bg)
    a2 = zarrio.create_level(bz, 0, [3, Hp // f, keep, Wp // f])      # single array, at L<level> resolution
    for s in range(keep):
        buf = np.zeros((3, Hp, Wp), np.uint8)
        buf[:, oy:oy + h, ox:ox + w] = np.asarray(bf[:, t0:b0, s, l0:r0])           # UNMASKED crop, L0 frame
        a2[:, :, s, :] = np.rint(buf.reshape(3, Hp // f, f, Wp // f, f).mean((2, 4))).astype(np.uint8)
    zarrio.write_metadata_level(bz, cfg.vx, cfg.vz, level=level, reduce=cfg.reduce, z_off=g["z_off"],
                                x_off=g["x_off"], y_anterior0=_src_y_anterior0(cfg))
    print(f"wrote background-intact blockface at L{level} ({keep} slices, {Hp // f}x{Wp // f})")


def harmonize_intensity(cfg, trim=5, smooth=0):
    """Cross-slice intensity harmonisation (its own modular step). Rescale each slice by a SINGLE
    grayscale GAIN so its robust brain luminance (the trimmed mean of the mean-over-channels brain,
    dropping the outer `trim`%) matches a reference -> uniform brightness across the stack. Key design
    choices, each fixing a failure
    mode of the old consecutive-overlap composition and of naive normalisation:
      - one gain for ALL channels  -> colour balance preserved (per-channel gains explode into colour
        artefacts on low-contrast slices, e.g. an all-gray frontal-pole slice turning green);
      - multiplicative gain (no 2-landmark stretch) -> unusual-composition slices are never distorted;
      - each slice matched INDEPENDENTLY to the reference -> no chaining, so no accumulated drift;
      - `smooth`=0: reference is one GLOBAL luminance (uniform stack). smooth>0: a robust median-smoothed
        reference over `smooth` slices, preserving slow anatomical brightness (anterior gray vs posterior
        white) while still removing per-slice exposure noise.
    Robust: tiny-brain slices get their level interpolated from neighbours; the gain is clipped (NOT
    smoothed across slices - see below). Operates on 02_blockface in place (fields measured at pyramid level L2, applied at level L0)."""
    from scipy.ndimage import median_filter
    bz = zarr.open_group(cfg.blockface, mode="r+")
    a0 = bz["0"]
    lum = np.asarray(bz["2"]).astype(np.float32).mean(0)                 # (rows, slices, cols) grayscale
    m2 = np.asarray(zarr.open_group(cfg.blockface_mask, mode="r")["2"])[0] > 0
    C, H, nz, W = a0.shape
    level = np.full(nz, np.nan)                                          # per-slice robust luminance
    for i in range(nz):
        mk = m2[:, i, :]
        if mk.sum() >= 100:
            d = lum[:, i, :][mk]
            lo, hi = np.percentile(d, [trim, 100 - trim])
            level[i] = d[(d >= lo) & (d <= hi)].mean()                   # robust trimmed mean
    ok = ~np.isnan(level)
    if ok.sum() < 2:
        print("too little brain to harmonise"); return
    level = np.interp(np.arange(nz), np.where(ok)[0], level[ok])         # fill tiny/empty slices
    ref = np.full(nz, np.median(level)) if smooth <= 0 else median_filter(level, size=int(smooth), mode="nearest")
    # gain = ref/level directly (each reliable slice self-normalises to the reference -> flat). NOT
    # median-smoothed across slices: smoothing would give an outlier-level slice its neighbours' gain,
    # so it wouldn't reach the reference -> the residual per-slice "intensity jump" spikes. Tiny-brain
    # slices are already handled by interpolating their level above.
    gain = np.clip(ref / level, 0.4, 2.5)
    # A per-slice scalar gain commutes with the (linear) mean-downsampling, so scale EVERY pyramid level
    # directly by the same gain rather than rebuilding the pyramid from L0 - avoids a full re-read/write.
    # Applied to the masked blockface AND (if present) its background-intact twin, with the SAME gains.
    for store in [cfg.blockface] + ([cfg.blockface_bg] if os.path.exists(cfg.blockface_bg) else []):
        bs = zarr.open_group(store, mode="r+")
        for lvl in [l for l in range(zarrio.NLEVELS) if str(l) in bs]:  # each store's ACTUAL levels (bg: one)
            a = bs[str(lvl)]
            for i in range(nz):
                a[:, :, i, :] = np.clip(np.asarray(a[:, :, i, :]).astype(np.float32) * gain[i], 0, 255).astype(np.uint8)
    print(f"harmonised intensity (grayscale gain, trim={trim}, smooth={smooth}): "
          f"ref={np.median(ref):.1f}, gain {gain.min():.2f}..{gain.max():.2f} ({nz} slices)")


def _poly_terms(yn, xn, deg):
    """2D monomials y^dy * x^dx for dy+dx <= deg (yn, xn scalars or broadcastable arrays)."""
    return [(yn ** dy) * (xn ** dx) for dy in range(deg + 1) for dx in range(deg + 1 - dy)]


def correct_shading(cfg, deg=2):
    """In-plane illumination (shading) correction: remove the one-sided lighting gradient (the dorsal-
    ventral 'shadow') so tissue reflectance is uniform WITHIN each section. Per slice, fit a degree-`deg`
    2D polynomial to the LOG luminance inside the brain - a smooth surface that captures the illumination
    gradient but not the (higher-frequency) anatomy - and divide every channel by exp(poly) normalised to
    mean 1. ONE factor for all channels, so colour balance + within-slice contrast are preserved (like
    harmonize); multiplicative, so no distortion. Per-2D: adapts to each section's shape. Runs on
    02_blockface in place, BEFORE harmonize_intensity (so the cross-slice gain sees the de-shadowed brain)
    and before histo->blockface matching (consistent intensity helps the registration). The polynomial is
    fit at L2 in normalised coords, so it re-evaluates at every pyramid level with no rebuild."""
    bz = zarr.open_group(cfg.blockface, mode="r+")
    C, H, nz, W = bz["0"].shape
    m2 = np.asarray(zarr.open_group(cfg.blockface_mask, mode="r")["2"])[0] > 0
    lum2 = np.asarray(bz["2"]).astype(np.float32).mean(0)                 # (rows2, slices, cols2)
    H2, W2 = m2.shape[0], m2.shape[2]
    Y2, X2 = np.mgrid[0:H2, 0:W2]
    terms2 = _poly_terms(Y2 / H2 - 0.5, X2 / W2 - 0.5, deg)               # basis on the L2 grid (reused)
    fields2 = {}                                                         # per-slice mean-1 illumination field at L2
    for i in range(nz):
        mk = m2[:, i, :]
        if mk.sum() < 200:
            continue
        ys, xs = np.where(mk)
        A = np.stack(_poly_terms(ys / H2 - 0.5, xs / W2 - 0.5, deg), 1)
        c = np.linalg.lstsq(A, np.log(np.clip(lum2[:, i, :][mk], 1, None)), rcond=None)[0]
        f = np.exp(sum(c[k] * terms2[k] for k in range(len(c))))
        fields2[i] = (f / f[mk].mean()).astype(np.float32)               # mean-1 within brain -> preserves level
    # Apply the SAME fitted field to the masked blockface AND (if present) the background-intact twin, so
    # their brains stay identical - only the visible background of 02_blockface_bg is additionally corrected.
    for store in [cfg.blockface] + ([cfg.blockface_bg] if os.path.exists(cfg.blockface_bg) else []):
        bs = zarr.open_group(store, mode="r+")
        for lvl in [l for l in range(zarrio.NLEVELS) if str(l) in bs]:   # each store's ACTUAL levels
            a = bs[str(lvl)]; _, HL, _, WL = a.shape                     # (bg has one array, at L2 == H2,W2)
            for i, f2 in fields2.items():
                f = f2 if (HL, WL) == (H2, W2) else cv2.resize(f2, (WL, HL), interpolation=cv2.INTER_LINEAR)
                img = np.asarray(a[:, :, i, :]).astype(np.float32)
                a[:, :, i, :] = np.clip(np.where(img > 0, img / f[None], 0), 0, 255).astype(np.uint8)
    print(f"corrected shading (per-slice deg{deg} illumination field): {len(fields2)} slices")


def _smooth_mask_nii(path, radius=1):
    """Gentle 3D smoothing of a label mask nifti IN PLACE (clean up the ragged periphery a little), keeping
    label 2 (interpolated slices) identifiable. Morphological closing then opening with a small 6-connected
    ball fills 1-voxel notches and shaves 1-voxel spurs without changing the overall shape. The per-slice
    label is preserved: a slice that had ANY interpolated (label-2) voxel keeps label 2, else 1."""
    import nibabel as nib
    from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure
    img = nib.load(path)
    d = np.asarray(img.dataobj)
    lab = np.squeeze(d).astype(np.uint8)                     # (rows, slices, cols) labels 0/1/2
    st = generate_binary_structure(3, 1)                     # 6-connected ball = gentlest 3D smoother
    sm = binary_opening(binary_closing(lab > 0, st, iterations=radius), st, iterations=radius)
    out = sm.astype(np.uint8)
    interp = np.where((lab == 2).any(axis=(0, 2)))[0]        # slices that were interpolated -> label 2
    out[:, interp, :] = out[:, interp, :] * 2
    nib.save(nib.Nifti1Image(out.reshape(d.shape), img.affine, img.header), path)
    print(f"  smoothed mask {os.path.basename(path)} (3D radius {radius}, {len(interp)} interpolated slices kept as 2)")


def _downsample_nii(src, dst, factor=4, kind="image"):
    """Write an L4 (=/16) nifti from an L2 (=/4) one by reducing the two IN-PLANE axes by `factor` while
    keeping the SLICE (through-plane) axis at full resolution. The slice axis is found from the affine - the
    two spatial axes with equal voxel size are in-plane, the odd one is the slice axis - so the slice/z axis
    is provably never downsampled, whatever the nifti axis order. image -> block mean; mask -> block max
    (keeps labels 1/2). Diagonal affine -> L4 = L2 with the in-plane scales x`factor` and the origin shifted
    to the new block centres, so L2 and L4 overlay exactly in fsleyes. Returns the two in-plane axes."""
    import nibabel as nib
    img = nib.load(src)
    d = np.asarray(img.dataobj)
    A = img.affine.copy()
    scales = np.linalg.norm(A[:3, :3], axis=0)              # voxel size of spatial axes 0,1,2
    inplane = min([(0, 1), (0, 2), (1, 2)], key=lambda p: abs(scales[p[0]] - scales[p[1]]))  # equal pair
    op = (lambda x, ax: x.max(ax)) if kind == "mask" else (lambda x, ax: np.rint(x.astype(np.float32).mean(ax)))
    for ax in inplane:                                      # reduce the two in-plane axes; slice axis kept
        n2 = d.shape[ax] - d.shape[ax] % factor
        d = np.take(d, np.arange(n2), axis=ax)
        sh = list(d.shape); sh[ax:ax + 1] = [n2 // factor, factor]
        d = op(d.reshape(sh), ax + 1)
    for ax in inplane:
        A[:3, 3] += A[:3, ax] * (factor - 1) / 2.0          # new voxel centre = old block centre
        A[:3, ax] *= factor
    nib.save(nib.Nifti1Image(d.astype(img.get_data_dtype()), A, img.header), dst)
    return inplane


def _squeeze_nii(path):
    """Rewrite a nifti with its degenerate NON-spatial singleton dims removed: (x,y,z,1,3) -> (x,y,z,3) for
    RGB, (x,y,z,1,1) -> (x,y,z) for a mask. fsleyes only renders RGB when the colour axis is the 4th
    (needs the dropped `1`); freeview handles either. Only axes >=3 that are length 1 are squeezed, never a
    spatial axis. In place (same filename)."""
    import nibabel as nib
    im = nib.load(path)
    d = np.asarray(im.dataobj)
    drop = tuple(i for i in range(3, d.ndim) if d.shape[i] == 1)
    if drop:
        nib.save(nib.Nifti1Image(np.squeeze(d, axis=drop), im.affine), path)
    return bool(drop)


def edit_niftis(cfg):
    """Finish the mask-editing exports (run after the L2 niftis are written by zarr2nii): gently 3D-smooth
    the mask, derive the L4 (/16) twins of the masked blockface, the background-intact blockface and the
    mask - so both a light (L4) and a full (L2) version are ready for fsleyes - and squeeze the degenerate
    4th dim off every one (RGB -> x,y,z,3) so fsleyes renders them. Drops the transient bg zarr."""
    _smooth_mask_nii(cfg.snap("02_blockface_mask_L2"))
    import nibabel as nib
    for base, kind in [("02_blockface", "image"), ("02_blockface_bg", "image"), ("02_blockface_mask", "mask")]:
        inplane = _downsample_nii(cfg.snap(base + "_L2"), cfg.snap(base + "_L4"), 4, kind)
        slice_ax = ({0, 1, 2} - set(inplane)).pop()
        n2 = nib.load(cfg.snap(base + "_L2")).shape[slice_ax]
        n4 = nib.load(cfg.snap(base + "_L4")).shape[slice_ax]
        print(f"  wrote {base}_L4 from {base}_L2 ({kind}); in-plane axes {inplane} /4, "
              f"slice axis {slice_ax} KEPT ({n2}=={n4})")
    for name in ("02_blockface_L2", "02_blockface_L4", "02_blockface_bg_L2", "02_blockface_bg_L4",
                 "02_blockface_mask_L2", "02_blockface_mask_L4"):
        _squeeze_nii(cfg.snap(name))                        # (x,y,z,1,3)->(x,y,z,3) so fsleyes shows RGB
    print("  squeezed the degenerate 4th dim off all L2/L4 niftis (fsleyes-RGB friendly)")
    import shutil
    if os.path.isdir(cfg.blockface_bg):
        shutil.rmtree(cfg.blockface_bg)                     # transient: only the bg niftis are kept
        print("  dropped transient 02_blockface_bg.ome.zarr")


def _squeeze_export(p):
    """The squeeze code/run.py `_export` ran after every zarr2nii: drop the length-1 axes >= 3 (RGB (x,y,z,1,3) -> (x,y,z,3))."""
    import nibabel as nib
    im = nib.load(p); d = np.asarray(im.dataobj)
    dr = tuple(i for i in range(3, d.ndim) if d.shape[i] == 1)
    nib.save(nib.Nifti1Image(np.squeeze(d, axis=dr) if dr else d, im.affine), p)


def export_stage3(cfg):
    """The stage-3 exports (code/run.py stage 3): L2 NIfTIs of the masked stack, of the background-intact twin (a single L2
    array, its level 0) and of the mask, then edit_niftis (mask smoothing, the L4 twins, the transient twin dropped)."""
    os.makedirs(cfg.nii_dir, exist_ok=True)
    for level, store, name in ((2, cfg.blockface, "02_blockface_L2"), (0, cfg.blockface_bg, "02_blockface_bg_L2"),
                               (2, cfg.blockface_mask, "02_blockface_mask_L2")):
        external.zarr2nii(level, store, cfg.snap(name))
        _squeeze_export(cfg.snap(name))
    edit_niftis(cfg)
