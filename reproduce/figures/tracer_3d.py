#!/usr/bin/env python3
"""The tracer pathways as a 3D scene in the MRI template (Fig. 10 a-c and the two supplementary videos).

You need (configs/products_mri.yaml)
  template_mri: the T2 channel and the brain mask; tracer_template_v2c and tracer_template_v2: the connected pathways, the bundle
  densities and the injection sites; env/viz3d.yml (pyvista on a headless VTK; PY_VIZ3D).

What it does
  draws the template's T2 on three planes (the midline, a coronal backdrop behind the pathways, a floor), each case's connected
  pathway as a smoothed surface in its color (the inferred joins in a lighter tint, the injection site opaque), and renders:
  oblique   from the front, above and the right
  lateral   from the right
  dorsal    from above
  glass     the lateral view with the brain's surface as a translucent shell instead of the planes
  and one turn about the vertical axis as a video (with --glass-video, a second one with the shell).

You get
  <out>_{oblique,lateral,dorsal,glass}.png (2400 x 1800), <out>_rotation.mp4 [+ <out>_rotation_glass.mp4] (10 s, 30 fps),
  <out>.provenance.json, and $LINC_NUMBERS/<record>.json (the planes, the window, the cases, the fraction of each pathway left of
  the midline). --for-composite renders the stills for Fig. 10 (no in-scene key or plane labels, letters at print size).

    $PY_VIZ3D reproduce/figures/tracer_3d.py [--out figures/tracer_3d] [--views ...] [--no-video] [--glass-video] [--for-composite]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import ndimage as ndi
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

pv.OFF_SCREEN = True
NAME = "fig_tracer_3d"
CASES = ["MF191", "MR243", "MR252"]                      # drawing order: MF8 first (largest), then MM1, MM2
LABEL = {"MR243": "MM1, area 10", "MR252": "MM2, SMA", "MF191": "MF8, area 13m (mirrored)"}
MIRROR = {"MF191"}
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as FS  # noqa: E402
COLOR = {S: FS.TRACER[S] for S in CASES}                  # the tracer cases' colors (figstyle.TRACER)
X_SAG, Y_BACK, Z_FLOOR = 0.0, -1.0, -1.5                 # mm: the mid-sagittal plane; the coronal backdrop; the axial floor
LO_PCT, HI_PCT, GAMMA, TOP, FLOOR_BASE = 5.0, 99.0, 0.85, 0.90, 0.10   # the planes' gray window; the floor's base gray outside the brain
FADE = 0.3                                                # a plane's opacity in the rotation frames where it would hide the pathways
GLASS_SMOOTH_MM, GLASS_OPACITY, GLASS_MIDLINE, GLASS_FLOOR = 1.0, 0.22, 0.2, 0.5   # the glass view: the shell's smoothing and opacity, the midline plane's, the floor's
GLASS_VIDEO_OPACITY = 0.15                                # the shell in the glass rotation
GLASS_COLOR = "#D0D0D0"                                   # the shell's gray (its back faces are culled, so the front surface alone blends over the pathways)
BG, FG = "black", "white"
WIN = (2400, 1800)


def lighten(hexcolor, f=0.55):
    r, g, b = (int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def load(p):
    im = nib.load(str(p)); A = im.affine
    assert np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3]))) and (np.diag(A)[:3] > 0).all(), "the template grid must be axis-aligned RAS"
    return np.asanyarray(im.dataobj), A


def surface(field, A, level=0.5, smooth_vox=0.6, taubin=20):
    """A smoothed closed surface of a scalar field (binary or graded, in [0, 1]) at `level`, in world mm."""
    if not (field > level).any():
        return None
    nz = np.argwhere(field > level); lo = np.maximum(nz.min(0) - 3, 0); hi = np.minimum(nz.max(0) + 4, field.shape)
    sub = field[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    sm = ndi.gaussian_filter(sub, smooth_vox) if smooth_vox else sub
    verts, faces, _, _ = measure.marching_cubes(sm, level=level, spacing=tuple(np.diag(A)[:3]))
    verts = verts + (lo * np.diag(A)[:3] + A[:3, 3])
    poly = pv.PolyData(verts, np.c_[np.full(len(faces), 3), faces].ravel())
    return poly.smooth_taubin(n_iter=taubin, pass_band=0.1) if taubin else poly


def textured_plane(gray, A, axis, pos_mm, keep):
    """The windowed T2 (gray in [0, 1]) on the plane at world position pos_mm along axis (0 = x, 1 = y, 2 = z), as an RGB texture,
    clipped to `keep` (world bounds xmin, xmax, ymin, ymax, zmin, zmax)."""
    grid = pv.ImageData(dimensions=np.array(gray.shape) + 1, spacing=np.diag(A)[:3], origin=A[:3, 3] - np.diag(A)[:3] / 2)
    grid.cell_data["rgb"] = np.repeat((np.clip(gray, 0, 1).ravel(order="F") * 255).astype(np.uint8)[:, None], 3, axis=1)
    origin = [0.0, 0.0, 0.0]; origin[axis] = pos_mm
    return grid.slice(normal=["x", "y", "z"][axis], origin=origin).clip_box(keep, invert=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output prefix (default figures/tracer_3d under the figures root)")
    ap.add_argument("--no-video", action="store_true"); ap.add_argument("--no-fade", action="store_true", help="keep every plane opaque in the rotation too")
    ap.add_argument("--back", type=float, default=Y_BACK); ap.add_argument("--floor", type=float, default=Z_FLOOR)
    ap.add_argument("--views", nargs="+", default=["oblique", "lateral", "dorsal", "glass"], choices=["oblique", "lateral", "dorsal", "glass"],
                    help="the stills to render (default: all four)")
    ap.add_argument("--glass-video", action="store_true", help="also a rotation with the brain shell (no backdrop) as <out>_rotation_glass.mp4")
    ap.add_argument("--record", default=NAME, help="the record's name under $LINC_NUMBERS (default fig_tracer_3d; the composite's stills use fig_tracer_3d_stills)")
    ap.add_argument("--for-composite", action="store_true",
                    help="stills for the paper's composite: no in-scene case key (the composite has one), no plane labels (the panel subtitles name the planes), "
                         "the orientation letters and the bar label at a size that prints at 80 mm panel width")
    a = ap.parse_args(argv); t0 = time.time()
    P = paths.products("products_mri"); TPL, V2, V2C = P["template_mri"], P["tracer_template_v2"], P["tracer_template_v2c"]
    prefix = Path(a.out) if a.out else Path("figures/tracer_3d")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    T2, A = load(TPL / "linc_mri_T2.nii.gz"); brain = load(TPL / "linc_mri_brainmask.nii.gz")[0] > 0
    T2 = np.where(brain, T2, 0).astype(np.float32); lo, hi = (float(v) for v in np.percentile(T2[brain], [LO_PCT, HI_PCT]))
    gray = np.where(brain, np.clip((T2 - lo) / (hi - lo), 0, 1) ** GAMMA * TOP, 0.0).astype(np.float32)
    rec = dict(planes=dict(sagittal_x_mm=X_SAG, sagittal_extent="the whole slice, opaque", backdrop_y_mm=a.back, floor_z_mm=a.floor,
                           floor_base_gray=FLOOR_BASE, capsule_plane="omitted (it would hide the posterior half of the pathways in the oblique view and the rotation)",
                           channel="T2", window=f"brain T2 percentiles {LO_PCT:g}-{HI_PCT:g} ({lo:.0f}-{hi:.0f}) to gray 0-{TOP}, gamma {GAMMA}",
                           rotation_fade=None if a.no_fade else FADE), cases={})
    inputs = [TPL / "linc_mri_T2.nii.gz", TPL / "linc_mri_brainmask.nii.gz"]
    meshes = []
    for S in CASES:
        C, _ = load(V2C / S / f"{S}_Bundles_mri-template_connected.nii.gz")
        D = np.clip(load(V2 / S / f"{S}_Bundles_mri-template_density.nii.gz")[0].astype(np.float32), 0, 1)
        I = load(V2 / S / f"{S}_Injection_site_mri-template_solid2.nii.gz")[0] > 0
        inputs += [V2C / S / f"{S}_Bundles_mri-template_connected.nii.gz", V2 / S / f"{S}_Bundles_mri-template_density.nii.gz",
                   V2 / S / f"{S}_Injection_site_mri-template_solid2.nii.gz"]
        if S in MIRROR:
            C, D, I = C[::-1], D[::-1], I[::-1]
        graded = np.where(C > 0, np.maximum(D, 0.55), 0.0).astype(np.float32)       # the density inside the connected label (joins at 0.55)
        path = surface(graded, A, level=0.5)
        bridge = surface((C == 2).astype(np.float32), A, smooth_vox=0.4, taubin=10)
        inj = surface(I.astype(np.float32), A, smooth_vox=0.5)
        meshes.append((S, path, bridge, inj))
        xw = A[0, 3] + np.arange(C.shape[0]) * A[0, 0]
        left = float((C > 0)[xw < X_SAG].sum() / max((C > 0).sum(), 1))
        rec["cases"][S] = dict(name=LABEL[S], mirrored=S in MIRROR, color=COLOR[S], pathway_vertices=int(path.n_points) if path else 0,
                               bridge_vertices=int(bridge.n_points) if bridge else 0, injection_vertices=int(inj.n_points) if inj else 0,
                               pathway_voxels=int((C > 0).sum()), bridge_voxels=int((C == 2).sum()), share_left_of_midline=round(left, 4))
    bx = brain.any((1, 2)).nonzero()[0]; by = brain.any((0, 2)).nonzero()[0]; bz = brain.any((0, 1)).nonzero()[0]
    W = lambda i, k: float(A[k, 3] + i * A[k, k])                                   # noqa: E731
    box = (W(bx[0], 0) - 1, W(bx[-1], 0) + 1, W(by[0], 1) - 1, W(by[-1], 1) + 1, W(bz[0], 2) - 1, W(bz[-1], 2) + 1)   # the brain's bounding box
    rec["planes"]["bounding_box_mm"] = [round(v, 2) for v in box]
    sag = textured_plane(gray, A, 0, X_SAG, box)
    back = textured_plane(gray, A, 1, a.back, box)
    floor = textured_plane(np.where(brain, gray, FLOOR_BASE).astype(np.float32), A, 2, a.floor, box)   # the slice on the dark base
    shell = surface(ndi.binary_fill_holes(brain).astype(np.float32), A, level=0.5, smooth_vox=GLASS_SMOOTH_MM / float(A[0, 0]), taubin=30).extract_largest()   # the brain's outer surface, smoothed
    rec["glass"] = dict(shell_vertices=int(shell.n_points), shell_smooth_mm=GLASS_SMOOTH_MM, shell_opacity=GLASS_OPACITY, midline_opacity=GLASS_MIDLINE,
                        floor=False, backdrop=False, what="the whole brain mask as a translucent shell, the pathways inside; the lateral camera")

    def scene(view, size=WIN, letters=(), zoom=1.25, key=True, glass=False, shell_opacity=None, bar="edge", with_floor=True, large=False):
        """The scene for one camera; glass = the brain's surface as a translucent shell with the pathways inside (the midline plane
        faint, no coronal backdrop, the floor dim) instead of the opaque planes."""
        p = pv.Plotter(off_screen=True, window_size=size); p.set_background(BG)
        if glass:
            actors = dict(sag=p.add_mesh(sag, scalars="rgb", rgb=True, lighting=False, opacity=GLASS_MIDLINE))
            if with_floor:
                actors["floor"] = p.add_mesh(floor, scalars="rgb", rgb=True, lighting=False, opacity=GLASS_FLOOR)
        else:                                                                     # the actors in the order of the paper's stills (the renders are byte-identical to them)
            actors = dict(sag=p.add_mesh(sag, scalars="rgb", rgb=True, lighting=False),
                          back=p.add_mesh(back, scalars="rgb", rgb=True, lighting=False),
                          floor=p.add_mesh(floor, scalars="rgb", rgb=True, lighting=False))
        for S, path, bridge, inj in meshes:
            if path is not None:
                p.add_mesh(path, color=COLOR[S], smooth_shading=True, specular=0.25, specular_power=25)
            if bridge is not None:
                p.add_mesh(bridge, color=lighten(COLOR[S]), smooth_shading=True, specular=0.2)
            if inj is not None:
                p.add_mesh(inj, color=COLOR[S], smooth_shading=True, ambient=0.3)
        if glass:                                                                 # after the opaque pathways: the front surface blends over them
            actors["shell"] = p.add_mesh(shell, color=GLASS_COLOR, opacity=shell_opacity if shell_opacity is not None else GLASS_OPACITY,
                                         smooth_shading=True, specular=0.15, specular_power=20, culling="back")
        fs_bar, fs_letter, lw = (48, 60, 10) if large else (26, 30, 6)
        if bar == "edge":                                                         # the 5 mm bar on the backdrop's ventral edge (the oblique and dorsal views)
            y0, z0 = a.back + 0.5, a.floor + 0.5
            p.add_mesh(pv.Line((2.0, y0, z0), (7.0, y0, z0)), color=FG, line_width=lw)
            p.add_point_labels([(4.5, y0, z0 + 2.0)], ["5 mm"], text_color=FG, font_size=fs_bar, shape=None, show_points=False, always_visible=True)
        else:                                                                     # the lateral views: along the A-P axis at the lower left, nearest the camera
            xb, yb, zb = box[1], box[2] + 3.0, box[4] + 2.0
            p.add_mesh(pv.Line((xb, yb, zb), (xb, yb + 5.0, zb)), color=FG, line_width=lw)
            p.add_point_labels([(xb, yb + 2.5, zb - 2.5)], ["5 mm"], text_color=FG, font_size=fs_bar, shape=None, show_points=False, always_visible=True)
        if not large:                                                             # the plane labels (the composite's subtitles name the planes instead)
            edge = [((X_SAG, box[3] - 6.0, box[5] - 1.5), "midline")] + ([((box[1] - 6.0, box[3] - 4.0, a.floor), "floor")] if with_floor else []) + \
                   ([((box[1] - 4.0, box[2] + 10.0, box[5] - 3.0), "brain surface")] if glass else [((box[1] - 8.0, a.back, box[5] - 1.5), f"y = {a.back:g} mm")])
            for pos, txt in edge:
                p.add_point_labels([pos], [txt], text_color="#BBBBBB", font_size=22, shape=None, show_points=False, always_visible=True)
        for pos, txt in letters:
            p.add_point_labels([pos], [txt], text_color=FG, font_size=fs_letter, shape=None, show_points=False, always_visible=True)
        if key:
            legend = p.add_legend([[LABEL[S], COLOR[S]] for S in ("MR243", "MF191", "MR252")], bcolor=None, border=False, face="rectangle")
            legend.SetPosition(0.70, 0.02); legend.SetPosition2(0.29, 0.11)   # by hand: pyvista's loc="lower right" lets the box run off the window, cutting the longest label
        p.camera_position = view; p.camera.zoom(zoom)
        return p, actors

    center = (9.0, 16.0, 14.0)
    views = {"oblique": [(95.0, 95.0, 85.0), center, (0.0, 0.0, 1.0)],       # anterior-superior-right
             "lateral": [(170.0, 16.0, 14.0), center, (0.0, 0.0, 1.0)],        # from the right
             "dorsal": [(9.0, 16.0, 160.0), center, (0.0, 1.0, 0.0)]}          # from above, anterior up
    SIDE_LETTERS = (((0.0, 45.0, 8.0), "A"), ((0.0, -24.5, 8.0), "P"), ((0.0, 8.0, 37.0), "S"), ((0.0, 8.0, -4.5), "I"), ((29.5, 16.0, 14.0), "R"))
    TOP_LETTERS = (((0.0, 45.0, 2.0), "A"), ((0.0, -24.5, 2.0), "P"), ((29.5, 10.0, 2.0), "R"), ((-29.5, 10.0, 2.0), "L"))
    views["glass"] = views["lateral"]                                           # the same camera as the lateral view, the brain as a shell
    LATERAL_LETTERS = tuple(l for l in SIDE_LETTERS if l[1] != "R")              # from the right, R points at the viewer: not drawn
    LETTERS = {"oblique": SIDE_LETTERS, "lateral": LATERAL_LETTERS, "dorsal": TOP_LETTERS, "glass": LATERAL_LETTERS}
    SETUP = {"oblique": dict(bar="edge"), "lateral": dict(bar="ap"), "dorsal": dict(bar="edge"), "glass": dict(glass=True, bar="ap", with_floor=False)}
    written = []
    if a.for_composite:
        rec["stills"] = dict(in_scene_key=False, plane_labels=False, letter_font_px=60, bar_font_px=48)
    for name in a.views:
        p, _ = scene(views[name], letters=LETTERS[name], key=not a.for_composite, large=a.for_composite, **SETUP[name])
        f = Path(f"{prefix}_{name}.png"); p.screenshot(str(f), transparent_background=False); p.close(); written.append(f)
        print(f"wrote {f} ({time.time() - t0:.0f} s)", flush=True)
    if not a.no_video:
        p, actors = scene(views["oblique"], size=(1600, 1200), letters=SIDE_LETTERS); f = Path(f"{prefix}_rotation.mp4")
        p.open_movie(str(f), framerate=30, quality=7); n = 300
        for i in range(n):
            p.camera.Azimuth(360.0 / n)                                    # VTK's Azimuth (pyvista's .azimuth is a property)
            if not a.no_fade:
                cx, cy = p.camera.position[0], p.camera.position[1]
                actors["sag"].GetProperty().SetOpacity(1.0 if cx >= X_SAG else FADE)
                actors["back"].GetProperty().SetOpacity(1.0 if cy >= a.back else FADE)
            p.write_frame()
        p.close(); written.append(f); print(f"wrote {f} ({time.time() - t0:.0f} s)", flush=True)
    if a.glass_video:
        p, actors = scene(views["oblique"], size=(1600, 1200), letters=SIDE_LETTERS, glass=True, shell_opacity=GLASS_VIDEO_OPACITY, with_floor=False); f = Path(f"{prefix}_rotation_glass.mp4")
        p.open_movie(str(f), framerate=30, quality=7); n = 300
        for i in range(n):
            p.camera.Azimuth(360.0 / n); p.write_frame()
        p.close(); written.append(f); print(f"wrote {f} ({time.time() - t0:.0f} s)", flush=True)
        rec["glass"]["video_shell_opacity"] = GLASS_VIDEO_OPACITY
    rec["environment"] = dict(pyvista=pv.__version__, vtk=str(pv.vtk_version_info))
    cmd = " ".join(["$PY_VIZ3D reproduce/figures/tracer_3d.py"] + (argv if argv is not None else sys.argv[1:]))
    wall = round(time.time() - t0, 1)
    prov = provenance.provenance_dict(str(prefix), inputs=inputs, outputs=written, cmd=cmd, wall_s=wall, extra=dict(figure=NAME, outputs=[str(w) for w in written]), md5=False)
    Path(f"{prefix}.provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    print(provenance.write_numbers(a.record, rec, inputs=inputs, md5=False, cmd=cmd))


if __name__ == "__main__":
    main()
