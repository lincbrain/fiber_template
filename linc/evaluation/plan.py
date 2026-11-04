"""`--plan` for the evaluation entry points (reproduce/numbers/{variants_leave_self_out_rebuild_fa025, variability_rebuild_rb_final,
lobo_t2_rebuild, r060_target_comparison}.py): print what the script would read, write and need, and run nothing. The same idea as the
build drivers' --plan (reproduce/templates/_driver.py), for scripts that are one command each.

    if plan.wanted():
        return plan.show("lobo_t2_rebuild", env="env/linc.yml (PY_MRI)", est="...", inputs={...}, outputs=[...], needs=[...], notes=[...])
"""
import os
import sys
from pathlib import Path


def wanted(argv=None):
    return "--plan" in (sys.argv[1:] if argv is None else argv)


def _short(p):
    s = str(p)
    for var in ("LINC_PRODUCTS", "LINC_TEMPLATE", "LINC_DATA", "LINC_WORK", "LINC_NUMBERS"):
        v = os.environ.get(var, "").rstrip("/")
        if v and (s == v or s.startswith(v + "/")):
            return f"${var}" + s[len(v):]
    return s


def show(name, *, env, est, inputs, outputs, needs=(), notes=(), cmd=None):
    """inputs: {label: path or [paths]}; a path that does not exist is marked. outputs: paths. needs: external programs / variables."""
    print(f"{name}: plan (nothing run)\n  env:  {env}\n  time: {est}")
    if cmd:
        print(f"  cmd:  {cmd}")
    print("  reads:")
    for label, p in inputs.items():
        ps = p if isinstance(p, (list, tuple)) else [p]
        for q in ps:
            q = Path(q)
            mark = "" if q.exists() else "   [not found]"
            print(f"    {label:<22} {_short(q)}{mark}")
    print("  writes:")
    for p in outputs:
        print(f"    {_short(p)}")
    if needs:
        print("  needs: " + "; ".join(needs))
    for n in notes:
        print(f"  note: {n}")
    return 0
