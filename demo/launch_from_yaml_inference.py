#!/usr/bin/env python
"""
Read infer_jobs.yaml and run the batch-inference script
"""
import subprocess, pathlib, re, yaml, sys

cfg = yaml.safe_load(open("demo/infer_jobs.yaml"))
script    = pathlib.Path(cfg["script"]).resolve()
mix       = cfg["mix"]
out_root  = pathlib.Path(cfg["out_root"]).resolve()
out_root.mkdir(exist_ok=True)

pat_ti = re.compile(r"_TI(\d+)_")          # captures the number after TI
pat_ke = re.compile(r"_KE(\d+)_")          # captures the number after KE
pat_pd = re.compile(r"_PD([a-z0-9]+)_")    # e.g. _PDg1_, _PDks_

for model_path in cfg["models"]:
    model = pathlib.Path(model_path).resolve()

    m1     = pat_ti.search(model.stem)
    if not m1:
        sys.exit(f"✗  Cannot parse TI<number> in {model.name}")
    t_eval = int(m1.group(1))               # 32, 64, …

    m2     = pat_ke.search(model.stem)
    if not m2:
        sys.exit(f"✗  Cannot parse KE<number> in {model.name}")
    k_eval = int(m2.group(1))               # 1, 2, …

    # Try to parse the PDE tag from filename; fallback to YAML default.
    pd = pat_pd.search(model.stem)
    pde_tag = pd.group(1) if pd else cfg.get("default_pde","sch")

    outdir = out_root / (model.stem + "_run")
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = ["python", str(script),
           "--model", str(model),
           "--mix",   mix,
           "--pde",   pde_tag,
           "--time_sampling", str(t_eval),
           "--kernel_scale", str(k_eval),
           "--outdir", str(outdir)]
    #print("▶", " ".join(cmd))
    print(model_path)
    subprocess.run(cmd, check=True)
