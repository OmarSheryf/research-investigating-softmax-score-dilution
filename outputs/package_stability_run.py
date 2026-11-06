"""Package the stabilization study with checksums and baseline dependencies."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root=Path(__file__).resolve().parent
out=root/'stability-run'
sources=[root/name for name in ['causal_transformer_experiment.py','stabilize_causal_training.py','analyze_stability.py','verify_stability_checkpoints.py','resume_stability.py','package_stability_run.py']]
baselines=[root/f'causal-run/checkpoints/seed-{seed}-step-1200.pt' for seed in [11,33,44]]
files=sources+baselines+sorted(p for p in out.rglob('*') if p.is_file() and p.name!='manifest.json')
manifest=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive=root/'training-stability-experiment.zip'
with ZipFile(archive,'w',ZIP_DEFLATED) as bundle:
    for file in files+[out/'manifest.json']:
        bundle.write(file,file.relative_to(root).as_posix())
with ZipFile(archive) as bundle:
    assert bundle.testzip() is None
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1)))
