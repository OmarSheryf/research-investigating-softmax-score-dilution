"""Build a checksum manifest and a portable archive of the saved experiment."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root=Path(__file__).resolve().parent
run=root/'causal-run'
sources=[root/name for name in ['causal_transformer_experiment.py','build_causal_report.py','verify_causal_checkpoints.py','package_causal_run.py','causal-original-results.json']]
files=sources+sorted(p for p in run.rglob('*') if p.is_file() and p.name!='manifest.json')
manifest=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
(run/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
archive=root/'causal-transformer-experiment.zip'
with ZipFile(archive,'w',ZIP_DEFLATED) as bundle:
    for path in files+[run/'manifest.json']:
        bundle.write(path,path.relative_to(root).as_posix())
with ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    assert sum('/checkpoints/' in name for name in bundle.namelist())==30
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1,checkpoints=30)))
