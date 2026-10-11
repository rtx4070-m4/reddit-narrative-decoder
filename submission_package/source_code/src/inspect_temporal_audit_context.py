from pathlib import Path
import json

root = Path(".")
targets = [
    root / "src/audit_full_corpus_phase7_vs_phase8.py",
    root / "src/audit_phase7b_phase8_root_cause_v2.py",
    root / "src/audit_phase7e_vs_phase8.py",
    root / "results/tables/phase7o_temporal_narratives/phase7o_manifest.json",
    root / "results/tables/phase7o_temporal_narratives_v17/phase7o_manifest.json",
    root / "results/tables/phase7o_temporal_narratives/phase7o_join_audit.csv",
    root / "results/tables/phase7o_temporal_narratives_v17/phase7o_join_audit.csv",
]

for path in targets:
    print("\n" + "=" * 90)
    print("FILE:", path)
    print("=" * 90)

    if not path.exists():
        print("MISSING")
        continue

    if path.suffix == ".json":
        try:
            print(json.dumps(json.loads(path.read_text(encoding="utf-8-sig")),
                             indent=2, ensure_ascii=False))
        except Exception as exc:
            print("JSON READ ERROR:", exc)
    elif path.suffix == ".py":
        print(path.read_text(encoding="utf-8-sig"))
    else:
        print(path.read_text(encoding="utf-8-sig", errors="replace"))
