# Audited non-speech descriptive evidence viewer

Open `descriptive_v1.html` directly in a browser. The 20 WAVs and compact V2
observations are embedded; the viewer makes no network calls. The evidence and
review links point to local sibling files for deeper inspection.

Rebuild from the frozen saved evidence:

```sh
python3 models/second_architecture/nonspeech_probe_v1/explorer/build_descriptive_v1.py
python3 models/second_architecture/nonspeech_probe_v1/explorer/build_descriptive_v1.py --check
python3 -m unittest discover -s models/second_architecture/nonspeech_probe_v1/explorer -p 'test_build_descriptive_v1.py'
```

The builder refuses a changed manifest, record file, case WAV hash, inventory,
silence repeat, or expected descriptive census. It loads no candidate models and
does not modify any raw evidence. `descriptive_v1_template.html` contains the
handwritten interface; `build_descriptive_v1.py` pins and extracts only the
fields displayed. The generated HTML is a standalone review artifact, not a
source of new causal evidence or a security finding.
