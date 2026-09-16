# Connectome data (not committed)

The large connectome files are git-ignored (they're big and freely available).
Fetch them into this `data/` folder before running:

## 1. Connectome connectivity + neuron list (ships in the vendored model repo)

From [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
(FlyWire v783 — the whole-brain LIF model this project builds on):

- `Connectivity_783.parquet`  (~100 MB — edges, synapse counts, neurotransmitter sign)
- `Completeness_783.csv`      (~3 MB — the neuron list; index = FlyWire root id)

```bash
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git /tmp/dbm
cp /tmp/dbm/Connectivity_783.parquet /tmp/dbm/Completeness_783.csv data/
```

## 2. FlyWire cell-type annotations (Schlegel et al. 2024)

`neuron_annotations.tsv` (~32 MB) — labels each neuron (Kenyon_Cell, MBON, PAM,
PPL1, APL, ALPN, `top_nt`, `side`, …), keyed to the same v783 root ids.

```bash
curl -sL https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv \
  -o data/neuron_annotations.tsv
```

After these three files are in `data/`, everything under `scripts/` runs.
See the top-level `README.md` for the science and credits.
