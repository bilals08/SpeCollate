# SpeCollate: MS/MS Database Search Pipeline

Install:

```bash
pip install specollate
```

Quick usage:

```python
from specollate import SpeCollateSearch

searcher = SpeCollateSearch(device='cuda')  # or 'cpu'
searcher.search(
    mgf_dir='path/to/mgf_dir',
    peptide_db='path/to/peptide_db',
    output_dir='./specollate_output'
)

```

Sample search:

```python
from specollate import SpeCollateSearch

searcher = SpeCollateSearch(device='cuda')  # or 'cpu'
searcher.search_with_sample_data(output_dir='./specollate_output')

```

Citation:

Tariq MU, Saeed F. SpeCollate: Deep cross-modal similarity network for mass spectrometry data based peptide deductions. PLoS One. 2021 Oct 29;16(10):e0259349. doi: 10.1371/journal.pone.0259349. PMID: 34714871; PMCID: PMC8555789.
