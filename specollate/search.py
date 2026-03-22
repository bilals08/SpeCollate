import json
import os
import tempfile
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from typing import Optional, Dict, Any

import torch
import pandas as pd

## filter warnings
import warnings
warnings.filterwarnings("ignore")

from .src.snapconfig import config
from .src.snaptrain import model as snap_model
from .src.snapsearch import pepdataset, specdataset, dbsearch, postprocess, preprocess


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "proteorift"


def get_specollate_model_path(cache_dir: Optional[str] = None) -> str:
    """Return path to cached Specollate model or download from HF Hub."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # If already cached somewhere under the cache dir, return that
    matches = list(cache_dir.rglob("specollate_model_weights.pt"))
    if matches:
        return str(matches[0])

    repo_default_id = "SaeedLab"
    print("Downloading Specollate model from HuggingFace Hub...")
    specollate_config_path = hf_hub_download(
        repo_id=f"{repo_default_id}/SpeCollate",
        filename="config.json",
        cache_dir=cache_dir,
        repo_type="model"
    )
    ## load file name from config
    with open(specollate_config_path, 'r') as f:
        config = json.load(f)
    model_filename = config.get("model_name", "specollate_model_weights.pt")

    specollate_path = hf_hub_download(
        repo_id=f"{repo_default_id}/SpeCollate",
        filename=model_filename,
        cache_dir=cache_dir,
        repo_type="model"
    )
    print(f"Specollate model downloaded to {specollate_path}")
    return specollate_path


def load_sample_data(cache_dir: Optional[str] = None, use_preprocessed: bool = True) -> Dict[str, str]:
    """Download sample dataset from HuggingFace and return paths.

    Returns a dict with either 'prep_path' and 'peptide_db' (if preprocessed available)
    or 'mgf_dir' and 'peptide_db' for raw files.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading sample data from HuggingFace...")
    repo_path = snapshot_download(repo_id="SaeedLab/sample-data-msms-search", repo_type="dataset", cache_dir=cache_dir)
    repo_path = Path(repo_path)

    if use_preprocessed:
        prep_path = repo_path / "preprocessed"
        peptide_db = repo_path / "raw" / "peptide_database"
        if prep_path.exists() and (prep_path / "specs.pkl").exists():
            print(f"Using preprocessed data: {prep_path}")
            return {"prep_path": str(prep_path), "peptide_db": str(peptide_db)}
        else:
            print("Preprocessed data not found; falling back to raw MGF files")
            use_preprocessed = False

    mgf_dir = repo_path / "raw" / "spectra"
    peptide_db = repo_path / "raw" / "peptide_database"
    return {"mgf_dir": str(mgf_dir), "peptide_db": str(peptide_db)}


class SpeCollateSearch:
    """Simple programmatic interface to run SpeCollate search."""

    def __init__(self, device: str = "auto", precursor_tolerance: float = 7, precursor_tolerance_type: str = "ppm", charge: int = 5):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        # store configuration values for use when rendering config files
        self.precursor_tolerance = precursor_tolerance
        self.precursor_tolerance_type = precursor_tolerance_type
        self.charge = charge

    def _create_config_file(self, mgf_dir, prep_dir, pep_dir, out_dir, model_name=None):
        contents = (
            "[search]\n"
            f"mgf_dir = {mgf_dir}\n"
            f"prep_dir = {prep_dir}\n"
            f"pep_dir = {pep_dir}\n"
            f"out_pin_dir = {out_dir}\n"
            f"model_name = {model_name or ''}\n"
            "spec_batch_size = 16384\n"
            "pep_batch_size = 16384\n"
            "search_spec_batch_size = 256\n"
            "keep_psms = 5\n"
            f"precursor_tolerance = {self.precursor_tolerance}\n"
            f"precursor_tolerance_type = {self.precursor_tolerance_type}\n"
            f"charge = {self.charge}\n"
            "\n"
            "[input]\n"
            "spec_size = 80000\n"
            "charge = 5\n"
            "use_mods = False\n"
            "\n"
            "[ml]\n"
            "batch_size = 1024\n"
            "pep_seq_len = 36\n"
            "dropout = 0.3\n"
        )
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False)
        tf.write(contents)
        tf.close()
        return tf.name

    def _preprocess_spectra(self, mgf_dir, prep_dir):
        # Create a temporary config file (so snapconfig can read `input` section)
        cfg = self._create_config_file(mgf_dir, prep_dir, "", "", model_name=None)
        # Point the snapconfig module at the new config
        config.PARAM_PATH = cfg
        try:
            # preprocess.mgfs expects (mgf_dir, prep_dir)
            preprocess.preprocess_mgfs(mgf_dir, prep_dir)
        finally:
            try:
                os.unlink(cfg)
            except Exception:
                pass

    def search(
        self,
        mgf_dir: str,
        peptide_db: str,
        output_dir: str = "./specollate_output",
        prep_dir: Optional[str] = None,
        use_cached_prep: bool = False,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run preprocessing (if needed) and database search.

        Returns paths to generated `target.pin` and `decoy.pin`.
        """
        os.makedirs(output_dir, exist_ok=True)

        cleanup_prep = False
        if prep_dir is None:
            prep_dir = tempfile.mkdtemp(prefix="specollate_prep_")
            cleanup_prep = True
        else:
            os.makedirs(prep_dir, exist_ok=True)

        prep_file = Path(prep_dir) / "specs.pkl"
        if not use_cached_prep or not prep_file.exists():
            self._preprocess_spectra(mgf_dir, prep_dir)

        # Create a minimal config and point the snapconfig module at it
        # If no model path provided, attempt to download/get cached Specollate weights
        if model_path is None:
            try:
                model_path = get_specollate_model_path()
            except Exception as e:
                print("Warning: failed to download Specollate model:", e)
                model_path = None

        cfg_file = self._create_config_file(mgf_dir, prep_dir, peptide_db, output_dir, model_path)
        config.PARAM_PATH = cfg_file

        # Build datasets and loaders
        spec_ds = specdataset.SpectralDataset(prep_dir)
        pep_ds = pepdataset.PeptideDataset(peptide_db)
        dec_ds = pepdataset.PeptideDataset(peptide_db, decoy=True)

        spec_batch_size = config.get_config(key="spec_batch_size", section="search")
        pep_batch_size = config.get_config(key="pep_batch_size", section="search")
        search_spec_batch_size = config.get_config(key="search_spec_batch_size", section="search")

        spec_loader = torch.utils.data.DataLoader(
            dataset=spec_ds, batch_size=spec_batch_size, collate_fn=dbsearch.spec_collate
        )

        pep_loader = torch.utils.data.DataLoader(
            dataset=pep_ds, batch_size=pep_batch_size, collate_fn=dbsearch.pep_collate
        )

        dec_loader = torch.utils.data.DataLoader(
            dataset=dec_ds, batch_size=pep_batch_size, collate_fn=dbsearch.pep_collate
        )

        # Load model
        vocab_size = 30
        embedding_dim = 512
        hidden_lstm_dim = 512
        lstm_layers = 2
        net = snap_model.Net(vocab_size=vocab_size, embedding_dim=embedding_dim,
                             hidden_lstm_dim=hidden_lstm_dim, lstm_layers=lstm_layers).to(self.device)
        model_name = config.get_config(key="model_name", section="search")
        if model_path:
            model_name = model_path
        if model_name:
            state = torch.load(model_name, map_location=self.device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                sd = state['model_state_dict']
            elif isinstance(state, dict):
                sd = state
            else:
                raise RuntimeError("Loaded model file does not contain a state dict")

            cleaned = {}
            for k, v in sd.items():
                if k.startswith('module.'):
                    cleaned[k[len('module.'):]] = v
                else:
                    cleaned[k] = v

            net.load_state_dict(cleaned)

        # runModel expects model in eval mode
        net.eval()

        # Run embeddings
        e_specs = dbsearch.runModel(spec_loader, net, "specs", self.device)
        e_peps = dbsearch.runModel(pep_loader, net, "peps", self.device)
        e_decs = dbsearch.runModel(dec_loader, net, "peps", self.device)

        # Run search
        search_loader = torch.utils.data.DataLoader(dataset=e_specs, batch_size=search_spec_batch_size, shuffle=False)
        datasets = {"spec_dataset": spec_ds, "pep_dataset": pep_ds, "dec_dataset": dec_ds}
        embeddings = {"e_specs": e_specs, "e_peps": e_peps, "e_decs": e_decs}

        pep_inds, pep_vals, dec_inds, dec_vals = dbsearch.search(search_loader, datasets, embeddings, self.device)

        # Write percolator files
        pin_charge = config.get_config(section="search", key="charge")
        charge_cols = [f"charge-{ch+1}" for ch in range(pin_charge)]
        cols = ["SpecId", "Label", "ScanNr", "SNAP", "ExpMass", "CalcMass", "deltCn", "deltLCn"] + charge_cols + ["dM", "absdM", "enzInt", "PepLen", "Peptide", "Proteins"]

        global_out = postprocess.generate_percolator_input(pep_inds, pep_vals, pep_ds, spec_ds, "target")
        df = pd.DataFrame(global_out, columns=cols)
        df.sort_values(by="SNAP", inplace=True, ascending=False)
        df.to_csv(os.path.join(output_dir, "target.pin"), sep="\t", index=False)

        global_out = postprocess.generate_percolator_input(dec_inds, dec_vals, dec_ds, spec_ds, "decoy")
        df = pd.DataFrame(global_out, columns=cols)
        df.sort_values(by="SNAP", inplace=True, ascending=False)
        df.to_csv(os.path.join(output_dir, "decoy.pin"), sep="\t", index=False)

        # cleanup
        try:
            os.unlink(cfg_file)
        except Exception:
            pass
        if cleanup_prep:
            shutil.rmtree(prep_dir)
        
        print(f"Search completed. Outputs written to {output_dir}")

        return {
            "output_dir": output_dir,
            "target_pin": os.path.join(output_dir, "target.pin"),
            "decoy_pin": os.path.join(output_dir, "decoy.pin"),
        }

    def search_with_sample_data(self, output_dir: str = "./specollate_output", use_preprocessed: bool = True, cache_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run search using the sample dataset hosted on HuggingFace.

        This will download the sample dataset (preprocessed if available) and run
        the same `search` pipeline, reusing preprocessed data when present.
        """
        print("Loading sample data from HuggingFace...")
        sample_data = load_sample_data(cache_dir=cache_dir, use_preprocessed=use_preprocessed)

        # If preprocessed data available, use it directly by pointing `prep_dir`
        if 'prep_path' in sample_data:
            prep_path = sample_data['prep_path']
            peptide_db = sample_data['peptide_db']
            # Ensure output dir exists
            os.makedirs(output_dir, exist_ok=True)
            # Use existing search method but skip preprocessing
            self.search(mgf_dir="", peptide_db=peptide_db, output_dir=output_dir, prep_dir=prep_path, use_cached_prep=True)
        else:
            # Raw MGF files: run full search (will preprocess)
            self.search(mgf_dir=sample_data['mgf_dir'], peptide_db=sample_data['peptide_db'], output_dir=output_dir)
