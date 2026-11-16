"""Installation checks; does not mislabel CPU import tests as GPU qualification."""
import argparse
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    args = parser.parse_args()
    import torch
    from .loading import load_draft
    from .runtime.dneo_production import activate_production_runtime
    profile = activate_production_runtime("h100", verify_sources=True)
    report = dict(package_version=importlib.metadata.version("dneo-inference"),
                  profile=profile["profile_id"], source_integrity="PASS",
                  gpu_available=torch.cuda.is_available(), gpu_qualification="NOT_RUN")
    if args.model:
        model = load_draft(args.model)
        report.update(model_load="PASS", tensors=len(model.state_dict()), layers=model.config.target_layer_ids)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
