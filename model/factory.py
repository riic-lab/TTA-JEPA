from model.tte_jepa import TTEJEPA, TTEJEPADownstream


def build_model(config, dimensions, schema):
    name = config.get("method", "tte_jepa")
    if name == "tte_jepa":
        return TTEJEPA(**dimensions, **config["model"])
    if name == "icarefm":
        from baselines.icarefm import ICareFM
        return ICareFM(dimensions, schema, **config["model"])
    if name == "hepa":
        from baselines.hepa import HEPA
        return HEPA(dimensions, **config["model"])
    if name == "ehrmamba":
        from baselines.ehrmamba import EHRMamba
        return EHRMamba(dimensions, **config["model"])
    raise ValueError(f"Unknown neural model: {name}")


def build_downstream(pretrained, config):
    if config.get("method", "tte_jepa") == "tte_jepa":
        return TTEJEPADownstream(pretrained, **config["downstream"])
    from baselines.readout import FrozenReadout
    return FrozenReadout(pretrained, config["method"], **config["downstream"])
