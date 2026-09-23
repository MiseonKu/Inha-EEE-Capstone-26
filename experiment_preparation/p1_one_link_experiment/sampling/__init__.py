"""Workspace candidate generation and target sampling."""

__all__ = ["generate_candidates", "generate_trial_schedule", "sample_uniform_targets"]


def __getattr__(name: str):
    """Load pandas-backed helpers only when the caller asks for them."""

    if name == "generate_trial_schedule":
        from .balanced_shuffle import generate_trial_schedule

        return generate_trial_schedule
    if name == "generate_candidates":
        from .candidate_generator import generate_candidates

        return generate_candidates
    if name == "sample_uniform_targets":
        from .uniform_bin_sampler import sample_uniform_targets

        return sample_uniform_targets
    raise AttributeError(name)
