"""Shared benchmark definitions used by translation, evaluation, and reporting."""

SC_SUBSETS = (
    "add_obj",
    "add_att",
    "replace_obj",
    "replace_att",
    "replace_rel",
    "swap_obj",
    "swap_att",
)
SCPP_SUBSETS = (
    "swap_obj",
    "swap_att",
    "replace_obj",
    "replace_att",
    "replace_rel",
)

BENCHMARK_SUBSETS = {"sc": SC_SUBSETS, "scpp": SCPP_SUBSETS}
CAPTION_FIELDS = {
    "sc": ("caption", "negative_caption"),
    "scpp": ("caption", "negative_caption", "caption2"),
}
EVALUATION_TO_BENCHMARK = {
    "sugarcrepe": "sc",
    "sugarcrepe_pp": "scpp",
}
BENCHMARK_NAMES = {"sc": "SugarCrepe", "scpp": "SugarCrepe++"}
