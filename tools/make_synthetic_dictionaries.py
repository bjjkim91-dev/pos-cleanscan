"""Generate the synthetic dictionaries shipped with this repo.

The real project used canonical value lists derived from a beverage company's product
master data, which cannot be published. This script writes small stand-in dictionaries in
the same format (`<ATTR>_value,ngram_token,alias_token`) from a hand-written vocabulary of
fictional drinks (Nuka-Cola, Slurm, Brawndo and friends), so every pipeline stage runs end
to end on `data/sample/` without touching anything real.

Run:  python tools/make_synthetic_dictionaries.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ngram_builder import build_ngram_fingerprint  # noqa: E402

OUT = ROOT / "dictionaries"

# value -> aliases (abbreviations retailers use)
VOCAB = {
    "BRAND": {
        "NUKA-COLA": ["NUKA COLA", "NUKACOLA", "NUKA", "NC"],
        "SLURM": ["SLRM"],
        "BRAWNDO": ["BRWNDO", "BRAWNDO CORP"],
        "SUNSET SARSAPARILLA": ["SUNSET SARS", "SUNSET"],
        "BUZZ COLA": ["BUZZ"],
        "VIM": [],
    },
    "FLAVOUR": {
        "ORIGINAL": ["CLASSIC", "REGULAR", "REG"], "COLA": [], "CHERRY": ["CHRY"], "ORANGE": ["ORNG", "ORG"], "GRAPE": ["GRP"],
        "LEMON LIME": ["LEMON-LIME", "LEM LIME", "LMN LM"], "BERRY": ["MIXED BERRY", "BRY"], "VANILLA": ["VAN"],
        "SARSAPARILLA": ["SARS"], "TROPICAL": ["TROP"], "LEMON": ["LEM"], "LIME": ["LM"], "TONIC WATER": ["TONIC"],
    },
    "CATEGORY": {
        "CARBONATED SOFT DRINKS": ["CSD", "SOFT DRINK", "SOFT DRINKS", "SODA", "POP"], "ENERGY": ["ENERGY DRINK", "ENERGY DRINKS"],
        "ISOTONICS": ["SPORTS DRINK", "SPORT DRINK", "SPORT", "ELECTROLYTE"], "WATER": ["WATER STILL", "STILL WATER"],
        "WATER SPARKLING": ["SPARKLING WATER"], "NON CARBONATED BEVERAGES": ["NCB", "ICED TEA", "RTD COFFEE"],
    },
    "SUB_BRAND": {
        "CLASSIC": [], "QUANTUM": ["QNTM"], "VICTORY": [], "WILD": [], "LOCO": [], "THIRST MUTILATOR": ["THIRST MUT", "MUTILATOR"],
        "EXTREME": ["XTREME"], "ZERO": ["ZERO SUGAR"],
    },
    "CALORIE": {"ZERO": ["DIET", "ZERO SUGAR", "SUGAR FREE"], "LOW": ["LIGHT"], "FULL": ["REGULAR"]},
    "PACK_CONTENT": {
        "CAN": ["CANS", "TIN", "CNS", "SLEEK"], "BOTTLE - PET": ["BTL", "BTTL", "PET", "BOTTLE"], "BOTTLE - GLASS": ["GLASS", "NRG GLASS"],
        "GABLE - TOP": ["GABLE", "CARTON"], "TETRA": ["TETRA PAK"], "POUCH": ["MIX"],
    },
    "SIZE_ML": {str(s): [] for s in [200, 250, 300, 330, 341, 355, 473, 500, 591, 710, 750, 946, 1000, 1500, 1750, 2000]},
    "COUNT": {str(c): [] for c in [1, 2, 4, 6, 8, 10, 12, 15, 18, 20, 24, 30, 36]},
    # Product names and groups follow the builder's format: [COMPANY] [CATEGORY] [PACK_SIZE] [CONTAINER] [VARIANT?]
    "PRODUCT_NAME": {n: [] for n in [
        "NUKA CSD 12 X 355 ML CAN CORE", "NUKA CSD 12 X 355 ML CAN FLAV", "NUKA CSD 2 L BTTL CORE", "NUKA CSD 2 L BTTL FLAV",
        "NUKA CSD 6 X 710 ML BTTL CORE", "NUKA CSD 591 ML BTTL", "NUKA CSD 24 X 355 ML CAN", "NUKA CSD 355 ML CAN",
        "SLURM 473 ML CAN", "SLURM 4 X 250 ML CAN", "SLURM 355 ML CAN", "SLURM 12 X 355 ML CAN",
        "BRAWNDO 946 ML BTTL", "BRAWNDO 710 ML BTTL", "BRAWNDO 8 X 591 ML BTTL", "BRAWNDO 591 ML BTTL",
        "SUNSET CSD 355 ML BTTL GLASS", "SUNSET CSD 6 X 355 ML BTTL GLASS", "BUZZ CSD 2 L BTTL", "BUZZ CSD 12 X 355 ML CAN", "VIM CSD 355 ML CAN",
    ]},
}
VOCAB["PRODUCT_GROUP"] = {n.replace(" CORE", "").replace(" FLAV", ""): [] for n in VOCAB["PRODUCT_NAME"]}


def write(attr: str, values: dict) -> int:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"dictionary_{attr}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"{attr}_value", "ngram_token", "alias_token"])
        for value, aliases in values.items():
            alias_tokens = sorted({t for a in aliases for t in build_ngram_fingerprint(a)})
            w.writerow([value, json.dumps(build_ngram_fingerprint(value)), json.dumps(alias_tokens)])
    return len(values)


if __name__ == "__main__":
    total = 0
    for attr, values in VOCAB.items():
        n = write(attr, values)
        total += n
        print(f"dictionary_{attr}.csv  {n:>4} values")
    print(f"{total} synthetic values written to {OUT}")
