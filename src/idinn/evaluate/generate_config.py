import json
import itertools
import os

lr_values = [2, 3, 4]
ce_values = [5, 10, 20]
demand_values = [4, 8]
b_values = [495, 95]

configs = [
    {"lr": lr, "ce": ce, "b": b, "demand": demand}
    for lr, ce, b, demand in itertools.product(lr_values, ce_values, b_values, demand_values)
]

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sourcing_config.json")

with open(output_path, "w") as f:
    json.dump(configs, f, indent=4)

print(f"Generated {len(configs)} configs -> {output_path}")