"""Pasta & Co. bread ordering optimization — cleaned-up version.

Uses the empirical 80% service-level model (not Poisson or Monte Carlo).

Important assumptions:
- Daily historical deliveries: 3 Rustic, 8 Demi, 1 Como.
- A date not appearing in the waste log had zero waste for these breads.
- One suspected duplicate set of September 30 waste entries is excluded.
- Observed sales are used as demand in the baseline run, even on sell-out days.
  Extra-demand scenarios (+1 and +2) are sensitivity tests, not observed facts.

Waste totals and results should be checked against the original store records.
"""

import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. HISTORICAL DATA AND CURRENT ORDER QUANTITIES
# --------------------------------------------------

start = "2026-09-13"
end = "2026-10-13"

# Daily order quantities confirmed by the manager
Q0 = {
    "Rustic": 3,
    "Demi": 8,
    "Como": 1
}

# Wholesale cost per unit (used only to report waste cost)
COST = {
    "Rustic": 2.49,
    "Demi": 0.97,
    "Como": 3.45
}

# Waste quantities transcribed from the photographs.
# Each tuple is (Rustic, Demi, Como).
#
# Missing dates are assumed to have zero bread waste.
# The possible duplicate on September 30 is excluded.

observed_waste = {
    "2026-09-13": (2, 0, 1),
    "2026-09-14": (0, 2, 1),
    "2026-09-15": (2, 0, 1),
    "2026-09-16": (2, 0, 0),
    "2026-09-17": (1, 0, 0),
    "2026-09-18": (0, 1, 1),
    "2026-09-19": (2, 0, 0),
    "2026-09-20": (3, 0, 0),
    "2026-09-23": (1, 6, 0),
    "2026-09-24": (0, 0, 1),
    "2026-09-27": (0, 0, 1),
    "2026-09-28": (2, 1, 0),
    "2026-09-29": (3, 5, 1),
    "2026-09-30": (2, 2, 0),
    "2026-10-01": (2, 5, 1),
    "2026-10-06": (0, 1, 0),
    "2026-10-07": (2, 3, 1),
    "2026-10-08": (0, 2, 0),
    "2026-10-09": (2, 0, 0),
    "2026-10-11": (0, 5, 0),
    "2026-10-12": (3, 2, 1),
    "2026-10-13": (3, 0, 1),
}

# --------------------------------------------------
# 2. CREATE THE DAILY DATASET
# --------------------------------------------------

rows = []

for date in pd.date_range(start, end):

    counts = observed_waste.get(
        str(date.date()), (0, 0, 0)
    )

    for bread, wasted in zip(Q0, counts):

        rows.append({
            "date": date,
            "weekday": date.day_name(),
            "bread": bread,
            "current_order": Q0[bread],
            "waste": wasted,
            "estimated_sales": Q0[bread] - wasted,
            "wholesale_cost": COST[bread]
        })

df = pd.DataFrame(rows)

# --------------------------------------------------
# 3. FIND THE MINIMUM ORDER MEETING 80% OF DEMAND
# --------------------------------------------------


def minimum_order(values, service=0.80):
    """Smallest integer quantity covering >= service share of observations."""

    values = sorted(values)

    # Position of the empirical 80th percentile
    position = int(np.ceil(service * len(values))) - 1

    return int(values[position])

# --------------------------------------------------
# 4. OPTIMIZE THE THREE ORDERING STRATEGIES
# --------------------------------------------------

def make_policies(sellout_extra=0, service=0.80):
    data = df.copy()

    # Baseline assumption: demand equals observed sales.
    # Sensitivity analysis: add 1 or 2 units on sold-out days.
    data["assumed_demand"] = (
        data["estimated_sales"]
        + np.where(data["waste"] == 0, sellout_extra, 0)
    )

    # STRATEGY 2: FIXED OPTIMIZATION
    # One optimized quantity per bread type, every day.
    fixed = (
        data.groupby("bread")["assumed_demand"]
        .apply(lambda x: minimum_order(x, service))
        .to_dict()
    )

    # STRATEGY 3: DAY-OF-WEEK OPTIMIZATION
    # One quantity for every bread and weekday.
    by_day = (
        data.groupby(["bread", "weekday"])["assumed_demand"]
        .apply(lambda x: minimum_order(x, service))
        .to_dict()
    )

    # STRATEGY 1: CURRENT ORDERS
    data["current_Q"] = data["current_order"]

    # STRATEGY 2: FIXED OPTIMAL ORDERS
    data["fixed_Q"] = data["bread"].map(fixed)

    # STRATEGY 3: WEEKDAY OPTIMAL ORDERS
    data["weekday_Q"] = [
        by_day[(bread, weekday)]
        for bread, weekday in zip(data["bread"], data["weekday"])
    ]

    # --------------------------------------------------
    # 5. EVALUATE EACH STRATEGY
    # --------------------------------------------------

    def metrics(data, quantity_column):
        Q = data[quantity_column]
        D = data["assumed_demand"]

        # Waste = max(Q - D, 0)
        waste = np.maximum(Q - D, 0)

        # Unmet demand = max(D - Q, 0)
        unmet = np.maximum(D - Q, 0)

        return {
            "wasted_units": int(waste.sum()),
            "waste_cost": float(
                (waste * data["wholesale_cost"]).sum()
            ),
            # Share of all bread-day observations covered by order:
            "service_level_pct": float(
                np.mean(D <= Q) * 100
            ),
            "unmet_units": int(unmet.sum())
        }

    results = pd.DataFrame({
        "Current": metrics(data, "current_Q"),
        "Fixed optimized": metrics(data, "fixed_Q"),
        "Weekday optimized": metrics(data, "weekday_Q")
    }).T

    return fixed, by_day, results


# --------------------------------------------------
# 6. RUN THE MODEL
# --------------------------------------------------

if __name__ == "__main__":
    # First: No additional demand on sold-out days
    fixed, weekday, results = make_policies(sellout_extra=0)

    print("FIXED OPTIMAL QUANTITIES:")
    print(fixed)

    print("\nWEEKDAY OPTIMAL QUANTITIES:")

    schedule = pd.Series(weekday).unstack(0)
    schedule = schedule.reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ])
    print(schedule)

    print("\nSTRATEGY COMPARISON:")
    print(results.round(2))

    # --------------------------------------------------
    # 7. SENSITIVITY ANALYSIS
    # --------------------------------------------------

    for extra in [0, 1, 2]:
        fixed, weekday, results = make_policies(
            sellout_extra=extra
        )
        print(f"\nADDITIONAL DEMAND ON SELLOUTS: {extra}")
        print(results.round(2))
