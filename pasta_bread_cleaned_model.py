"""

Important assumptions:

- Daily historical deliveries are 3 Rustic, 8 Demi, 1 Como.
- A date not appearing in the waste log had zero waste for these breads.
- One suspected duplicate set of September 30 waste entries is excluded.
- Observed sales are used as demand in the baseline run, even on sell-out days.

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
# This is a dict of observed waste counts for each day in the historical period.
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

# These are merely houskeeping steps to create a dataframe with the following columns:
# - date: the date of the observation
# - weekday: the day of the week
# - bread: the type of bread (Rustic, Demi, Como)
# - current_order: the current order quantity for that bread
# - waste: the observed waste for that bread on that date
# - estimated_sales: the estimated sales for that bread on that date (current order minus waste)
# - wholesale_cost: the wholesale cost per unit for that bread
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

# This function returns the smallest integer quantity that covers at least the specified
# service amount, which is currently 80%. It takes the parameters of the observed
# sales and the service level and returns the minimum order quantity that
# meets the service level requirement.
def minimum_order(values, service=0.80):

    values = sorted(values)

    # Position of the 80th percentile
    position = int(np.ceil(service * len(values))) - 1

    return int(values[position])

# --------------------------------------------------
# 4. OPTIMIZE THE THREE ORDERING STRATEGIES
# --------------------------------------------------

# this function calculates the optimal order quantities for each of the three strategies:
# current orders, fixed optimization, and day-of-week optimization. It returns the fixed
# optimal quantities, weekday optimal quantities, and a comparison of the results for
# each strategy.
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

    # This function evaluates one ordering strategy. It returns its waste,
    # waste cost, service level, and unmet demand.
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


# This function resamples historical demand and tests all three strategies.
# It returns simulated performance and paired waste comparisons.
def monte_carlo(simulations=10000, seed=381, sellout_extra=0):
    fixed, by_day, _ = make_policies(sellout_extra=sellout_extra)
    data = df.copy()
    data["assumed_demand"] = (
        data["estimated_sales"]
        + np.where(data["waste"] == 0, sellout_extra, 0)
    )

    rng = np.random.default_rng(seed)
    simulated_demand = np.empty((simulations, len(data)), dtype=int)

    for column, row in data.reset_index(drop=True).iterrows():
        demand_pool = data.loc[
            (data["bread"] == row["bread"])
            & (data["weekday"] == row["weekday"]),
            "assumed_demand"
        ].to_numpy()
        simulated_demand[:, column] = rng.choice(
            demand_pool, size=simulations, replace=True
        )

    quantities = {
        "Current": data["current_order"].to_numpy(),
        "Fixed optimized": data["bread"].map(fixed).to_numpy(),
        "Weekday optimized": np.array([
            by_day[(bread, weekday)]
            for bread, weekday in zip(data["bread"], data["weekday"])
        ])
    }
    costs = data["wholesale_cost"].to_numpy()
    simulated = {}

    for strategy, quantity in quantities.items():
        waste = np.maximum(quantity - simulated_demand, 0)
        unmet = np.maximum(simulated_demand - quantity, 0)
        simulated[strategy] = {
            "waste": waste.sum(axis=1),
            "cost": (waste * costs).sum(axis=1),
            "unmet": unmet.sum(axis=1),
            "service": (simulated_demand <= quantity).mean(axis=1) * 100
        }

    summary_rows = []
    for strategy, values in simulated.items():
        summary_rows.append({
            "strategy": strategy,
            "mean_waste": values["waste"].mean(),
            "waste_95_low": np.quantile(values["waste"], 0.025),
            "waste_95_high": np.quantile(values["waste"], 0.975),
            "mean_waste_cost": values["cost"].mean(),
            "mean_service_pct": values["service"].mean(),
            "mean_unmet": values["unmet"].mean()
        })

    comparison_rows = []
    current_waste = simulated["Current"]["waste"]
    for strategy in ["Fixed optimized", "Weekday optimized"]:
        reduction = current_waste - simulated[strategy]["waste"]
        comparison_rows.append({
            "strategy": strategy,
            "mean_waste_reduction": reduction.mean(),
            "reduction_95_low": np.quantile(reduction, 0.025),
            "reduction_95_high": np.quantile(reduction, 0.975),
            "probability_less_waste": np.mean(reduction > 0)
        })

    return (
        pd.DataFrame(summary_rows).set_index("strategy"),
        pd.DataFrame(comparison_rows).set_index("strategy")
    )


if __name__ == "__main__":
    simulation_summary, simulation_tests = monte_carlo()
    print("\nMONTE CARLO RESULTS (10,000 RUNS):")
    print(simulation_summary.round(2).to_string())
    print("\nPAIRED WASTE COMPARISONS:")
    print(simulation_tests.round(3).to_string())
