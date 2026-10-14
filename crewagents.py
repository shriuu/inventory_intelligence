import csv
from pathlib import Path

from crewai import Agent, Crew, Process, Task

# find /Users/se/Downloads/bharat-inventory-intelligence \
#   -type f \
#   \( -name "*.csv" -o -name "*.xlsx" -o -name "*.json" \) \
#   -not -path "*/.git/*"
# cd /Users/se/Downloads/bharat-inventory-intelligence

# for f in data/raw/*.csv; do
#   echo ""
#   echo "========== $f =========="
#   head -n 3 "$f"
# done
# source .venv/bin/activate
# python crewagents.py
# ----------------------------------
# 1. Load actual inventory data
# ----------------------------------

INVENTORY_FILE = Path(
    "/Users/se/Downloads/bharat-inventory-intelligence"
    "/data/raw/inventory.csv"
)


def load_inventory():
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {INVENTORY_FILE}"
        )

    with INVENTORY_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "item_id",
            "item_name",
            "available_stock",
            "daily_demand",
            "lead_time_days",
            "reorder_point",
            "criticality_score",
        }

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing inventory columns: {sorted(missing)}"
            )

        records = list(reader)

    if not records:
        raise ValueError("Inventory CSV contains no records.")

    return records


# ----------------------------------
# 2. Calculate risk metrics in Python
# ----------------------------------

def prepare_inventory():
    records = load_inventory()
    prepared = []

    for row in records:
        try:
            stock = float(row["available_stock"])
            demand = float(row["daily_demand"])
            lead_time = float(row["lead_time_days"])
            reorder_point = float(row["reorder_point"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric inventory data for "
                f"{row.get('item_id', 'unknown item')}"
            ) from exc

        if stock < 0 or demand < 0 or lead_time < 0:
            raise ValueError(
                f"Negative stock, demand, or lead time for "
                f"{row['item_id']}"
            )

        days_of_supply = (
            stock / demand if demand > 0 else None
        )

        coverage_gap = (
            max(0, lead_time - days_of_supply)
            if days_of_supply is not None
            else None
        )

        prepared.append({
            "item_id": row["item_id"],
            "item_name": row["item_name"],
            "available_stock": stock,
            "daily_demand": demand,
            "days_of_supply": (
                round(days_of_supply, 2)
                if days_of_supply is not None
                else None
            ),
            "lead_time_days": lead_time,
            "lead_time_coverage_gap_days": (
                round(coverage_gap, 2)
                if coverage_gap is not None
                else None
            ),
            "reorder_point": reorder_point,
            "below_reorder_point": stock < reorder_point,
            "criticality_score": row["criticality_score"],
            "supplier_id": row["supplier_id"],
            "warehouse_id": row["warehouse_id"],
        })

    prepared.sort(
        key=lambda item: (
            item["lead_time_coverage_gap_days"] or 0
        ),
        reverse=True,
    )

    return prepared


# ----------------------------------
# 3. Define the agents
# ----------------------------------

inventory_analyst = Agent(
    role="Inventory Risk Analyst",
    goal=(
        "Identify inventory risks using supplied, "
        "validated inventory records."
    ),
    backstory=(
        "You specialize in inventory planning, production "
        "continuity, and supply chain risk. You distinguish "
        "measured facts from assumptions and never invent "
        "missing data."
    ),
    verbose=True,
)

planner = Agent(
    role="Inventory Planner",
    goal=(
        "Recommend practical, prioritized actions to "
        "reduce inventory and production risk."
    ),
    backstory=(
        "You are an experienced supply chain planner. "
        "You prioritize production continuity and base "
        "recommendations on verified evidence."
    ),
    verbose=True,
)


# ----------------------------------
# 4. Define the tasks
# ----------------------------------

risk_task = Task(
    description=(
        "Analyze the following actual Bharat Mobility "
        "inventory records:\n\n"
        "{inventory_data}\n\n"
        "Identify the highest-risk items. For each item, "
        "report its item ID, days of supply, supplier lead "
        "time, coverage gap, reorder-point status, and "
        "criticality score.\n\n"
        "Prioritize items with the largest positive coverage "
        "gaps and consider reorder-point breaches and "
        "criticality. Explain potential risks without "
        "claiming that a production stoppage is certain.\n\n"
        "If daily demand is zero, days of supply cannot be "
        "calculated using this method. Clearly flag missing "
        "or uncertain information. Do not invent values."
    ),
    expected_output=(
        "A prioritized inventory risk assessment based "
        "only on the supplied records, including item IDs, "
        "calculated metrics, evidence, and uncertainties."
    ),
    agent=inventory_analyst,
)

planning_task = Task(
    description=(
        "Using the inventory risk assessment, recommend "
        "three prioritized actions for the planner.\n\n"
        "For each action, explain which risk it addresses, "
        "why it is urgent, and what should be verified "
        "before implementation. Consider supplier "
        "expediting, substitute availability, and stock "
        "reallocation only when supporting evidence exists.\n\n"
        "Do not claim that a supplier has been contacted, "
        "stock has been transferred, or an action has been "
        "executed. Do not invent financial savings."
    ),
    expected_output=(
        "Three prioritized recommendations with rationale, "
        "supporting item IDs, and verification steps."
    ),
    agent=planner,
    context=[risk_task],
)


# ----------------------------------
# 5. Create the crew
# ----------------------------------

inventory_crew = Crew(
    agents=[inventory_analyst, planner],
    tasks=[risk_task, planning_task],
    process=Process.sequential,
    verbose=True,
)


# ----------------------------------
# 6. Execute the crew
# ----------------------------------

if __name__ == "__main__":
    inventory_data = prepare_inventory()

    print(f"\nLoaded {len(inventory_data)} inventory records.")

    inputs = {
        "inventory_data": str(inventory_data)
    }

    result = inventory_crew.kickoff(inputs=inputs)

    print("\n===== FINAL INVENTORY RECOMMENDATIONS =====\n")
    print(result)