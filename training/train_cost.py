"""Step 4: repair-cost range. PUBLIC DATA HAS NO REAL REPAIR COSTS, so this trains on
SYNTHETIC data purely to make the pipeline run. Replace with data/repair_costs.csv
(severity, n_parts, glass, lamp, vehicle_age, cost) of verified invoices before trusting any number."""
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import GradientBoostingRegressor
rng = np.random.default_rng(0); n = 3000
df = pd.DataFrame({"severity": rng.integers(0,3,n), "n_parts": rng.integers(1,6,n),
                   "glass": rng.integers(0,2,n), "lamp": rng.integers(0,2,n), "vehicle_age": rng.integers(0,15,n)})
df["cost"] = (6000 + 14000*df.severity + 4500*df.n_parts + 9000*df.glass + 6000*df.lamp) * rng.normal(1, .15, n)
# real data: df = pd.read_csv("data/repair_costs.csv")
X, y = df.drop(columns="cost"), df.cost
lo = GradientBoostingRegressor(loss="quantile", alpha=.1).fit(X, y)
hi = GradientBoostingRegressor(loss="quantile", alpha=.9).fit(X, y)
joblib.dump((lo, hi, list(X.columns)), "models/cost.joblib")
