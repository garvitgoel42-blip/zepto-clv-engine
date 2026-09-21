# zepto-clv-engine
A clv engine project.
# Quick Commerce Customer Segmentation & CLV Prediction Engine

## What this is

A synthetic-data project modeling customer behavior on an Indian quick-commerce
app (Zepto-style, 10-minute grocery delivery), built to demonstrate an
end-to-end analytics pipeline: customer segmentation via clustering, and
6-month customer lifetime value (CLV) prediction from a new customer's first
3 orders. This is Project 1 of a 3-part portfolio series exploring the quick
commerce business model from different analytical angles.

## Why synthetic data

No access to Zepto's actual transaction data exists outside the company, so
this project builds a data *generator* instead of using a found dataset.
The generator's assumptions are grounded, wherever possible, in public
reporting on Zepto, Blinkit, and Swiggy Instamart (pricing policy, festival
demand spikes, subscription program mechanics, delivery economics). Where no
public data existed, that's stated explicitly rather than presented as fact
— see "Assumptions & Limitations" below.

## Status

**Data generation pipeline: in progress.** Customer archetypes, pricing,
churn, subscription behavior, and category/seasonal effects are built and
sanity-checked. Segmentation (KMeans) and CLV modeling (XGBoost) have not
started yet.

*(This section gets updated every session — treat it as the most current
source of truth on where the project actually stands.)*

## Pipeline

1. **Synthetic transaction generation** ← currently here
2. Customer-level feature engineering
3. KMeans customer segmentation
4. XGBoost CLV prediction model
5. Scoring function: first 3 orders → 6-month CLV prediction
6. Packaging: cleaned notebooks, charts, write-up

## Repo structure

```
zepto-clv-engine/
├── README.md            this file
├── DESIGN_NOTES.md       research findings, decisions, and what got cut
├── notebooks/            interactive Jupyter notebooks
├── scripts/              runnable .py versions of each pipeline step
├── data/                 generated CSVs (not committed if large — see .gitignore)
└── outputs/              charts, saved model files
```

## Key design decisions

- **Four customer archetypes** (Power User, Habitual-Regular, Promo-Hunter,
  Trialist), deliberately overlapping rather than cleanly separated, so
  segmentation has to do real work rather than trivially recovering labels.
- **Archetype weights are Pareto-skewed**, not evenly split: Power Users are
  ~8% of customers but were designed to drive a disproportionate share of
  revenue, reflecting how quick-commerce revenue concentration is reported
  to work in practice.
- **Churn is hazard-based, not a flat monthly rate** — steep in a customer's
  first month, decaying to a low steady rate after, modeling the
  "acquisition coupon wears off" pattern common in discount-driven apps.
- **Wallet-share**: each customer has a latent fraction of their total
  quick-commerce spend that actually comes to this platform. Multi-homing
  (shopping across Zepto/Blinkit/Instamart) is real and well-documented in
  this market, so a quiet customer isn't necessarily a churned one.
- **Zepto Pass changes behavior, not just price**: subscribing increases
  order frequency *and* basket size. An earlier draft assumed baskets would
  shrink (more frequent, smaller top-ups) — that was checked against
  reporting and found backwards; Zepto's own stated strategy is to push
  subscribers toward larger, more premium baskets.
- **Free-delivery threshold anchoring**: carts that would land just under
  the ₹199 threshold get "padded" up to clear it, for price-conscious
  archetypes only, mirroring a real, commonly observed shopping behavior.
- **Pricing has three layers** (MRP → Zepto's everyday price → final bill
  after coupon), so "savings vs. MRP" — the number a customer actually
  feels good about — is explicit rather than an abstract discount %.

## Assumptions & Limitations — please read this section

This is the most important part of the README, and it stays here
permanently, not just during development.

**What this project demonstrates:** the ability to design a realistic
data-generating process, build a segmentation and prediction pipeline on
top of it, and correctly recover known structure from noisy, overlapping
data. That is a genuine, transferable analytics skill.

**What this project does NOT demonstrate:** real insight into actual Zepto
customers. Every customer "archetype" and behavior pattern here was
hand-designed by me before any modeling happened — so when the pipeline
later "discovers" that, say, discount-driven customers churn faster, that
isn't a finding about the real world. It's confirmation that the pipeline
correctly found the pattern I built in. This distinction matters and I'm
stating it plainly rather than letting the results imply otherwise.

**Numbers in this project fall into two categories:**
- *Researched and cited* (in DESIGN_NOTES.md) — e.g. the ₹199 free-delivery
  threshold, festival order-volume increases, monsoon category-share data.
- *Stated modeling assumptions* — e.g. the exact shape of the churn decay
  curve, the Pass basket-size lift percentage, wallet-share distributions.
  These are chosen to be directionally realistic based on available
  reporting, not presented as verified Zepto figures. Several early
  assumptions were revised after fact-checking turned up contradicting
  evidence (documented in DESIGN_NOTES.md) — that process is part of the
  project, not something to hide.

**Deliberately out of scope for this version:** contribution margin /
cost-to-serve modeling, app engagement or clickstream data, service-failure
history, household-size effects, acquisition-channel funnels, dark-store
capacity constraints. These are real, meaningful dimensions of quick
commerce customer behavior — they're just a different, larger project than
what a portfolio piece on this timeline should try to be.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib jupyter
```

## Running the generator

```bash
cd scripts
python generate_transactions.py
```

Outputs `transactions.csv` and `truth_archetypes.csv` to `../data/`, and
prints sanity-check diagnostics (archetype mix, retention curve, category
distribution, Pass subscriber effects, GMV concentration).
