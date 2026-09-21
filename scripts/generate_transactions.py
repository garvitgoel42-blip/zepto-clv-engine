"""
Synthetic quick-commerce transaction generator -- v3
Project 1: Customer Segmentation & CLV Prediction Engine (Zepto case study)

v3 changes from v2, following the locked-in v1 feature list:
    - State-machine architecture: each customer carries explicit state
      (churned? subscriber? which month did they convert?) and small
      policy functions decide when that state changes. The main loop
      just orchestrates -- it doesn't contain the actual logic.
    - Hazard-based churn: front-loaded ("hockey stick") monthly churn
      probability instead of one flat rate -- steep in months 1-2,
      decaying to a low steady rate after.
    - Wallet-share: each customer has a latent fraction of their total
      quick-commerce spend that actually comes to us. A "quiet" customer
      may just be shopping on a competitor that month, not gone.
    - Zepto Pass, corrected: subscribing raises order frequency AND
      basket size (not a smaller-basket effect -- that assumption was
      checked against reporting and found backwards; Zepto's own
      strategy is explicitly to push subscribers toward bigger,
      premium baskets, not more numerous small ones).
    - Archetype weights corrected: Power User down to ~8% of the base
      (was 17%) to reflect realistic revenue concentration instead of
      an even split.
    - Shopping mission / category: each order gets a category, drawn
      from an archetype-specific mix, with overrides for festival dates
      and a June monsoon-category effect.
    - Day-level weighting: salary-cycle (days 1-5), weekends, and two
      festival dates (the ones that actually fall in our Jan-Jun 2026
      window: Valentine's Day, Holi) get elevated order-day weight
      instead of every day being drawn uniformly.

Honesty notes (for the README / DESIGN_NOTES, not just this docstring):
    - Some numbers here are researched and cited (free-delivery
      threshold, festival order-volume upticks, monsoon category share).
    - Others are *stated modeling assumptions*, not researched figures
      (exact hazard decay rates, Pass basket-size lift, wallet-share
      distributions). They're chosen to be directionally realistic,
      not presented as real Zepto numbers.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# ==============================================================
# 1) Static behavioral parameters, by archetype
# ==============================================================

ARCHETYPE_WEIGHTS = {
    "power_user": 0.08,          # corrected down from 17% -- revenue concentration, not an even split
    "habitual_regular": 0.32,
    "promo_hunter": 0.33,
    "trialist": 0.27,
}

ARCHETYPE_PARAMS = {
    "power_user": {
        "lambda_month": 10.0,
        "basket_median": 380,
        "basket_sigma": 0.40,
        "discount_alpha": 2,
        "discount_beta": 18,
        "high_ticket_prob": 0.025,
        "pads_for_free_delivery": False,
        "payment_mix": {"upi": 0.65, "card": 0.30, "cash": 0.05},
        "category_mix": {"grocery_staples": 0.45, "snacks_impulse": 0.20,
                          "personal_care": 0.15, "party_occasion": 0.10,
                          "premium_beauty": 0.10},
        "base_hazard": 0.02,       # steady-state monthly churn once past the early window
        "early_hazard_boost": 0.03,
        "hazard_decay_rate": 0.8,
        "pass_conversion_prob": 0.06,   # per-month chance of subscribing, while not yet a subscriber
        "wallet_share_beta": (8, 2),    # mean ~0.80 -- loyal, low multi-homing
    },
    "habitual_regular": {
        "lambda_month": 4.0,
        "basket_median": 330,
        "basket_sigma": 0.42,
        "discount_alpha": 3,
        "discount_beta": 17,
        "high_ticket_prob": 0.015,
        "pads_for_free_delivery": True,
        "payment_mix": {"upi": 0.60, "card": 0.25, "cash": 0.15},
        "category_mix": {"grocery_staples": 0.60, "snacks_impulse": 0.20,
                          "personal_care": 0.15, "party_occasion": 0.03,
                          "premium_beauty": 0.02},
        "base_hazard": 0.05,
        "early_hazard_boost": 0.20,
        "hazard_decay_rate": 0.6,
        "pass_conversion_prob": 0.04,
        "wallet_share_beta": (5, 3),    # mean ~0.625
    },
    "promo_hunter": {
        "lambda_month": 3.0,
        "basket_median": 300,
        "basket_sigma": 0.45,
        "discount_alpha": 6,
        "discount_beta": 14,
        "high_ticket_prob": 0.007,
        "pads_for_free_delivery": True,
        "payment_mix": {"upi": 0.55, "card": 0.20, "cash": 0.25},
        "category_mix": {"grocery_staples": 0.35, "snacks_impulse": 0.35,
                          "personal_care": 0.15, "party_occasion": 0.10,
                          "premium_beauty": 0.05},
        "base_hazard": 0.08,
        "early_hazard_boost": 0.37,
        "hazard_decay_rate": 0.55,
        "pass_conversion_prob": 0.015,  # chases per-order deals -- less inclined to commit
        "wallet_share_beta": (3, 5),    # mean ~0.375 -- heavy multi-homer
    },
    "trialist": {
        "lambda_month": None,           # special-cased: exactly one order, no monthly loop
        "basket_median": 350,
        "basket_sigma": 0.40,
        "discount_alpha": 8,
        "discount_beta": 12,
        "high_ticket_prob": 0.005,
        "pads_for_free_delivery": True,
        "payment_mix": {"upi": 0.55, "card": 0.15, "cash": 0.30},
        "category_mix": {"grocery_staples": 0.30, "snacks_impulse": 0.30,
                          "personal_care": 0.15, "party_occasion": 0.15,
                          "premium_beauty": 0.10},
    },
}

HIGH_TICKET_MEDIAN = 2200
HIGH_TICKET_SIGMA = 0.28

FREE_DELIVERY_THRESHOLD = 199   # current Zepto MOV as of Aug 2026 (researched)
PAD_ZONE_LOW = 150
PAD_PROBABILITY = 0.55

KIRANA_DENSITY_WEIGHTS = {"new_society": 0.40, "established": 0.60}
KIRANA_FREQUENCY_MULTIPLIER = {"new_society": 1.10, "established": 0.85}

PASS_FREQUENCY_MULTIPLIER = 1.30    # stated assumption, not a researched figure
PASS_BASKET_MULTIPLIER = 1.15       # stated assumption -- direction (up) is researched, size is not

# Festival dates that fall inside our Jan-2026-to-Jun-2026 simulation window.
# Multipliers here are modest, whole-basket assumptions -- NOT the same as the
# 4x/14x category-specific spikes found in research, which applied to single
# SKUs (chocolate, decorative lights), not a customer's whole basket.
FESTIVAL_DATES = {
    "2026-02-14": {"label": "valentines_day", "category": "festive_special",
                    "day_weight": 2.2, "basket_multiplier": 1.35},
    "2026-03-04": {"label": "holi", "category": "festive_special",       # date assumed, not confirmed
                    "day_weight": 2.5, "basket_multiplier": 1.30},
}
MONSOON_MONTH_INDEX = 5             # June, if start_date is Jan 2026 -- 0-indexed month
MONSOON_CATEGORY_SHARE = 0.20       # ~1 in 5 June orders includes a monsoon item (researched, Flipkart Minutes)


# ==============================================================
# 2) Customer state -- what CAN change over the simulation
# ==============================================================

@dataclass
class CustomerState:
    customer_id: int
    archetype: str
    area_type: str
    assortment_reliability: float
    wallet_share: float
    base_lambda: float          # this customer's personal order-rate heterogeneity
    basket_median: float        # this customer's personal basket-size heterogeneity

    active_months: int = 0
    is_churned: bool = False
    is_pass_subscriber: bool = False
    subscription_month: int = None


# ==============================================================
# 3) Policy functions -- decide WHEN state changes
# ==============================================================

def churn_hazard(customer):
    """Monthly probability of churning this month, given current state."""
    params = ARCHETYPE_PARAMS[customer.archetype]
    age = customer.active_months
    hazard = params["base_hazard"] + params["early_hazard_boost"] * np.exp(
        -params["hazard_decay_rate"] * max(age - 1, 0)
    )
    # More reliable stock -> a bit stickier; less reliable -> a bit more churn-prone
    reliability_adj = 1.3 - 0.3 * customer.assortment_reliability
    return min(hazard * reliability_adj, 0.95)


def update_churn_state(customer, rng):
    if rng.random() < churn_hazard(customer):
        customer.is_churned = True


def pass_conversion_probability(customer):
    return ARCHETYPE_PARAMS[customer.archetype]["pass_conversion_prob"]


def update_subscription_state(customer, month, rng):
    if customer.is_pass_subscriber:
        return
    if rng.random() < pass_conversion_probability(customer):
        customer.is_pass_subscriber = True
        customer.subscription_month = month


def get_behavior_parameters(customer):
    """Derive this month's effective behavior from archetype + current state."""
    params = ARCHETYPE_PARAMS[customer.archetype].copy()
    params["lambda_month"] = customer.base_lambda * KIRANA_FREQUENCY_MULTIPLIER[customer.area_type]
    params["basket_median"] = customer.basket_median

    if customer.is_pass_subscriber:
        params = apply_pass_effects(params)

    return params


def apply_pass_effects(params):
    params = params.copy()
    params["lambda_month"] *= PASS_FREQUENCY_MULTIPLIER
    params["basket_median"] *= PASS_BASKET_MULTIPLIER
    params["pads_for_free_delivery"] = False    # threshold doesn't apply to subscribers
    params["discount_alpha"] += 2                # a modestly richer discount tier
    return params


# ==============================================================
# 4) Activity generators -- turn behavior parameters into orders
# ==============================================================

def _draw_basket_values(rng, n, median, sigma, high_ticket_prob):
    core = rng.lognormal(mean=np.log(median), sigma=sigma, size=n)
    is_high_ticket = rng.random(n) < high_ticket_prob
    if is_high_ticket.any():
        tail = rng.lognormal(mean=np.log(HIGH_TICKET_MEDIAN), sigma=HIGH_TICKET_SIGMA,
                              size=is_high_ticket.sum())
        core[is_high_ticket] = tail
    return core, is_high_ticket


def _day_weights(month_start, month_length_days):
    """
    Build a per-day weight array for this month: baseline 1.0, boosted for
    salary-cycle days (1st-5th), weekends, and festival dates.
    """
    days = pd.date_range(month_start, periods=month_length_days, freq="D")
    weights = np.ones(month_length_days)

    is_salary_window = days.day <= 5
    weights[is_salary_window] *= 1.30

    is_weekend = days.dayofweek >= 5   # Sat=5, Sun=6
    weights[is_weekend] *= 1.25

    festival_day_idx = {}
    for date_str, info in FESTIVAL_DATES.items():
        festival_date = pd.Timestamp(date_str)
        matches = days == festival_date
        if matches.any():
            idx = int(np.where(matches)[0][0])
            weights[idx] *= info["day_weight"]
            festival_day_idx[idx] = info

    return weights / weights.sum(), festival_day_idx


def _draw_categories(rng, n, order_days, festival_day_idx, month_idx, category_mix):
    names = list(category_mix.keys())
    probs = list(category_mix.values())
    categories = rng.choice(names, size=n, p=probs)

    for i, day_idx in enumerate(order_days):
        if day_idx in festival_day_idx and rng.random() < 0.6:
            categories[i] = festival_day_idx[day_idx]["category"]
        elif month_idx == MONSOON_MONTH_INDEX and rng.random() < MONSOON_CATEGORY_SHARE:
            categories[i] = "monsoon_essentials"

    return categories


def _apply_free_delivery_padding(rng, gross_amount, pads):
    if not pads:
        return gross_amount
    n = len(gross_amount)
    in_pad_zone = (gross_amount >= PAD_ZONE_LOW) & (gross_amount < FREE_DELIVERY_THRESHOLD)
    will_pad = in_pad_zone & (rng.random(n) < PAD_PROBABILITY)
    if will_pad.any():
        gross_amount = gross_amount.copy()
        gross_amount[will_pad] = FREE_DELIVERY_THRESHOLD + rng.uniform(0, 40, size=will_pad.sum())
    return gross_amount


def _apply_pricing_layers(rng, n, gross_amount, discount_alpha, discount_beta):
    platform_discount_rate = rng.uniform(0.05, 0.15, size=n)
    mrp_amount = gross_amount / (1 - platform_discount_rate)
    discount_rate = rng.beta(discount_alpha, discount_beta, size=n)
    discount_amount = gross_amount * discount_rate
    net_amount = gross_amount - discount_amount
    savings_vs_mrp = mrp_amount - net_amount
    return mrp_amount, discount_rate, discount_amount, net_amount, savings_vs_mrp


def simulate_month(customer, month_idx, month_start, month_length_days, rng):
    """Generate this customer's orders for one active month."""
    params = get_behavior_parameters(customer)

    # Wallet-share: this customer's TRUE demand is diluted by however much of
    # their spend goes to competitors. What we observe is only our slice.
    observed_lambda = params["lambda_month"] * customer.wallet_share
    n_orders = rng.poisson(max(observed_lambda, 0))
    if n_orders == 0:
        return []

    day_weights, festival_day_idx = _day_weights(month_start, month_length_days)
    order_day_idx = rng.choice(month_length_days, size=n_orders, p=day_weights)
    order_dates = month_start + pd.to_timedelta(order_day_idx, unit="D")

    gross_amount, is_ht = _draw_basket_values(
        rng, n_orders, params["basket_median"], params["basket_sigma"], params["high_ticket_prob"]
    )
    gross_amount = _apply_free_delivery_padding(rng, gross_amount, params["pads_for_free_delivery"])

    for day_idx in order_day_idx:
        if day_idx in festival_day_idx:
            gross_amount = gross_amount.copy()
            gross_amount[order_day_idx == day_idx] *= festival_day_idx[day_idx]["basket_multiplier"]

    mrp, disc_rate, disc_amt, net, savings = _apply_pricing_layers(
        rng, n_orders, gross_amount, params["discount_alpha"], params["discount_beta"]
    )
    categories = _draw_categories(rng, n_orders, order_day_idx, festival_day_idx, month_idx, params["category_mix"])
    payment = rng.choice(list(params["payment_mix"].keys()), size=n_orders, p=list(params["payment_mix"].values()))

    rows = []
    for date, m, g, dr, da, n_amt, sv, cat, pay, ht in zip(
        order_dates, mrp, gross_amount, disc_rate, disc_amt, net, savings, categories, payment, is_ht
    ):
        rows.append({
            "customer_id": customer.customer_id,
            "order_date": date,
            "mrp_amount": m,
            "gross_amount": g,
            "discount_rate": dr,
            "discount_amount": da,
            "net_amount": n_amt,
            "savings_vs_mrp": sv,
            "category": cat,
            "payment_method": pay,
            "high_ticket_flag": bool(ht),
            "is_pass_subscriber": customer.is_pass_subscriber,
        })
    return rows


def simulate_trialist(customer, start, rng):
    """Trialists get exactly one order, in month 0, then are done."""
    params = ARCHETYPE_PARAMS["trialist"]
    order_day = int(rng.integers(0, 30))
    order_date = start + pd.to_timedelta(order_day, unit="D")

    gross, is_ht = _draw_basket_values(rng, 1, params["basket_median"], params["basket_sigma"],
                                        params["high_ticket_prob"])
    gross = _apply_free_delivery_padding(rng, gross, params["pads_for_free_delivery"])
    mrp, disc_rate, disc_amt, net, savings = _apply_pricing_layers(
        rng, 1, gross, params["discount_alpha"], params["discount_beta"]
    )
    category = rng.choice(list(params["category_mix"].keys()), p=list(params["category_mix"].values()))
    payment = rng.choice(list(params["payment_mix"].keys()), p=list(params["payment_mix"].values()))

    return [{
        "customer_id": customer.customer_id,
        "order_date": order_date,
        "mrp_amount": mrp[0],
        "gross_amount": gross[0],
        "discount_rate": disc_rate[0],
        "discount_amount": disc_amt[0],
        "net_amount": net[0],
        "savings_vs_mrp": savings[0],
        "category": category,
        "payment_method": payment,
        "high_ticket_flag": bool(is_ht[0]),
        "is_pass_subscriber": False,
    }]


# ==============================================================
# 5) Orchestration -- the loop stays boring on purpose
# ==============================================================

def generate_transactions(n_customers=4000, months=6, start_date="2026-01-01",
                           customer_rate_cv=0.18, customer_basket_log_sd=0.12,
                           random_seed=42):
    rng = np.random.default_rng(random_seed)
    start = pd.Timestamp(start_date)

    archetype_names = list(ARCHETYPE_WEIGHTS.keys())
    archetype_probs = list(ARCHETYPE_WEIGHTS.values())
    area_names = list(KIRANA_DENSITY_WEIGHTS.keys())
    area_probs = list(KIRANA_DENSITY_WEIGHTS.values())

    transactions = []
    truth = []

    for customer_id in range(1, n_customers + 1):
        archetype = rng.choice(archetype_names, p=archetype_probs)
        area_type = rng.choice(area_names, p=area_probs)
        params = ARCHETYPE_PARAMS[archetype]

        assortment_reliability = rng.beta(8, 2)
        wallet_alpha, wallet_beta = params.get("wallet_share_beta", (5, 5))
        wallet_share = rng.beta(wallet_alpha, wallet_beta)

        truth.append({
            "customer_id": customer_id,
            "archetype": archetype,
            "area_type": area_type,
            "assortment_reliability": assortment_reliability,
            "wallet_share": wallet_share,
        })

        if archetype == "trialist":
            transactions.extend(simulate_trialist(
                CustomerState(customer_id, archetype, area_type, assortment_reliability,
                              wallet_share, base_lambda=0, basket_median=params["basket_median"]),
                start, rng,
            ))
            continue

        shape = 1 / (customer_rate_cv ** 2)
        scale = params["lambda_month"] * (customer_rate_cv ** 2)
        base_lambda = rng.gamma(shape=shape, scale=scale)
        basket_median = params["basket_median"] * rng.lognormal(mean=0, sigma=customer_basket_log_sd)

        customer = CustomerState(
            customer_id=customer_id, archetype=archetype, area_type=area_type,
            assortment_reliability=assortment_reliability, wallet_share=wallet_share,
            base_lambda=base_lambda, basket_median=basket_median,
        )

        for month_idx in range(months):
            if customer.is_churned:
                break

            customer.active_months += 1
            update_subscription_state(customer, month_idx, rng)

            month_start = start + pd.DateOffset(months=month_idx)
            next_month = start + pd.DateOffset(months=month_idx + 1)
            month_length_days = (next_month - month_start).days

            rows = simulate_month(customer, month_idx, month_start, month_length_days, rng)
            transactions.extend(rows)

            # Churn is evaluated AFTER this month's activity, not before --
            # a customer always gets a chance to order in a month before we
            # decide whether they return for the next one. Evaluating it
            # first (as an earlier draft of this did) meant customers could
            # churn before ever placing an order in their "active" month,
            # which silently made month-0 retention far too low.
            update_churn_state(customer, rng)

    transactions = pd.DataFrame(transactions).sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    truth = pd.DataFrame(truth)
    return transactions, truth


def monthly_retention_curve(transactions, truth, start_date="2026-01-01", months=6):
    start = pd.Timestamp(start_date)
    tx = transactions.copy()
    tx["month_idx"] = ((tx["order_date"] - start).dt.days // 30.44).clip(upper=months - 1).astype(int)
    n_customers = truth["customer_id"].nunique()
    return pd.Series(
        [tx.loc[tx["month_idx"] >= m, "customer_id"].nunique() / n_customers for m in range(months)],
        index=[f"M{m}" for m in range(months)],
    )


if __name__ == "__main__":
    transactions, truth = generate_transactions(n_customers=4000, months=6, random_seed=42)

    print(transactions.head())
    print("\nShape:", transactions.shape)

    print("\nArchetype mix:")
    print(truth["archetype"].value_counts(normalize=True).round(3))

    print("\nRetention curve (front-loaded churn -- should drop fast then flatten):")
    print(monthly_retention_curve(transactions, truth).round(3))

    print("\nCategory mix (overall):")
    print(transactions["category"].value_counts(normalize=True).round(3))

    pass_orders = transactions[transactions["is_pass_subscriber"]]
    non_pass_orders = transactions[~transactions["is_pass_subscriber"]]
    print(f"\nPass subscriber orders: {len(pass_orders)} ({len(pass_orders)/len(transactions):.1%} of all orders)")
    print(f"Avg basket -- Pass subscribers: {pass_orders['gross_amount'].mean():.0f} "
          f"vs non-subscribers: {non_pass_orders['gross_amount'].mean():.0f}")

    print("\nGMV share by archetype (checking Power User concentration):")
    gmv_by_archetype = (
        transactions.merge(truth, on="customer_id")
        .groupby("archetype")["net_amount"].sum()
        .sort_values(ascending=False)
    )
    print((gmv_by_archetype / gmv_by_archetype.sum()).round(3))

    print("\nWallet-share summary by archetype:")
    print(truth.groupby("archetype")["wallet_share"].mean().round(3))

    transactions.to_csv("/Users/garvitgoel/zepto-clv-engine/data/transactions.csv", index=False)
    truth.to_csv("/Users/garvitgoel/zepto-clv-engine/data/truth_archetypes.csv", index=False)
    print("\nSaved to ../data/")
