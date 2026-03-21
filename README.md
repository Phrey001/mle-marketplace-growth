# MLE Marketplace Growth Platform

A self-directed ML engineering project that uses two growth levers on the Online Retail II e-commerce dataset:

- Purchase Propensity: decide which users to target when marketing budget is limited
- Recommender: decide which products to surface for personalized discovery

## Introduction & Architecture

The slide images below are exported from [docs/architecture.pptx](docs/architecture.pptx) and provide the fastest overview of the repo through three ideas:

- Purchase Propensity: which users should be targeted first when marketing budget is limited?
- Recommender: which products should be shown next to improve discovery?
- Shared Feature Layer: point-in-time features, explicit offline evaluation, and separate serving-style artifact generation for both engines

After that, use [docs/README.md](docs/README.md) to navigate the detailed specs and quickstarts.

Detailed purchase propensity and recommender notes are included in the appendix slide groups below.

<details>
<summary>Architecture & Problem Framing</summary>

![Architecture Main Slide 1](docs/architecture_exports/1%20Main/Slide1.PNG)
![Architecture Main Slide 2](docs/architecture_exports/1%20Main/Slide2.PNG)
![Architecture Main Slide 3](docs/architecture_exports/1%20Main/Slide3.PNG)
![Architecture Main Slide 4](docs/architecture_exports/1%20Main/Slide4.PNG)
![Architecture Main Slide 5](docs/architecture_exports/1%20Main/Slide5.PNG)

</details>

<details>
<summary>Appendix - Purchase Propensity</summary>

![Purchase Propensity Slide 6](docs/architecture_exports/2%20Appendix%20-%20Purchase%20Propensity/Slide6.PNG)
![Purchase Propensity Slide 7](docs/architecture_exports/2%20Appendix%20-%20Purchase%20Propensity/Slide7.PNG)
![Purchase Propensity Slide 8](docs/architecture_exports/2%20Appendix%20-%20Purchase%20Propensity/Slide8.PNG)
![Purchase Propensity Slide 9](docs/architecture_exports/2%20Appendix%20-%20Purchase%20Propensity/Slide9.PNG)
![Purchase Propensity Slide 10](docs/architecture_exports/2%20Appendix%20-%20Purchase%20Propensity/Slide10.PNG)
![Purchase Propensity Slide 11](docs/architecture_exports/2%20Appendix%20-%20Purchase%20Propensity/Slide11.PNG)

</details>

<details>
<summary>Appendix - Recommender</summary>

![Recommender Slide 12](docs/architecture_exports/3%20Appendix%20-%20Recommender/Slide12.PNG)
![Recommender Slide 13](docs/architecture_exports/3%20Appendix%20-%20Recommender/Slide13.PNG)
![Recommender Slide 14](docs/architecture_exports/3%20Appendix%20-%20Recommender/Slide14.PNG)

</details>


## What This Repo Demonstrates

- Built ML pipelines for two marketplace growth problems on the Online Retail II e-commerce dataset: budget-constrained user targeting and personalized item retrieval
- Implemented growth decisioning models: purchase propensity modeling (`logistic_regression`, `xgboost` + calibration) with a revenue prediction model (`xgboost` regression) to allocate fixed marketing budgets across users
- Implemented recommender models: Matrix Factorization (`TruncatedSVD`), a PyTorch two-tower neural network, and a popularity baseline (user-agnostic heuristic) to promote product discovery at the user level
- Designed offline evaluation for both policy decisioning and recommender ranking
- Structured a clear separation between offline training/evaluation flows and serving-style artifact generation

## Getting Started

Environment setup (one-time):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Use engine-specific runbooks:
  - [docs/purchase_propensity/quickstart.md](docs/purchase_propensity/quickstart.md)
  - [docs/recommender/quickstart.md](docs/recommender/quickstart.md)
- Engine contracts (split/model/artifact/acceptance): [docs/purchase_propensity/spec.md](docs/purchase_propensity/spec.md), [docs/recommender/spec.md](docs/recommender/spec.md).
- Generated artifacts are intentionally committed in this repo as demo evidence; quickstarts and specs define the canonical artifact layouts.

## Tech Stack

| Scope | Stack |
|---|---|
| Shared | Python, DuckDB |
| Purchase propensity | scikit-learn, XGBoost |
| Recommender | scikit-learn (MF baseline), PyTorch (two-tower), FAISS (ANN retrieval) |

## Key Limitations

| Limitation | Why It Matters |
|---|---|
| No online experimentation / A/B testing | Repo is local/offline with no live traffic, so online impact cannot be measured directly. |
| No offline causal incrementality estimation for promotions | Dataset lacks randomized treatment assignment and reliable promo-exposure logs. |
| No recommender Stage 2 re-ranking | Current recommender scope is intentionally Stage 1 retrieval only. |

Policy and evaluation definitions live in engine specs to avoid doc duplication:
- [docs/purchase_propensity/spec.md](docs/purchase_propensity/spec.md)
- [docs/recommender/spec.md](docs/recommender/spec.md)

## Dataset

[Base dataset: Online Retail II (UCI transactional dataset)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data)

This Online Retail II dataset contains transactions from a UK-based non-store online retail business between 2009-12-01 and 2011-12-09. The company mainly sells unique all-occasion giftware, and many customers are wholesalers.
