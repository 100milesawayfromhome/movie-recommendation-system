# 🎬 Personalized Recommendation Systems for Content Discovery

> A complete, reproducible recommendation engine built on the **Netflix Prize dataset** — learning user
> taste from **100,480,507 ratings** to predict unseen scores and generate personalized Top‑10 lists.

Our best model reaches an **RMSE of 0.9093**, beating Netflix's own **Cinematch (0.9514)**, while a
hybrid ranker delivers the strongest **MAP@10**. Seven models are implemented and compared end‑to‑end on
the full dataset.

<p align="center">
  
</p>

---

## 📊 Headline Results

Evaluated on the dataset's official **`probe.txt`** held‑out set (1.4M ratings), full 100M‑rating run.
Relevance for ranking = a true rating **≥ 3.5** (per the problem statement).

| Model | RMSE ↓ | MAE ↓ | MAP@10 ↑ | Recall@10 | NDCG@10 | HitRate@10 |
|-------|:------:|:-----:|:--------:|:---------:|:-------:|:----------:|
| MostPopular *(baseline)* | — | — | 0.0083 | 0.0224 | 0.0133 | 0.0393 |
| Bias Baseline | 0.9841 | 0.7744 | 0.0039 | 0.0098 | 0.0059 | 0.0170 |
| Funk SVD | 0.9225 | 0.7153 | 0.0055 | 0.0172 | 0.0099 | 0.0317 |
| SVD++ | 0.9106 | 0.7033 | 0.0093 | 0.0257 | 0.0156 | 0.0487 |
| **Ensemble** 🏆 | **0.9093** | 0.7018 | 0.0095 | 0.0242 | 0.0154 | 0.0457 |
| BPR | — | — | 0.0131 | 0.0355 | 0.0214 | 0.0637 |
| **Hybrid** 🏆 | — | — | **0.0208** | **0.0591** | **0.0349** | **0.1063** |

🏆 **Ensemble** wins on rating accuracy (RMSE). **Hybrid** wins on ranking quality (MAP@10).
Ranking‑only models don't predict star ratings, so RMSE is not applicable.

> **Key insight:** the best *rating* model and the best *recommender* are **different models** — accuracy
> and ranking are distinct objectives. We serve the Ensemble for prediction and the Hybrid for discovery.

---

## 🗂️ Repository Structure

```
.
├── README.md                       ← you are here
├── requirements.txt                ← Python dependencies
├── problemStatement.pdf            ← the original challenge brief
│
├── 📓: kaggleV2(2.0).ipynb         ← Main notebook
│
├── 📄 Deliverables
│   ├── Technical_Report.pdf          
│   └── Presentation.pptx             
│
├── 📈 Results & assets
│   ├── model_comparison.csv          final metrics from the full-data run
│   └── report_assets/                generated charts (EDA, RMSE, MAP@10)

```

---

## 🚀 Quick Start

### Option A — Run on Kaggle (recommended, zero setup)

The notebook **auto‑detects Kaggle** and finds the data automatically.

1. Create a new Kaggle Notebook.
2. **Add Data** → search **“Netflix Prize Data”** (`netflix-inc/netflix-prize-data`) → Add.
3. Upload **`netflix_recommender_v2.ipynb`**.
4. In the config cell, set **`FULL_RUN = True`** for all 100M ratings (or `False` for a fast subset).
5. **Run All.** Full run trains end‑to‑end on a single CPU session in ≈ 35–45 min.

### Option B — Run locally

```bash
# 1. Clone and enter the repo
git clone <your-repo-url> && cd <repo>

# 2. Create an isolated environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from Kaggle and unzip into ./archive/
#    https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
#    (archive/ should contain combined_data_1..4.txt, movie_titles.csv, probe.txt, qualifying.txt)

# 5. Launch
jupyter notebook netflix_recommender_v2.ipynb
```

> 💡 The first run parses ~2 GB of text and caches it to `ratings_full.pkl` (≈ 1.5 GB), so subsequent
> runs start instantly. Keep `FULL_RUN = False` for a quick laptop pass on `combined_data_1.txt` only.

---

## ⚙️ Configuration

All knobs live in the **config cell** near the top of the notebook:

| Setting | Default | Purpose |
|---------|:-------:|---------|
| `FULL_RUN` | `False` | `True` = all 100M ratings; `False` = fast subset (file 1 only) |
| `MIN_USER_RATINGS` / `MIN_MOVIE_RATINGS` | 5 / 20 | filter cold users/movies before training |
| `MF_FACTORS` | 100 | latent dimensions for Funk SVD / SVD++ |
| `MF_EPOCHS`, `MF_LR`, `MF_REG` | 25, 0.005, 0.02 | SGD schedule + regularization |
| `BPR_FACTORS`, `BPR_EPOCHS` | 64, 20 | ranking model size / training length |
| `RELEVANCE_THRESHOLD` | 3.5 | a movie is "relevant" if true rating ≥ this |
| `TOP_K` | 10 | length of recommendation lists |

---

## 🧩 Methodology in Brief

### The pipeline
**Ingest** (stream‑parse 4 text files → tidy table) → **Split** (leakage‑free probe set) →
**EDA** → **Train 7 models** → **Evaluate** (RMSE + MAP@10) → **Recommend** (Top‑10 + explanations + cold‑start).

### The models — a ladder of ideas
1. **Bias Baseline** — `μ + bᵤ + bᵢ`; cheap, strong, and the cold‑start fallback.
2. **Funk SVD** — SGD matrix factorization on **observed ratings only** (the key fix over zero‑imputed SVD).
3. **SVD++** — adds implicit feedback (*which* films a user rated); a pillar of the winning Prize solution.
4. **BPR** — Bayesian Personalized Ranking; optimizes item **ordering** directly → best tool for Top‑K.
5. **Ensemble** — **leakage‑free linear stacking** of the rating models; guaranteed ≥ best single model.
6. **Hybrid** — rank‑blend of BPR + SVD++; best overall ranking.

All three SGD models are **numba‑JIT‑compiled**, so a full epoch over 99M ratings runs in under a minute.

### Evaluation 
- **Test set:** the dataset's own `probe.txt`, removed from training via an anti‑join (no leakage).
- **Blend split:** test is further divided into a **25% slice** (used *only* to learn ensemble weights) and a
  disjoint **75% slice** — **every reported number comes from the 75% slice**, so the learned ensemble can't
  leak into its own score.
- **Metrics:** **RMSE/MAE** (accuracy) + **MAP@10 / Precision / Recall / NDCG / HitRate / Coverage** (ranking).

---

## 💡 What the Model Learned (sample output)

The system infers taste **purely from behaviour — no genre labels**:

```
USER 963054 — rated highly: Batman Begins (5), Pirates of the Caribbean (4), Indiana Jones (4)
  → Recommended: Lost: Season 1  (because you liked Batman Begins, sim 0.51)
  → Recommended: Battlestar Galactica  (because you liked Batman Begins, sim 0.50)

“People who liked Miss Congeniality also liked…”
  → Two Weeks Notice (0.83) · The Wedding Planner (0.82) · The Princess Diaries (0.82) · Legally Blonde (0.79)
```

A coherent **action‑adventure** cluster and a clean **rom‑com** cluster emerge with zero metadata.




## 📚 Dataset & Acknowledgements

- **Netflix Prize Dataset** — 100,480,507 ratings · 480,189 users · 17,770 movies (1998–2005).
  [kaggle.com/datasets/netflix-inc/netflix-prize-data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)


---


