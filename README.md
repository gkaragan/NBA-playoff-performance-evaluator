# NBA Playoff Performance Evaluator: A Context-Aware Statistical Framework 
🏀 *Measuring Playoff Risers and Droppers*  

## The Problem  

During the playoffs, debates often spark about which players rise to the occasion and which fall short.  
Using just raw stats can be misleading: different players have different expectations - top players are expected to perform.

This project provides a **context-aware model** that uses weighted regression, baseline performance tiering,  
and robust statistical methods to fairly evaluate playoff risers and droppers. Player performance is judged using a cumulative 'impact score' which is the sum of the impact scores for each relevant statistic. The model is peer-relative: expected playoff performance and impact scores are calculated relative to the playoff field of the given season. This keeps results grounded in the actual competitive environment of that particular year, rather than external or historical benchmarks. The results will show which players truly separated themselves from the rest of the pack.

---

## The Solution  
This evaluator quantifies playoff risers and droppers using:  
- **Baseline-adjusted expectations**: Weighted regression predicts playoff stats from regular season baselines, the residual isolates true over/under-performance. 
- **Tier-based context**: Players are grouped into tiers by regular season impact, so stars are held to tougher standards while role players are judged relative to their role. 
- **Reliability scaling**: – Impact scores are scaled by playoff sample size (games × minutes), so small-sample anomalies don’t distort results.  
- **Robust statistics**: – Residuals are standardized using MAD (median absolute deviation) instead of standard deviation, reducing the effect of statistical noise and outliers. 
---

## 🎯 Key Features  

### 1. Baseline-Adjusted Residuals  
- For each stat, a weighted regression predicts playoff performance from regular season performance.  
- The difference (actual – expected) = **residual**, reveals true over/underperformance.
- A large residual indicates that the player greatly outperformed what was expected given their RS baseline.  

👉 Example: A 20 PPG scorer rising to 23 PPG is *above expectation* if the playoff data suggests players with similar baselines usually decline.  

[View PTS Regression Chart](./pts_regression.png)


---

### 2. Tier-Based Context  
Players are grouped into tiers using a composite score (PTS, AST, REB, TS%) based on their regular season (RS) performance:  

- **Tier 1**: 85th+ percentile — the most elite RS performers 
- **Tier 2**: 70th-85th — high performers  
- **Tier 3**: 50th-70th — strong contributors  
- **Tier 4**: Below 50th — other  

**Why it matters:**  
- Stars get harsher penalties for decline and bigger credit for improvement. 
- Role players aren’t unfairly overvalued for small jumps; but, are still rewarded for strong improvements. This is crucial to ensure that large raw improvements from small baselines do not outweigh improvements from elite baselines. Small baseline improvements should be weighed highly only if meaningful, that is, they have elevated the players performance into a 'strong' portion of the sample.
- Sustaining elite production relative to peers softens decline penalties.  

---

### 3. Smart Stat Weighting  
Each stat contributes differently to playoff impact:  

```python
STAT_WEIGHTS = {
    # Scoring & Efficiency (60% total) - most important in playoffs
    "PTS": 0.4,
    "TS_PCT": 0.20,
    # Playmaking & Ball Security (22.5% total)
    "AST": 0.175,
    "PCT_TOV": 0.05,  # Turnovers more costly in playoffs 
    # Rebounding (12.5% total)
    "REB": 0.125,
    # Defense/Hustle (5% total)
    "STL": 0.025, # Reduced weights because defensive stats are noisy in small playoff samples
    "BLK": 0.025,
}
```

Special handling:  
- **TS%**: Low-volume shooters (FGA < 10) don’t get inflated bonuses for efficiency improvements. This is important because for big men with low volume, the majority of their shots come at the rim off of lobs/pnr - they are supposed to have a high TS%. For specialist shooters, while impressive, the weight must be reduced because they are not the focal point of the offense. 
- **STL/BLK**: Scaled down due to volatility. If the sample included only players playing in multiple postseasons, then the STL/BLK performance would be much more accurate and increasing the weight would be feasible.

---

### 4. Statistical Robustness  
- **Weighted Least Squares (WLS)**: Players with more playoff minutes anchor the regression line. Previous versions of the model had less strict restrictions - the MIN limit was lower. Hence, noisy performances could distort the expectation so more strict restrictions were applied. This is because we are interested in meaningful improvements.
- **Robust Z-scores (MAD)**: Reduces outlier influence. Z's allow comparability accross stats and reflect rarity relative to the playoff population. Peer-relative: each player’s deviation is judged against how all playoff players shifted.
- **Robust Standard Errors (HC3)**: Ensures inference isn’t misled by heteroskedasticity.  
- **Reliability scaling**: Impact scores scaled by minutes × games — stabilizes rankings. Reasonable upper bound - after a certain MIN threshold, performance should be taken for what it is.

**Diagnostics (for model validation)**
When testing individual stat regressions, diagnostics are used to check model validity:

- Goodness of Fit (R², Adjusted R²): Measure how much variance in playoff stats is explained by regular season stats, ensuring the baseline captures meaningful predictive power.
- Coefficient Significance (p-values): Verify that regression coefficients (e.g., slope of RS → PO) are statistically meaningful and not just noise.
- Residual Analysis: Visual checks (residuals vs. fitted plots, histograms, QQ-plots) to evaluate assumptions like linearity and normality.
- Heteroskedasticity Test (Breusch–Pagan): Detect whether residual variance changes with player values, which can bias inference if ignored.
- Robust Standard Errors (HC3): Adjust coefficient uncertainty estimates when heteroskedasticity is present, providing reliable inference even with noisy real-world data.

These diagnostics are not used in final scoring but provide a quality check to ensure the modeling is statistically sound.

---

## 📊 Key Results (2024–25 Season)  

### Top Risers  
- Kawhi Leonard (#1): Big scoring boost (+3.5 PPG) and efficiency gain (+4.1% TS) with increased playmaking (+1.6 APG), showing comprehensive improvement.
- Donovan Mitchell (#2): Huge scoring surge (+5.6 PPG) while maintaining good efficiency.
- Jayson Tatum (#3): Moderate scoring increase (+1.3 PPG) with strong rebounding improvement (+2.8 RPG) while maintaining his efficiency levels.
- Giannis Antetokounmpo (#4): Solid scoring (+2.6 PPG) and rebounding (+3.5 RPG) gains with improved efficiency (+2.6% TS).
- Jalen Brunson (#5): Notable scoring increase (+3.4 PPG) while maintaining elite playmaking.

### Notable Droppers  
- Jaren Jackson Jr., Austin Reaves, Jalen Green, Tyler Herro, Karl-Anthony Towns

[View Top Risers and Droppers](./top_risers_droppers.png)

### Full Rankings
The repository includes the complete [rankings.csv](./rankings.csv) file with impact scores for players meeting the minimum qualifications for the 2024–25 playoffs.

---

## 🔧 Technical Pipeline  
1. **Data Collection**: NBA API (via `nba_api`) for RS and PO stats.  
2. **Feature Engineering**: Stat differences from RS to PO.  
3. **Tier Classification**: Composite RS score → percentiles → Tier 1–4.  
4. **Baseline Modeling**: WLS regression per stat (PO ~ RS).  
5. **Residual Analysis**: Robust z-scores (MAD).  
6. **Impact Calculation**: Tier-based scoring, stat weights, reliability scaling.  

---

## 📈 Model Validation   

👉 Not all stats fit equally well:  
- REB, PTS = strong fit.  
- BLK, STL = noisier, skewed residuals.  
- TS% = handled with extra safeguards to avoid over valuing low volume shooters maintaining efficiency. 

---

## 🚀 Future Enhancements  
- Multi-season analysis (consistency over the years).  
- Opponent adjustment (defensive difficulty).  
- Clutch weighting (close games / elimination games / crunch time).  
- Position-specific baselines (guards vs. bigs) - however, basketball is becoming positionless.  
- Team-context adjustment (role changes due to injuries).
- Using historical data to fit the regresssions could prove useful in being a more stable prediction method.
- Additional inference methods.
- Top career playoff risers.

---

## ⚠️ Limitations  
- One-season focus (RS + PO data required).  
- Defensive impact partially captured (STL/BLK noisy) - more comprehensive defensive metrics are required to fully capture defensive impact.  
- Opponent quality not yet included.  

---

## 📂 Usage  
```bash
# Run the complete analysis
python model.py

# Output: rankings.csv with playoff impact scores
```

---

### Why This Matters  
This model doesn’t just say “Player X scored more in the playoffs.” It asks:  
- *Relative to expectations, who truly elevated?*  
- *Did stars hold up under pressure, or fade?*  

Context is crucial in judging performance; improvements or declines should be treated dynamically, since raw numbers do not tell the full story.
The model bridges the gap between **basketball intuition** and **data-driven evidence**, making playoff debates more grounded and insightful.  
