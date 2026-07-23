[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vfT9PNyy)
# Assignment 4 – Survival Analysis

## 🎯 Goal

To independently implement and compare three survival analysis methods:

- **Kaplan–Meier Analysis** – Non-parametric survival curves
- **Cox Proportional Hazards Model** – Semi-parametric regression
- **Random Survival Forest** – Machine learning approach

You will generate survival curves, compute hazard ratios, and evaluate predictive performance using real clinical data.

---

## 🧬 Dataset Description

This assignment uses the **RADCURE dataset** from The Cancer Imaging Archive (TCIA).

**RADCURE** is a large head and neck cancer cohort with CT images and comprehensive clinical data collected from 2005-2017 at the University Health Network in Toronto, Canada.

**Dataset Properties**

- **Patients**: 3,346 head and neck cancer cases
- **Primary site**: Oropharyngeal cancer (50%), larynx (25%), nasopharynx (12%), hypopharynx (5%)
- **Follow-up**: Median 5 years, 60% alive at last follow-up
- **Demographics**: Median age 63, 80% male
- **Staging**: 7th edition TNM staging system

**Clinical Data Includes**

- **Survival outcomes**: Time to death/last follow-up, vital status
- **Demographics**: Age, sex
- **Diagnosis**: Cancer subsite, TNM stage, HPV status
- **Treatment**: Radiation dose, chemotherapy, surgery
- **Follow-up**: Disease recurrence, metastasis

**Data Access**

1. **Download from TCIA**: [RADCURE Collection](https://www.cancerimagingarchive.net/collection/radcure/)
   - Click "Clinical data" row → Download XLSX file (463KB)
   - File: `RADCURE-clinical-data.xlsx`

2. **Or download from Blackboard**: Clinical data file is posted in the assignment folder

**Citation Required**

When using RADCURE data, you must cite:

> Welch, M. L., Kim, S., Hope, A., Huang, S. H., Lu, Z., Marsilla, J., Kazmierski, M., Rey-McIntyre, K., Patel, T., O'Sullivan, B., Waldron, J., Kwan, J., Su, J., Soltan Ghoraie, L., Chan, H. B., Yip, K., Giuliani, M., Princess Margaret Head And Neck Site Group, Bratman, S., … Tadic, T. (2023). Computed Tomography Images from Large Head and Neck Cohort (RADCURE) (Version 4) [Dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/J47W-NM11

> **Note:** For this assignment, you only need the **clinical data** (XLSX file), not the CT images.

---

## 🔬 Required Models

You will implement three survival analysis approaches:

### 1. Kaplan–Meier Analysis

**What it does:** Estimates survival probability over time for different patient groups.

**You must:**
- Generate survival curves comparing ≥2 patient groups
  - Examples: age groups (< 60 vs ≥ 60), treatment arms, tumor stage
- Compute log-rank test to compare survival distributions
- Create a publication-quality plot

**Outputs:**
```
km_survival_plot.png    # Survival curves with confidence intervals
```

**Example grouping variables:**
- Age: younger vs older patients
- Treatment: chemotherapy vs radiation vs combination
- Stage: early (I/II) vs advanced (III/IV)

---

### 2. Cox Proportional Hazards Model

**What it does:** Models the relationship between covariates and survival time using hazard ratios.

**You must:**
- Include **≥3 covariates** in your model
  - Mix of continuous (age, biomarker levels) and categorical (stage, treatment)
- Compute hazard ratios with 95% confidence intervals
- Test the **proportional hazards assumption**
  - Use Schoenfeld residuals test
  - Report p-values for each covariate

**Outputs:**
```
cox_summary.csv         # Coefficients, hazard ratios, p-values, confidence intervals
```

**Example covariates:**
- Age (continuous)
- Stage (categorical: I, II, III, IV; best to keep numbers only i.e., 1, 2, ... versus 1A, 1B, ...)
- TX Modality (categorical)
- ECOG PS (ordinal)
- Smoking PY (continuous)

---

### 3. Random Survival Forest (RSF)

**What it does:** Machine learning ensemble method that handles non-linear relationships and interactions.

**You must:**
- Train a Random Survival Forest model
- Compute **concordance index (C-index)** to evaluate predictive accuracy
- Extract **variable importance** scores
- Visualize feature importance

**Outputs:**
```
rsf_importance.png      # Bar chart of feature importance
```

**Interpretation:**
- C-index = 0.5: Random predictions
- C-index = 1.0: Perfect predictions
- Typical clinical models: 0.6–0.75

---

## 📊 Workflow Summary

Your analysis will follow these steps:

1. **Load and prepare data**
   - Read survival dataset
   - Handle categorical encoding
   - Verify time/event columns

2. **Kaplan–Meier Analysis**
   - Define comparison groups
   - Fit KM estimator for each group
   - Generate survival plot
   - Run log-rank test

3. **Cox Proportional Hazards**
   - Select ≥3 covariates
   - Fit Cox model
   - Extract hazard ratios
   - Test PH assumption

4. **Random Survival Forest**
   - Prepare feature matrix
   - Train RSF model
   - Compute C-index
   - Extract importance scores

5. **Generate all outputs**
   - Save plots (`.png`)
   - Save model summary (`.csv`)

---

## ✅ Autograding & Verification

Automated tests run when you push your code. The autograder verifies:

### File Outputs Exist
- `km_survival_plot.png`
- `cox_summary.csv`
- `rsf_importance.png`

### Metrics Validity
- **Hazard ratios**: HR > 0 (hazard ratios are always positive)
- **C-index**: 0 ≤ C-index ≤ 1

### Model Execution
- **KM curves**: ≥2 groups compared
- **Cox model**: ≥3 predictors included
- **RSF**: Variable importance values returned

The GitHub component is **not graded for**:
- Code style or efficiency
- Statistical interpretation
- Model optimization

It **verifies that**:
- You independently implemented all three models
- Outputs were generated from your code
- Results are computationally valid

> **Important:** If the GitHub Classroom component is not completed, marks associated with uploaded outputs and related questions may be deducted or withheld.

---

## 🧱 Repository Structure

```text
assn4-survival-analysis/
├── .github/
│   └── workflows/classroom.yml        # CI workflow for autograding
├── students/                          # Your implementations go here
│   ├── kaplan_meier.py               # KM curves + log-rank test
│   ├── cox_model.py                  # Cox PH regression
│   ├── random_survival_forest.py     # RSF implementation
│   └── visualization.py              # Plotting helpers
├── tests/                            # pytest tests used by CI
│   ├── conftest.py                   # Test fixtures
│   └── test_submission.py            # Autograder checks
├── notebooks/                        # Optional analysis notebook
│   └── survival_analysis.ipynb
├── data/                             # Dataset (or loading instructions)
│   └── README.md
├── outputs/                          # Generated files go here
├── requirements.txt                  # Python dependencies
├── validate_submission.py            # Local verification script
├── AI_DISCLOSURE.md                  # AI usage disclosure
└── README.md                         # This file
```

---

## 🚀 Getting Started

### 1. Set up your environment

```bash
# Clone your student repository
git clone <your-repo-url>
cd assn4-survival-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download RADCURE data

**Option A: From TCIA**
1. Visit [RADCURE Collection](https://www.cancerimagingarchive.net/collection/radcure/)
2. Scroll to "Data Access" → Find "Clinical data" row
3. Click DOWNLOAD button (463KB XLSX)
4. Save as `data/RADCURE-clinical-data.xlsx`

**Option B: From Blackboard**
- Download from Assignment 4 folder → Save in `data/` directory

See [data/README.md](data/README.md) for detailed instructions.

### 3. Explore the notebook (optional)

```bash
jupyter notebook notebooks/survival_analysis.ipynb
```

### 3. Implement the three modules

Edit files in `students/`:
- `kaplan_meier.py` – Implement `fit_km()` and `logrank_test()`
- `cox_model.py` – Implement `fit_cox()` and `test_ph_assumption()`
- `random_survival_forest.py` – Implement `fit_rsf()` and `get_importance()`

### 4. Test locally

```bash
# Run all tests
pytest tests -v

# Run specific test
pytest tests/test_submission.py::test_kaplan_meier -v

# Validate outputs
python validate_submission.py
```

### 5. Generate outputs

Run your analysis script or notebook to create:
- `outputs/km_survival_plot.png`
- `outputs/cox_summary.csv`
- `outputs/rsf_importance.png`

### 6. Complete AI disclosure

Edit `AI_DISCLOSURE.md` to document any AI assistance used.

### 7. Submit

```bash
git add .
git commit -m "Complete survival analysis assignment"
git push origin main
```

Tests run automatically via GitHub Actions.

---

## 📚 Key Concepts

### Censoring

**Right-censored data:** Patient still alive at study end or lost to follow-up.

```python
# time=365, event=0 → Patient alive at 1 year (censored)
# time=180, event=1 → Patient died at 6 months (event)
```

### Kaplan–Meier Estimator

Non-parametric method that estimates survival function from censored data.

**Survival probability:** $S(t) = P(\text{survival time} > t)$

### Hazard Ratio (Cox Model)

**HR = 1:** No effect  
**HR > 1:** Increased risk (worse survival)  
**HR < 1:** Decreased risk (better survival)

**Example:**
- HR = 2.5 for chemotherapy → 2.5× higher hazard of death vs control
- HR = 0.6 for early stage → 40% reduction in hazard vs late stage

### Proportional Hazards Assumption

**Assumption:** Hazard ratio remains constant over time.

**Test using Schoenfeld residuals:**
- p > 0.05: Assumption satisfied
- p < 0.05: Assumption violated (consider stratification or time-varying effects)

### Concordance Index (C-index)

**Definition:** Probability that model correctly orders pairs of patients by survival time.

Similar to AUC-ROC but for survival data.

---

## 🔧 Tools & Libraries

This assignment uses:

- **lifelines** – KM curves, Cox model, statistical tests
- **scikit-survival** – Random Survival Forest, C-index
- **pandas** – Data manipulation
- **matplotlib/seaborn** – Visualization
- **numpy** – Numerical operations

See `requirements.txt` for complete dependencies.

---

## 📖 References

**Tutorials:**
- [lifelines documentation](https://lifelines.readthedocs.io/)
- [scikit-survival guide](https://scikit-survival.readthedocs.io/)

**Theory:**
- Klein & Moeschberger: *Survival Analysis*
- Collett: *Modelling Survival Data in Medical Research*

---

## 💡 Tips

**Data Preparation:**
- Ensure time > 0 for all observations
- Event indicator: 1 = event occurred, 0 = censored
- Handle missing values before modeling

**Model Selection:**
- Use KM for simple group comparisons
- Use Cox when you want interpretable hazard ratios
- Use RSF when relationships are non-linear or complex

**Common Issues:**
- **Convergence errors:** Check for perfect separation or collinearity
- **PH assumption violated:** Consider stratification or time-varying coefficients
- **Low C-index:** May need feature engineering or more predictive variables

---

## 📝 Submission Checklist

- ✅ All three models implemented in `students/`
- ✅ Three output files generated in `outputs/`
- ✅ Local tests pass (`pytest tests -v`)
- ✅ `AI_DISCLOSURE.md` completed
- ✅ Code pushed to GitHub
- ✅ GitHub Actions tests pass (check repository Actions tab)

---

## ❓ FAQ

**Q: Can I use different grouping variables for KM?**  
A: Yes, any clinically meaningful grouping with ≥2 categories.

**Q: Which covariates should I include in Cox model?**  
A: Mix continuous and categorical, ≥3 total. Choose clinically relevant features.

**Q: What if my C-index is low?**  
A: That's okay! The autograder only checks that it's in valid range [0,1].

**Q: Can I use a different dataset?**  
A: Yes, as long as it has time, event, and ≥3 covariates.

**Q: Do I need to interpret the results?**  
A: Not for GitHub submission. Interpretation is assessed separately on Blackboard.

---

Good luck! Remember: the goal is to execute the models correctly and generate valid outputs. Statistical interpretation will be assessed separately.
