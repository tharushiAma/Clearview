# Chapter 7: Explainability Framework

## 7.1 Motivation

Mixed sentiment resolution is a novel claim. Without explainability, the following questions cannot be answered empirically:
1. Does the model actually separate aspect-specific signals or does it conflate them?
2. Which tokens drive each aspect-specific prediction?
3. Does the GCN contribute to resolution or is the attention module sufficient?

Multi-method explainability provides complementary evidence from different theoretical frameworks.

## 7.2 Attention Visualization

Method: Extract MHA attention weights from AspectAwareRoBERTa.aspect_attention.

The aspect embedding is the query, so attention weights (batch, 1, seq_len) directly show which tokens the model uses to form its aspect-specific sentiment representation.

Limitation: Attention weights are not necessarily proportional to feature importance (Wiegreffe and Pinter, 2019). They show what the model attends to, not necessarily what determines the output.

Implementation: predict(text, aspect, return_attention=True) returns tokens and attention weights.

## 7.3 LIME

Reference: Ribeiro et al., Why Should I Trust You? KDD 2016.

Method: Randomly remove words from the input and observe prediction changes. Fit a local linear model to the perturbed inputs to estimate word-level importances.

Implementation: explain_with_lime(text, aspect) in inference.py uses LimeTextExplainer with the same SentimentPredictor.predict() call path as production, ensuring the explanation reflects actual model behavior.

Properties: Model-agnostic, human-interpretable word-level importances, computationally expensive (requires multiple forward passes).

## 7.4 SHAP

Reference: Lundberg and Lee, A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.

Method: Shapley values from cooperative game theory. Each token contribution is its average marginal contribution across all possible token coalitions.

Implementation: explain_with_shap(text, aspect) uses shap.Explainer with Partition algorithm. Baseline: randomly masked versions of input. Reports per-token SHAP values for predicted class.

Properties: Satisfies efficiency, symmetry, dummy, and consistency axioms. Globally consistent. More computationally expensive than LIME.

## 7.5 Integrated Gradients

Reference: Sundararajan et al., Axiomatic Attribution for Deep Networks. ICML 2017.

Formula: IG(x) = (x - x_baseline) * integral_0_1 (dF(x_baseline + alpha*(x - x_baseline)) / dx) d_alpha

Axioms satisfied:
- Sensitivity: if a feature changes the output, it receives nonzero attribution
- Implementation Invariance: functionally equivalent models get identical attributions
- Completeness: sum of attributions = F(x) - F(x_baseline)

Implementation: explain_with_integrated_gradients(text, aspect) in inference.py uses Captum LayerIntegratedGradients on the RoBERTa embedding layer.

Configuration:
- Baseline: all-PAD token embedding (uninformative)
- n_steps: 50 interpolation steps
- Verification: convergence delta |sum(attributions) - (F(x) - F(baseline))| should be < 0.05

Why IG is preferred for research claims: It provides mathematically verified attributions. The completeness check confirms the explanation is consistent with the model computation.

## 7.6 MSR Delta (Novel Contribution)

### Motivation

IG, LIME, and SHAP explain individual aspect predictions. MSR Delta specifically demonstrates mixed sentiment resolution: it proves the model separates aspect-specific signals rather than conflating them.

### Method

For a focus aspect A and a review text:
1. Predict baseline confidence conf_A = P(label_A | full_text)
2. For each token t_i, replace with [MASK] and re-predict conf_A_masked_i = P(label_A | text with t_i masked)
3. delta_i = conf_A - conf_A_masked_i

Interpretation:
- Large positive delta: t_i actively supports the focus aspect prediction
- Near-zero delta: t_i is irrelevant to focus aspect (even if expressing opinion about another aspect)
- Large negative delta: t_i suppresses the focus aspect prediction

### Mixed Sentiment Resolution Evidence

For a review: Great colour but the smell is awful

When focus_aspect = colour (predicted: positive):
- Expected: delta(great) >> 0, delta(colour) > 0, delta(awful) ~= 0, delta(smell) ~= 0

When focus_aspect = smell (predicted: negative):
- Expected: delta(awful) >> 0, delta(smell) > 0, delta(great) ~= 0, delta(colour) ~= 0

This pattern proves the model is NOT conflating aspects. If it were, delta(awful) would be nonzero for the colour aspect and vice versa.

### Cross-Aspect Summary

The explain_msr_delta method also prints a cross-aspect summary showing the model predictions for all other aspects simultaneously, demonstrating multi-aspect awareness.

### Implementation

explain_msr_delta(text, focus_aspect, top_k=10) in inference.py:
1. Computes baseline prediction
2. Iterates over each token, masking one at a time
3. Records confidence delta per token
4. Returns top-k positive delta tokens (most important for focus aspect)

Computational cost: O(seq_len) forward passes. Approximately 128 passes for a max-length review.

## 7.7 XAI Integration in ClearView Website

The website XAI page (/xai route) provides:
- Interactive token highlighting using IG attributions
- LIME and SHAP token visualizations (tabs)
- MSR Delta comparison showing gained/lost importance across aspects
- Raw JSON bundle for technical inspection

The backend /explain endpoint in backend_server.py routes to TrainedModelXAI which implements all methods via the trained model.
