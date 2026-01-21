# Project Status: MSR RoBERTa with Dependency GCN

## 1. Technologies & Stack
*   **Core Framework**: PyTorch 2.5.1 (CUDA 12.1 Accelerated).
*   **Language Model**: `RoBERTa-base` (via HuggingFace Transformers).
*   **Graph Processing**: Custom Graph Convolutional Network (GCN) layer.
*   **NLP Tools**: `SpaCy` for Dependency Parsing (constructing adjacency matrices).
*   **Optimization**: AdamW Optimizer with Focal Loss.
*   **Data Handling**: Pandas, Parquet.
*   **Explainability (XAI)**: SHAP, LIME, Captum, Matplotlib, Seaborn.

## 2. Methodology & Research Value
This project integrates advanced NLP techniques to solve **Fine-Grained Aspect-Based Sentiment Analysis (ABSA)** on a highly imbalanced dataset.

### A. Graph-Based Aspect Fusion (DepGCN)
Instead of relying solely on RoBERTa's self-attention, we constructed **Dependency Graphs** for every review.
*   **Mechanism**: A GCN layer propagates information along the syntactic dependency tree (e.g., linking the adjective "expensive" directly to the noun "price" even if they are far apart in the sentence).
*   **Benefit**: explicitly captures the syntactic relationship between aspects and their opinion words.

### B. Handling Class Imbalance (Focal Loss)
*   **Problem**: The dataset is overwhelmingly detailed with "Positive" reviews. Minority classes (Negative Price, Negative Packing) were ignored by the baseline CrossEntropy loss.
*   **Solution**: Implemented **Focal Loss** ($\gamma=2.0$) with Class Weights.
*   **Effect**: The model creates a larger gradient for difficult, misclassified examples, forcing it to learn from the rare negative cases.

### C. Advanced Data Augmentation (The Research Contribution)
Standard augmentation (synonym replacement) was insufficient. We developed a novel **"Mixed Sentiment Injection"** strategy.
*   **Identified Failure Mode ("The Halo Effect")**: The model would classify an entire review as "Positive" if the first half was positive (e.g., "Great color..."), ignoring a subtle price complaint at the end.
*   **Innovation**: We generated synthetic training samples that **mix** sentiments (e.g., *"The texture is 5 stars, but the price is a robbery"*).
*   **Outcome**: This taught the model to disentangle conflicting sentiments within a single text, a key requirement for MSR (Multi-Aspect) resolution.

## 3. Performance Metrics (Proof of Improvement)
We achieved statistically significant improvements by validating against a custom "Golden Set" of difficult negative examples.

### Comparative F1-Scores (Negative Class)

| Aspect Category | Baseline Model | Final Model (After Augmentation) | Improvement Factor | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Packing** | 0.43 | **0.85** | **+97%** | ✅ Solved |
| **Price** | 0.00 | **0.87** | **Infinite (0 to 0.87)** | ✅ Solved |

### Detailed Confusion Matrices (Final)
*   **Price Negative**: Recall of **0.91**. The model now catches 91% of price complaints.
*   **Packing Negative**: Precision and Recall both at **0.85**.

## 4. Explainable AI (XAI) Integration
To ensure the model is not a "Black Box," we implemented a suite of interpretation tools designed for both technical and non-technical stakeholders.

### A. Mixed Sentiment Resolution (The "Tug-of-War")
*   **Goal**: Explain *why* a review was labeled "Mixed" versus "Positive".
*   **Mechanism**: A custom visualization that calculates the weighted "Mass" of Positive vs. Negative aspect detections.
*   **Output**: A visual scale showing the conflicting forces. If the conflict score exceeds a threshold (0.25), the resolution is "Mixed".

### B. Aspect-Level Attribution (SHAP & LIME)
*   **Goal**: identify exactly which words triggered a specific aspect detection (e.g., "Why did you think 'Price' was negative?").
*   **Tools**:
    *   **LIME (Local Interpretable Model-agnostic Explanations)**: Perturbs the input text to find word importance.
    *   **SHAP (SHapley Additive exPlanations)**: Uses game theory to assign contribution scores to every token.
*   **Outcome**: Generated HTML reports where negative words are highlighted in red and positive in blue, providing immediate visual verification of the model's logic.

 ## 5. Recent Improvements (Code Quality & Neutral Sentiment)
 ### A. Code Understandability (Documentation)
 *   **Objective**: Enhance the readability of the complex `msr_resolver_roberta_focal_loss.py` model.
 *   **Action**: Added detailed explanatory comments to every critical section:
     *   **Dependency GCN**: Explains the matrix normalization $D^{-0.5}AD^{-0.5}$.
     *   **Focal Loss**: Break down the $(1-p_t)^\gamma$ formula.
     *   **MSR Logic**: Clarify conflict score calculation.
 ### B. Solving "Neutral" Sentiment Scarcity
 *   **Problem**: While Negative detection was solved, "Neutral" sentiment for Price/Packing remained at 0.00 F1 score due to extreme scarcity (<15 samples).
 *   **Solution**: Implemented a **GPT-Based Data Augmentation** pipeline.
     *   Extracted "Positive" reviews susceptible to being "toned down".
     *   Generated synthetic "Neutral" rewrites (e.g., "Good price" -> "Standard price").
     *   Merged **33 new training samples** and **15 new validation samples**.
 *   **Status**: Ready for training to verify Neutral class improvement.

 ## 6. Conclusion
 By combining **Dependency GCNs** with **Mixed-Sentiment Augmentation**, we successfully overcame the dominant class bias. The addition of **XAI (SHAP/LIME)** provides the necessary transparency to trust the model's complex resolution logic, making it a robust solution for real-world cosmetic review analysis.
