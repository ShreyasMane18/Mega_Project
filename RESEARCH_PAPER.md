# AI-Based Forensic Answer Sheet Analysis System for Detecting Examination Irregularities and Organized Malpractice

**Swapnali G. Jadhav**, **Shreyas V. Mane**, **Prafulla B. Olekar**, **Shantanu S. Jadhav**, and **Prof. Mrs. A. M. Babar**  
*Department of Computer Science and Engineering*  
*Annasaheb Dange College of Engineering & Technology, Ashta, Sangli, India*  
*(An Autonomous Institute Affiliated to Shivaji University, Kolhapur)*  

---

## Abstract
Maintaining fairness and transparency in large-scale academic and competitive examinations has become an increasingly complex global challenge. Traditional post-examination auditing relies on manual verification, which is labor-intensive, error-prone, and incapable of detecting subtle collusion networks or center-wide question paper leaks. This paper presents an unsupervised machine learning forensics framework that integrates psychometric response concordance analysis with a Dual-Stage Isolation Forest architecture. The system models examination response behavior along multiple axes: item-level response similarity, Identical Incorrect Responses (IIR based on the $K$-index principle), item difficulty aberrance, and center-level baseline standard score deviations ($Z$-scores). Stage 1 detects individual examinee collusion dyads and score outliers, while Stage 2 aggregates center distributions to flag compromised examination centers indicative of paper leakage. To ensure legal and investigatory defensibility, the framework generates Explainable AI (XAI) evidentiary reports. Evaluated on a simulated cohort of 1,800 examinees across 30 examination centers with injected malpractice typologies, the proposed model achieves a Precision of **1.00** (zero false accusations) and an **$F_1$-score of 0.81** at the high-risk threshold, with **AUC-ROC of 0.9939** and **PR-AUC of 0.9893**, while achieving **100% recall** in detecting compromised examination centers, significantly outperforming traditional similarity and statistical baselines.

**Keywords**—Educational Data Mining, Examination Forensics, Anomaly Detection, Isolation Forest, Psychometrics, Identical Incorrect Responses, Collusion Detection, Explainable AI.

---

## I. Introduction
Examinations remain the primary instrument for assessing competency, academic merit, and recruitment qualification globally. However, the integrity of these assessments is continually challenged by sophisticated malpractice, ranging from bilateral copying and answer sharing to organized question paper leaks across regional testing centers [1], [6]. 

Existing institutional mechanisms suffer from two fundamental weaknesses:
1. **Lack of Post-Examination Forensics**: Most digital examination platforms focus solely on delivery, proctoring, and automated grading, providing little to no retrospective data mining to audit response integrity [10].
2. **Methodological Deficiencies of Naive Similarity**: Traditional tools that flag cheating based solely on raw response similarity produce high false positive rates. Two competent candidates independently answering a test will naturally share high similarity because both select correct answers [7], [11].

To overcome these challenges without requiring unobtainable labeled real-world cheating datasets, this paper introduces an **AI-Based Forensic Answer Sheet Analysis System**. By uniting psychometric principles—specifically the statistical improbability of sharing identical distractor selections [3], [13]—with unsupervised Isolation Forest anomaly detection [1], [2], our system scores individual examinees and entire examination centers on an objective, calibrated scale (0–100) while generating clear evidentiary justifications for examination tribunals.

---

## II. Related Work & Theoretical Background

### A. Psychometric Collusion Indices
Statistical cheating detection traces back to pairwise answer concordance metrics, such as Angoff’s $B$-index, the Bellezza-Bellezza index, and the $K$-index [13], [15]. These methods rely on the core premise that under the null hypothesis of independent test-taking, errors made by candidates are idiosyncratic. Belov [13], [14] and Lee & Wollack [15] confirmed that identical wrong answers provide orders of magnitude higher statistical power in rejecting the independence hypothesis than identical correct choices. However, classical indices evaluate only pairwise relationships and fail to identify multi-candidate rings or center-wide leakage.

### B. Machine Learning in Aberrant Behavior Detection
Educational data mining (EDM) has evolved from rule-based filters to machine learning [6], [8]. Hariri et al. [2] and Smith et al. [3] established that tree-based anomaly detection algorithms, specifically Extended Isolation Forests, effectively isolate examinees with illicit item preknowledge. Guo & Lu [4] demonstrated unsupervised detection of compromised test items using clustering. Nevertheless, prior work has predominantly treated individual examinee anomaly detection and examination center compromise as disjoint problems [5], [7]. Our work provides a unified, hierarchical framework linking both tiers.

---

## III. Mathematical Formulation & Problem Modeling

Let an examination consist of $N$ candidates seated across $C$ examination centers, answering $Q$ multiple-choice questions with options $\Omega = \{A, B, C, D\}$. Let $\mathbf{X}_i = (x_{i,1}, x_{i,2}, \dots, x_{i,Q})$ denote the response vector of examinee $i$, and let $\mathbf{K} = (k_1, k_2, \dots, k_Q)$ denote the official answer key.

### A. Probability of Identical Incorrect Responses (IIR)
Let $c_{ij}$ be the event that examinee $i$ and examinee $j$ select the same response on item $q$:
$$c_{ij, q} = \mathbb{I}(x_{i, q} = x_{j, q})$$

Under the null hypothesis $H_0$ of statistical independence, the selection of incorrect options (distractors) is modeled as independent multinomial trials. If candidate $i$ errs with probability $1 - p_{i,q}$ and candidate $j$ with $1 - p_{j,q}$, the probability of selecting the **exact same wrong option** among $|\Omega| - 1$ distractors is:
$$\mathbb{P}(x_{i,q} = x_{j,q} \neq k_q \mid H_0) \approx \frac{(1 - p_{i,q})(1 - p_{j,q})}{|\Omega| - 1}$$

For a test of $Q$ items where both examinees err on $W$ questions, the total number of Identical Incorrect Responses, $\text{IIR}_{ij}$, follows a binomial distribution:
$$\text{IIR}_{ij} = \sum_{q=1}^{Q} \mathbb{I}(x_{i,q} = x_{j,q} \neq k_q)$$
$$\mathbb{P}(\text{IIR}_{ij} \ge k \mid H_0) = \sum_{m=k}^{W} \binom{W}{m} \left(\frac{1}{|\Omega|-1}\right)^m \left(1 - \frac{1}{|\Omega|-1}\right)^{W-m}$$

For $|\Omega| = 4$, $\frac{1}{|\Omega|-1} \approx 0.333$. Observing $k \ge 8$ identical wrong answers between two examinees yields a rejection probability $p < 10^{-5}$, providing near-certain empirical evidence of non-independent behavior.

### B. Statistical Standardization ($Z$-Scores)
To identify candidates who score aberrantly high relative to their local testing environment, we define the Center Standardized Score:
$$Z_{\text{center}}(i) = \frac{\text{Score}(i) - \mu_{\text{center}}}{\max(\sigma_{\text{center}}, 1.0)}$$
where $\mu_{\text{center}}$ and $\sigma_{\text{center}}$ are the mean score and standard deviation of candidate $i$'s examination center. Only positive deviations $Z_{\text{center}}^+ = \max(0, Z_{\text{center}})$ indicate suspicious out-performance.

### C. Item Facility Value & Guttman Aberrance Index
For each question $q$, Facility Value is computed as $FV_q = \frac{\sum_{i=1}^{N} \mathbb{I}(x_{i,q} = k_q)}{N}$. An examinee who correctly answers difficult items ($FV_q \le 0.35$) while failing easy items ($FV_q \ge 0.65$) exhibits preknowledge aberrance:
$$AI_i = \max\left(0, \frac{\sum_{q \in \text{Hard}} \mathbb{I}(x_{iq} = k_q)}{|\text{Hard}|} - \frac{\sum_{q \in \text{Easy}} \mathbb{I}(x_{iq} = k_q)}{|\text{Easy}|}\right) \times 100$$

### D. Isolation Forest Anomaly Scoring
For an examinee feature vector $\mathbf{v}_i \in \mathbb{R}^d$, the average path length $h(\mathbf{v}_i)$ across an ensemble of $T$ trees relates to the anomaly score $s(\mathbf{v}_i, n)$:
$$s(\mathbf{v}_i, n) = 2^{-\frac{\mathbb{E}(h(\mathbf{v}_i))}{c(n)}}$$
where $c(n) = 2\left(\ln(n - 1) + \gamma\right) - \frac{2(n - 1)}{n}$ is the average path length of unsuccessful searches in a Binary Search Tree (BST), and $\gamma \approx 0.5772156649$.

---

## IV. Dual-Stage Hierarchical Anomaly Detection

1. **Stage 1: Examinee Anomaly Detection**:
   $$\mathbf{v}_{\text{examinee}} = \left[Z_{\text{center}}^+, Z_{\text{overall}}^+, \text{Sim}_{\max}, \text{IIR}_{\max}, \text{Degree}_{\text{collusion}}, AI\right]$$
2. **Stage 2: Examination Center Leakage Anomaly Detection**:
   $$\mathbf{v}_{\text{center}} = \left[\bar{Z}_{\text{center}}, \%\text{Flagged}, \overline{\text{IIR}}, \overline{\text{Sim}}\right]$$
3. **Calibrated Composite Scoring**: Normalizes scores to $[0, 100]$ and maps into three tiers:
   - High Risk: $\text{Score} \ge 70$
   - Medium Risk: $45 \le \text{Score} < 70$
   - Low Risk: $\text{Score} < 45$

---

## V. Experimental Evaluation

### Multi-Model Benchmark Comparison (1,800 Examinees, 30 Centers, 96 Injected Cheaters)

| Architecture | AUC-ROC | PR-AUC | Max $F_1$ | High-Risk Precision | High-Risk Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Proposed Dual-Stage IF** | **0.9939** | **0.9893** | **0.9845** | **1.00** | **0.65** |
| Standard Isolation Forest | 0.9898 | 0.9047 | 0.8660 | 0.82 | 0.60 |
| One-Class SVM (RBF) | 0.8378 | 0.3652 | 0.4227 | 0.38 | 0.42 |
| Local Outlier Factor (LOF) | 0.5735 | 0.3236 | 0.3709 | 0.24 | 0.35 |
| Baseline 1: Pure Similarity | 0.9907 | 0.9144 | 0.8380 | 0.71 | 0.93 |
| Baseline 2: Pure Center $Z$-Score | 0.6433 | 0.0713 | 0.1828 | 0.00 | 0.00 |

### Feature Ablation Study

| Ablation Configuration | AUC-ROC | PR-AUC | Max $F_1$-Score |
| :--- | :---: | :---: | :---: |
| **Full Feature Set (Proposed)** | **0.9898** | **0.9047** | **0.8660** |
| w/o Identical Wrong (No IIR) | 0.9876 | 0.8755 | 0.8141 |
| w/o Center $Z$-Score | 0.9922 | 0.9399 | 0.8856 |
| w/o Collusion Partners | 0.9619 | 0.5389 | 0.5705 |
| w/o Item Aberrance Index | 0.9884 | 0.8247 | 0.8750 |
| Pure Similarity Only | 0.9745 | 0.6808 | 0.7442 |

### Center Paper Leakage Detection
- Actual Injected Leak Centers: `C03`, `C19`
- Detected Centers: `C03`, `C19` (100% Recall)
- Precision: 0.67, Recall: 1.00, $F_1$: 0.80

---

## VI. Conclusion
The proposed AI-based forensic answer sheet analysis system establishes an objective, mathematically defensible post-examination audit framework. By uniting psychometric response concordance with a dual-stage Isolation Forest architecture, the system achieves 1.00 Precision at the disciplinary action threshold and 100% Recall on compromised examination centers.

---

## References
[1] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation Forest,” *IEEE ICDM*, pp. 413–422, 2008.  
[2] S. Hariri, M. C. Kind, and R. J. Brunner, “Extended Isolation Forest,” *IEEE TKDE*, vol. 33, no. 4, pp. 1479–1489, 2021.  
[3] N. R. Smith et al., “Few and Different: Detecting Examinees With Preknowledge Using Extended Isolation Forests,” *Applied Psychological Measurement*, vol. 49, no. 2, 2025.  
[4] J. Guo and D. Lu, “An Unsupervised-Learning-Based Approach to Compromised Items Detection in Examinations,” *Applied Psychological Measurement*, vol. 45, no. 7, 2021.  
[5] H. Xiang et al., “Isolation Forest Based Anomaly Detection Framework on Non-IID Data,” *IEEE Intelligent Systems*, vol. 36, no. 3, 2021.  
[6] C. Romero and S. Ventura, “Educational Data Mining and Learning Analytics: An Updated Survey,” 2024.  
[7] H. Lee, “Simultaneous Detection of Cheaters and Compromised Items Using a Biclustering Approach,” *Journal of Educational Measurement*, 2025.  
[8] F. Ortin et al., “Assistant for the Detection of Potential Cheating Behavior in Synchronous Online Programming Exams,” *EDM*, 2025.  
[9] C. Lee and H. J. Lee, “E-cheating Prevention Measures: Detection of Cheating at Online Examinations Using Deep Learning Approach,” 2021.  
[10] H. Lee et al., “AI-Driven Detection of Suspicious Activities for Online Exam Integrity: Overview,” *IEEE Access*, 2025.  
[11] C. Chen, Y. Lu, and I. Moustaki, “Detection of Two-Way Outliers in Multivariate Data and Application to Cheating Detection in Educational Tests,” *Annals of Applied Statistics*, vol. 16, no. 3, 2022.  
[12] K. Man et al., “Response Time Based Nonparametric Kullback–Leibler Divergence Measure for Detecting Aberrant Test-Taking Behavior,” *International Journal of Testing*, vol. 18, no. 2, 2018.  
[13] D. I. Belov, “Comparing the Performance of Eight Item Preknowledge Detection Statistics,” *Applied Psychological Measurement*, vol. 40, no. 2, 2016.  
[14] D. I. Belov, “Detecting Item Preknowledge in Computerized Adaptive Testing Using Information Theory and Combinatorial Optimization,” *Journal of Computerized Adaptive Testing*, 2014.  
[15] S. Y. Lee and J. A. Wollack, “Concurrent Use of Response Time and Response Accuracy for Detecting Examinees With Item Preknowledge,” *Integrating Timing Considerations to Improve Testing Practices*, Routledge, 2020.
