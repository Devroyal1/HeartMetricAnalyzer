## Performance Metrics

This project uses several key performance metrics to evaluate machine learning models. Understanding these metrics is essential for interpreting model results and making informed decisions.

---

### Accuracy

**Definition:**  
Accuracy measures the proportion of correct predictions made by the model out of all predictions. It answers the question: "How often is the model correct?"

**Formula:**  
\[
\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}} \times 100\%\
\]

**Example:**  
Suppose a hospital uses a model to predict disease presence in 100 patients:
- Correct predictions: 90 (80 true positives, 10 true negatives)
- Incorrect predictions: 10 (5 false positives, 5 false negatives)

\[
\text{Accuracy} = \frac{90}{100} \times 100\% = 90\%\
\]

**Note:**  
High accuracy can be misleading if the dataset is imbalanced (e.g., most patients do not have the disease).

---

### Precision

**Definition:**  
Precision evaluates the accuracy of positive predictions. It tells us what proportion of predicted positives are actually correct.

**Formula:**  
\[
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} \times 100\%\
\]

**Example:**  
A medical test predicts 50 patients as positive:
- True positives: 45
- False positives: 5

\[
\text{Precision} = \frac{45}{45 + 5} \times 100\% = 90\%\
\]

**Importance:**  
High precision is crucial in medical diagnostics to avoid unnecessary treatment for healthy patients.

---

### Sensitivity (Recall)

**Definition:**  
Also known as recall or true positive rate, sensitivity measures the model’s ability to correctly identify actual positive cases.

**Formula:**  
\[
\text{Sensitivity} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} \times 100\%\
\]

**Example:**  
Out of 100 patients with the disease:
- True positives: 80
- False negatives: 20

\[
\text{Sensitivity} = \frac{80}{80 + 20} \times 100\% = 80\%\
\]

**Importance:**  
High sensitivity ensures that most actual cases are detected, minimizing missed diagnoses.

---

### F1 Score

**Definition:**  
The F1 Score combines precision and recall into a single metric, providing a balance between the two. It is especially useful when you need to balance false positives and false negatives.

**Formula:**  
\[
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\
\]

**Range:**  
- 0: Worst performance (no correct positive predictions)
- 1: Perfect precision and recall

**Interpretation:**
- F1 Score > 0.9: Excellent
- 0.8 – 0.9: Good
- 0.5 – 0.8: Average
- < 0.5: Poor

---
