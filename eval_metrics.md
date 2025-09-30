
# **1. Evaluation Metrics in NLP**

## **(a) ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**

**What it is:**

* ROUGE measures the **overlap between generated text and reference text(s)**, based on n-grams, word sequences, or longest common subsequences.
* Originally designed for **summarization evaluation**, but widely used in text generation tasks.

**Types of ROUGE:**

1. **ROUGE-N**: Overlap of n-grams (e.g., ROUGE-1 for unigrams, ROUGE-2 for bigrams).
2. **ROUGE-L**: Based on the **Longest Common Subsequence (LCS)** between candidate and reference.
3. **ROUGE-W**: Weighted LCS, penalizes shorter subsequences less.
4. **ROUGE-S / ROUGE-SU**: Measures **skip bigram** overlap (allowing gaps between words).

**Importance:**

* Captures **content recall** well (especially useful in summarization where covering key information is important).
* Multiple variants allow fine-grained evaluation (lexical overlap, sequence similarity, etc.).
* Easy to compute, interpretable.

**Drawbacks:**

* **Surface-level metric**: Only matches words/phrases, ignoring meaning.
* Sensitive to **synonyms/paraphrases** (e.g., “doctor” vs. “physician” gets penalized).
* More focused on **recall**, not precision (good if you want maximum coverage, less good if conciseness matters).
* May overvalue longer outputs that include more overlapping tokens.

---

## **(b) BLEU (Bilingual Evaluation Understudy)**

**What it is:**

* BLEU measures **n-gram precision**: how many n-grams in the candidate appear in the reference(s).
* Widely used in **machine translation** evaluation.
* Includes a **brevity penalty** to avoid rewarding overly short outputs.

**Types within BLEU:**

* BLEU-1, BLEU-2, BLEU-3, BLEU-4 → based on unigram, bigram, trigram, 4-gram precision.
* Typically, **BLEU-4** (up to 4-grams) is reported.

**Importance:**

* First major automatic metric to correlate with human judgment in MT.
* Balances **precision** (quality of chosen words) and **brevity penalty** (discourages overly short outputs).
* Still widely reported in NLP papers for comparability.

**Drawbacks:**

* **Insensitive to meaning**: Only measures surface n-gram overlap.
* Penalizes **valid paraphrases** (e.g., "He is quick" vs. "He is fast").
* Works best with **multiple references**—with one reference, it’s too harsh.
* Poor for **open-ended tasks** like dialogue or abstractive summarization, since exact overlap is rare.
* Correlation with human judgment is weaker for high-quality outputs.

---

## **(c) BERTScore**

**What it is:**

* A **semantic similarity metric** that uses contextual embeddings (from models like BERT, RoBERTa) instead of raw n-gram overlap.
* For each token in the candidate, finds the **most similar token embedding in the reference** (and vice versa), then computes precision, recall, and F1.

**Variants:**

* **BERTScore-P (Precision):** Emphasizes correctness of chosen words.
* **BERTScore-R (Recall):** Emphasizes coverage of reference meaning.
* **BERTScore-F1:** Balanced measure.
* Can use different pretrained models (e.g., multilingual BERT for cross-lingual tasks).

**Importance:**

* **Captures semantics** (e.g., "doctor" ≈ "physician"), making it robust to paraphrases.
* Stronger correlation with human judgment than BLEU/ROUGE, especially for **abstractive and generative tasks**.
* Model-based → can evolve as better embeddings appear.

**Drawbacks:**

* **Computationally expensive** compared to ROUGE/BLEU (requires running embeddings).
* **Bias from pretrained models** (domain mismatch can reduce reliability).
* Less interpretable: numbers don’t correspond directly to word overlap.
* Sensitive to **model choice** (BERT vs. RoBERTa vs. DeBERTa may give different scores).

---

## **Critical Comparison**

| Metric        | Focus                                             | Strengths                                     | Weaknesses                                                 |
| ------------- | ------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| **ROUGE**     | Recall & n-gram overlap                           | Simple, interpretable, good for summarization | Misses synonyms/paraphrases, recall-heavy                  |
| **BLEU**      | Precision & n-gram overlap (with brevity penalty) | Standard for MT, balances length & precision  | Poor for open-ended text, bad with one reference           |
| **BERTScore** | Semantic similarity via embeddings                | Captures meaning, robust to paraphrasing      | Computationally heavy, model-dependent, less interpretable |

---

 **In practice:**

* Use **ROUGE** for summarization benchmarks.
* Use **BLEU** for machine translation (but report others too).
* Use **BERTScore** for abstractive/natural generation tasks where semantics matter more than exact wording.



---

# **Formulas**

---

## **(a) ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**

**ROUGE-N (n-gram recall):**

$$
\text{ROUGE-N} = \frac{\sum_{S \in \{\text{Reference Summaries}\}} \sum_{gram_n \in S} \min(\text{Count}_{\text{candidate}}(gram_n), \text{Count}_{S}(gram_n))}{\sum_{S \in \{\text{Reference Summaries}\}} \sum_{gram_n \in S} \text{Count}_{S}(gram_n)}
$$

* Measures **recall of n-grams** in candidate vs. reference(s).

**ROUGE-L (Longest Common Subsequence):**

$$
\text{ROUGE-L} = \frac{LCS(X, Y)}{\text{length}(Y)}
$$

* Where $LCS(X,Y)$ is the length of the longest common subsequence between candidate $X$ and reference $Y$.  
* Recall-oriented, but also precision/F1 versions exist.

---

## **(b) BLEU (Bilingual Evaluation Understudy)**

**Modified n-gram precision:**

$$
p_n = \frac{\sum_{C \in \text{Candidates}} \sum_{gram_n \in C} \min(\text{Count}_{C}(gram_n), \max_{R \in \text{References}} \text{Count}_{R}(gram_n))}{\sum_{C \in \text{Candidates}} \sum_{gram_n \in C} \text{Count}_{C}(gram_n)}
$$

* Clips counts to avoid over-counting repeated words.

**Final BLEU score (up to N-grams, usually N=4):**

$$
\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^N w_n \log p_n \right)
$$

* $w_n = \tfrac{1}{N}$ (uniform weights).  

**Brevity Penalty (BP):**

$$
BP =
\begin{cases}
1 & \text{if } c > r \\
e^{(1-r/c)} & \text{if } c \leq r
\end{cases}
$$

Where:  
- $c$ = length of candidate  
- $r$ = length of reference  

---

## **(c) BERTScore**

* Uses embeddings from a pretrained LM (e.g., BERT).

**Cosine similarity between candidate and reference tokens:**

$$
\text{sim}(x_i, y_j) = \frac{E(x_i) \cdot E(y_j)}{\|E(x_i)\|\|E(y_j)\|}
$$

Where:  
- $E(x_i)$ = embedding of token $x_i$ in candidate  
- $E(y_j)$ = embedding of token $y_j$ in reference  

**Precision (P), Recall (R), and F1:**

$$
P = \frac{1}{|X|} \sum_{x_i \in X} \max_{y_j \in Y} \text{sim}(x_i, y_j)
$$

$$
R = \frac{1}{|Y|} \sum_{y_j \in Y} \max_{x_i \in X} \text{sim}(y_j, x_i)
$$

$$
F1 = \frac{2PR}{P+R}
$$

* Captures **semantic similarity** at token level.



---

# **2. Reference-Free Evaluations in NLP**

### **What they are**

* **Reference-free evaluations** assess the **quality of generated text without requiring human-written reference outputs**.
* Instead of comparing candidate text to one or more references (as in ROUGE, BLEU, BERTScore), they directly measure aspects like:

  * **Fluency** (does the text look grammatically correct/natural?)
  * **Coherence** (is the text logically consistent?)
  * **Factual consistency** (is the text accurate with respect to a source document?)
  * **Diversity and originality** (is it repetitive or generic?).

These are especially useful in **open-ended tasks** (dialogue, summarization, creative generation), where multiple valid outputs exist and reference-based overlap metrics are too restrictive.

---

### **Example: Perplexity**

One widely used **reference-free metric** is **perplexity**.

* **Definition:** Perplexity measures how well a language model predicts the generated sequence.
* For a candidate sequence $x = (x_1, x_2, ..., x_T)$:

$$
\text{Perplexity}(x) = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t})\right)
$$

* Lower perplexity → the text is more fluent and aligned with natural language patterns learned by the model.

---

### **Advantages of Reference-Free Evaluation**

1. **No need for human references**

   * Saves time and cost (especially for large datasets).
   * Works even when multiple diverse valid answers exist.

2. **Better at measuring fluency & grammaticality**

   * Perplexity, for example, directly measures whether text is “natural” under a language model.

3. **Useful in low-resource settings**

   * Where reference datasets are hard to obtain (e.g., new domains or languages).

---

### **Disadvantages (vs. ROUGE, BLEU, BERTScore)**

1. **Does not measure semantic adequacy**

   * A fluent but factually wrong summary can still have low perplexity.
   * BLEU/ROUGE/BERTScore check if generated text **matches the intended meaning** of references.

2. **Model bias**

   * Metrics like perplexity depend on the underlying LM used for evaluation (e.g., GPT-2 vs. GPT-4 may judge differently).

3. **Poor interpretability**

   * Scores like perplexity don’t map directly to human judgment (e.g., what does “PP = 30” mean for quality?).

4. **Incomplete quality assessment**

   * Fluency is not enough; tasks like translation or summarization need to be **semantically faithful**, which reference-based metrics can capture.

---

### **Quick Comparison**

| Metric Type         | Example                                                     | Pros                                                    | Cons                                                    |
| ------------------- | ----------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| **Reference-based** | ROUGE, BLEU, BERTScore                                      | Captures semantic overlap, interpretable, task-specific | Needs human references, penalizes paraphrasing          |
| **Reference-free**  | Perplexity, QA-based factuality checks, embedding coherence | No need for references, useful in open-ended tasks      | Doesn’t ensure correctness or adequacy, model-dependent |

---





