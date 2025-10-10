# Full Fine-tuning & Baseline Performance



### finetuning eval and train metrics
![](/plots/train_plot_1.png)
![](/plots/eval_plot_1.png)

> To view the graphs above you can use the following command
```bash
tensorboard --logdir /logs
```

|Epoch|	Training Loss|	Validation Loss|	Accuracy|	Precision|	Recall	|F1|
|---|---|---|---|---|---|---|
|1	|0.288700|	0.182780|	0.940674	|0.940485|	0.940674|	0.940567|
|2	|0.162000|	0.177315|	0.946552|	0.946379|	0.946552|	0.946417|
|3	|0.118500|	0.190550|	0.948981	|0.948953|	0.948981|	0.948944|
|4	|0.080700|	0.242351|	0.947806|	0.947806|	0.947806|	0.947774|
|5	|0.052700|	0.284349|	0.948433|	0.948430|	0.948433|	0.948430|




# METRIC FOR EVALUATION

||baseline|from scratch|8bits bnb|4bits bnb|
|---|---|---|---|---|
|Accuracy|0.9441|0.9447|0.9438|0.9446|
|Precision|0.9439|0.9446|0.9436|0.9444|
|Recall|0.9441|0.9447|0.9438|0.9446|
|F1-score|0.9440|0.9446|0.9437|0.9445|
|memory footprint|474.71 MB|119.025|156.36|115.86|
|inference latency(in ms)|1.01|1.0667|3.18|2.12|

> for confusion matrix checkout the .ipynb files in /

```text?code_stdout&code_event_index=2
Memory Reduction 4bits vs Baseline (%): 75.59%
Memory Reduction 4bits vs Baseline (Factor): 4.10x
Latency Increase 8bits vs Baseline (Factor): 3.15x
Latency Increase 4bits vs Baseline (Factor): 2.10x

Full Data Frame:
                        baseline  from scratch  8bits bnb  4bits bnb
Accuracy                  0.9441        0.9447     0.9438     0.9446
Precision                 0.9439        0.9446     0.9436     0.9444
Recall                    0.9441        0.9447     0.9438     0.9446
F1-score                  0.9440        0.9446     0.9437     0.9445
memory footprint (MB)   474.7100      119.0250   156.3600   115.8600
inference latency (ms)    1.0100        1.0667     3.1800     2.1200

```

The table presents a comprehensive evaluation of a machine learning model under four different conditions, primarily comparing a standard implementation (baseline) with a highly optimized one (**from scratch**) and two forms of model quantization using the **Bitsandbytes (bnb)** library.

The analysis highlights a crucial trade-off between **model performance (quality)** and **resource efficiency (memory and speed)**.

-----

## 1\. Predictive Performance Analysis

The first four metrics—**Accuracy, Precision, Recall, and F1-score**—measure the model's predictive quality.

  * **Negligible Performance Difference:** Across all four methods, the performance metrics are extremely close, clustering around **0.944**.
      * The **'from scratch'** and **'4bits bnb'** methods actually show a marginal *improvement* over the **'baseline'** (e.g., Accuracy of $0.9447$ and $0.9446$ vs. $0.9441$).
      * The **'8bits bnb'** model has the lowest scores (e.g., Accuracy of $0.9438$), but the drop is minimal, suggesting that even with aggressive memory optimization, the model's ability to make correct predictions is **well-preserved**.

**Key Takeaway:** The process of quantization (8-bit and 4-bit) has had **no significant negative impact** on the model's overall predictive performance.

-----

## 2\. Resource Efficiency Analysis

The final two metrics—**memory footprint** and **inference latency**—measure the model's operational efficiency.

### Memory Footprint (MB) 💾

This metric shows the most dramatic and intended effects of the optimization techniques.

  * **Baseline:** The original model consumes **$474.71 \text{ MB}$**.
  * **Quantization Success:** Both the 'from scratch' and quantized models achieve a massive reduction in memory consumption.
      * The **'4bits bnb'** model is the most memory efficient at **$115.86 \text{ MB}$**. This represents a $\mathbf{75.59\%}$ reduction, making the model about **$4.1$ times smaller** than the baseline.
      * The **'from scratch'** model is also highly efficient at **$119.025 \text{ MB}$**.
      * The **'8bits bnb'** model, as expected, is larger than the 4-bit and 'from scratch' models at **$156.36 \text{ MB}$**.

**Key Takeaway:** Model quantization, especially **4-bit quantization**, is highly effective for deployment on memory-constrained devices, such as smaller GPUs or edge devices, significantly shrinking the model size with virtually no performance cost.

### Inference Latency (ms) ⏳

This metric measures the time taken to generate a single prediction.

  * **Baseline & 'From Scratch':** The non-quantized models are the fastest. The **'baseline'** is the fastest at **$1.01 \text{ ms}$**, and **'from scratch'** is almost identical at $1.0667 \text{ ms}$.
  * **Quantization Overhead:** The quantized models are significantly slower than the baseline, indicating an overhead in the process of running quantized models (which often requires on-the-fly de-quantization or specialized kernels).
      * The **'8bits bnb'** model is the slowest at **$3.18 \text{ ms}$**, making it $\mathbf{3.15}$ times slower than the baseline.
      * The **'4bits bnb'** model is faster than the 8-bit version but still much slower than the baseline at **$2.12 \text{ ms}$** ($\mathbf{2.1}$ times slower).

**Key Takeaway:** While quantization saves memory, it introduces a substantial penalty in inference speed. There is an inverse relationship: the more memory saved, the slower the inference generally becomes, though the 4-bit quantization appears to be better optimized for speed than the 8-bit in this specific test.

-----

## 3\. Summary of Trade-offs and Best Use Cases

| Method | Predictive Performance | Memory Footprint | Inference Latency | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | High | Very Large | Fastest | Applications where **speed is paramount** and high-end hardware is available. |
| **From Scratch** | Highest | Small | Very Fast | Excellent general-purpose choice; high performance with **strong memory efficiency** and negligible speed loss. |
| **8bits bnb** | High (Slightly lowest) | Medium | Slowest | Offers memory savings, but its latency is the greatest disadvantage. |
| **4bits bnb** | Highest | **Smallest** | Slow | Ideal for **extreme memory constraints** (e.g., fine-tuning on consumer GPUs) where a **$2\text{x}$ slowdown in latency is acceptable** for a $\mathbf{4\text{x}}$ memory reduction. |