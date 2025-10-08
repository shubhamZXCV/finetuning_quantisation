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

> for confusion matrix checkout the .ipynb files in /src