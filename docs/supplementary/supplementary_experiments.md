# S4 Temperature-Control Experiment

We sampled 100 of the 500 tasks stratified by source (33 HumanEval, 33 APPS introductory, and 34 APPS interview tasks). Each of the six models generated five outputs per task at T=0 and T=0.5 with the prompts in Section S1, and these outputs were compared with the existing T=0.8 outputs for the same tasks. Each model contributes 500 samples per temperature, giving 3,000 samples per temperature and 9,000 in total.

A sample is hallucinated if it carries any label. Per-category rates use all outputs as the denominator, and the logic-category rate counts outputs with at least one of GLM, CI, LoLM, or II. Temperature differences are paired differences over tasks after averaging the five outputs within each task, with the six models weighted equally. Confidence intervals come from a task-cluster bootstrap stratified by source (5,000 resamples) that keeps all models, temperatures, and generations of each sampled task. Tables S10 and S11 report the results.

**Table S10. Overall Hallucination Prevalence at Three Temperatures**

| **Model**           | **T=0, $n$ (%)** | **T=0.5, $n$ (%)** | **T=0.8, $n$ (%)** | **$\Delta(0.8-0)$, pp** |    **95% CI**    |
|:--------------------|:----------------:|:------------------:|:------------------:|:---------------------------------:|:----------------:|
| CodeLlama-7B-I      |    345 (69.0)    |     343 (68.6)     |     324 (64.8)     |              $-4.20$              | $[-11.20, 3.00]$ |
| DSC-V2-Lite-I-16B   |    222 (44.4)    |     218 (43.6)     |     253 (50.6)     |               6.20                | $[-2.60, 15.00]$ |
| Qwen2.5-Coder-32B-I |    208 (41.6)    |     212 (42.4)     |     248 (49.6)     |               8.00                | $[-0.40, 16.40]$ |
| GPT-4o              |    160 (32.0)    |     159 (31.8)     |     199 (39.8)     |               7.80                | $[1.40, 14.20]$  |
| DeepSeek-V2.5-1210  |    135 (27.0)    |     138 (27.6)     |     165 (33.0)     |               6.00                | $[-2.40, 14.40]$ |
| Claude-3.5-Sonnet   |    140 (28.0)    |     150 (30.0)     |     208 (41.6)     |               13.60               | $[6.20, 21.20]$  |
| All six models      |  1,210 (40.33)   |   1,220 (40.67)    |   1,397 (46.57)    |               6.23                | $[2.37, 10.17]$  |

***Note:*** pp = percentage points. All five hallucination judgments agree in 85.67%, 75.50%, and 72.17% of task-model pairs at T=0, 0.5, and 0.8, respectively.

**Table S11. Subcategory Prevalence at Three Temperatures ($n$ = 3,000 per temperature)**

| **Subcategory** |   **T=0**    |  **T=0.5**   |  **T=0.8**   |
|:----------------|:------------:|:------------:|:------------:|
| DTM             |  27 (0.90%)  |  25 (0.83%)  |  35 (1.17%)  |
| DSM             |  45 (1.50%)  |  51 (1.70%)  |  54 (1.80%)  |
| GLM             | 618 (20.60%) | 593 (19.77%) | 741 (24.70%) |
| CI              |  45 (1.50%)  |  53 (1.77%)  |  82 (2.73%)  |
| LoLM            | 428 (14.27%) | 442 (14.73%) | 426 (14.20%) |
| II              | 189 (6.30%)  | 196 (6.53%)  |  62 (2.07%)  |
| CD              |  0 (0.00%)   |  1 (0.03%)   | 141 (4.70%)  |
| DC              |  17 (0.57%)  |  10 (0.33%)  |  50 (1.67%)  |
| AKE             |  19 (0.63%)  |  10 (0.33%)  |  43 (1.43%)  |
| RSI             |  22 (0.73%)  |  14 (0.47%)  | 125 (4.17%)  |
| RO              | 162 (5.40%)  | 165 (5.50%)  |  59 (1.97%)  |
| SE              |  17 (0.57%)  |  24 (0.80%)  |  84 (2.80%)  |
| NCC             |  31 (1.03%)  |  28 (0.93%)  |  15 (0.50%)  |

***Note:*** Labels overlap. Outputs with any logic label number 1,083, 1,089, and 1,243 (89.50%, 89.26%, and 88.98% of hallucinated outputs) at T=0, 0.5, and 0.8, respectively.

# S5 Type Classification: Evaluation Rules and Per-category Results

Each hallucinated test sample receives one predicted subcategory, and seven-category predictions are obtained by mapping it to its top-level category. A prediction is correct if it belongs to the sample’s label set. For per-category metrics, each sample has one reference category: the predicted category if the prediction is correct, and otherwise the lowest-numbered category in its label set. NONE or unparsable outputs count as errors and as predictions of no category. Precision, recall, and F1 are computed one-vs-rest against the reference categories; macro-F1 averages F1 over categories, and weighted metrics weight each category by its number of reference samples, so that accuracy equals weighted recall. Tables S12 and S13 give the per-category results.

**Table S12. Per-category Precision, Recall, and F1 for the Seven Top-level Categories**

| Category | DeepSeek-V3 P | DeepSeek-V3 R | DeepSeek-V3 F1 | Qwen3 P | Qwen3 R | Qwen3 F1 | GPT-4o P | GPT-4o R | GPT-4o F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DSI | 0.323 | 0.588 | 0.417 | 0.139 | 0.647 | 0.229 | 0.072 | 0.647 | 0.129 |
| LI | 0.915 | 0.923 | 0.919 | 0.860 | 0.701 | 0.772 | 0.926 | 0.638 | 0.755 |
| RDC | 0.700 | 0.560 | 0.622 | 0.600 | 0.480 | 0.533 | 0.588 | 0.400 | 0.476 |
| EKI | 0.895 | 0.971 | 0.932 | 0.927 | 0.729 | 0.816 | 0.963 | 0.371 | 0.536 |
| RSI | 0.000 | 0.000 | 0.000 | 0.083 | 0.143 | 0.105 | 0.000 | 0.000 | 0.000 |
| RMI | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SSI | 0.900 | 0.375 | 0.529 | 0.688 | 0.458 | 0.550 | 0.625 | 0.417 | 0.500 |
| Macro-F1 |  |  | 0.488 |  |  | 0.429 |  |  | 0.342 |

| Category | Claude-3.5 P | Claude-3.5 R | Claude-3.5 F1 | Llama-3.1 P | Llama-3.1 R | Llama-3.1 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DSI | 0.071 | 0.647 | 0.129 | 0.049 | 0.529 | 0.090 |
| LI | 0.909 | 0.570 | 0.701 | 0.802 | 0.473 | 0.595 |
| RDC | 0.647 | 0.440 | 0.524 | 0.389 | 0.280 | 0.326 |
| EKI | 0.949 | 0.529 | 0.679 | 0.784 | 0.414 | 0.542 |
| RSI | 0.111 | 0.143 | 0.125 | 0.000 | 0.000 | 0.000 |
| RMI | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SSI | 0.733 | 0.458 | 0.564 | 0.444 | 0.333 | 0.381 |
| Macro-F1 |  |  | 0.389 |  |  | 0.276 |

**Table S13. Per-category Precision, Recall, and F1 for the Thirteen Subcategories**

| **Subcategory** | **Metric** | **DeepSeek-V3** | **Qwen3** | **GPT-4o** | **Claude-3.5** | **Llama-3.1** |
|:----------------|:-----------|:---------------:|:---------:|:----------:|:--------------:|:-------------:|
| DTM | P | 0.300 | 0.091 | 0.048 | 0.058 | 0.038 |
|                 | R          |      0.429      |   0.429   |   0.429    |     0.429      |     0.286     |
|                 | F1         |      0.353      |   0.150   |   0.086    |     0.102      |     0.068     |
| DSM | P | 0.333 | 0.128 | 0.070 | 0.072 | 0.049 |
|                 | R          |      0.500      |   0.500   |   0.500    |     0.500      |     0.400     |
|                 | F1         |      0.400      |   0.204   |   0.123    |     0.127      |     0.087     |
| GLM | P | 0.709 | 0.571 | 0.644 | 0.690 | 0.533 |
|                 | R          |      0.732      |   0.442   |   0.342    |     0.411      |     0.253     |
|                 | F1         |      0.720      |   0.499   |   0.447    |     0.515      |     0.343     |
| CI | P | 0.538 | 0.323 | 0.375 | 0.435 | 0.240 |
|                 | R          |      0.467      |   0.333   |   0.300    |     0.333      |     0.200     |
|                 | F1         |      0.500      |   0.328   |   0.333    |     0.377      |     0.218     |
| LoLM | P | 0.612 | 0.415 | 0.491 | 0.540 | 0.370 |
|                 | R          |      0.571      |   0.371   |   0.267    |     0.324      |     0.190     |
|                 | F1         |      0.591      |   0.392   |   0.346    |     0.405      |     0.252     |
| II | P | 0.583 | 0.370 | 0.421 | 0.474 | 0.286 |
|                 | R          |      0.538      |   0.385   |   0.308    |     0.346      |     0.231     |
|                 | F1         |      0.560      |   0.377   |   0.356    |     0.400      |     0.255     |
| CD | P | 0.625 | 0.412 | 0.462 | 0.538 | 0.286 |
|                 | R          |      0.526      |   0.368   |   0.316    |     0.368      |     0.211     |
|                 | F1         |      0.571      |   0.389   |   0.375    |     0.438      |     0.242     |
| DC | P | 0.400 | 0.200 | 0.200 | 0.250 | 0.100 |
|                 | R          |      0.333      |   0.167   |   0.167    |     0.167      |     0.167     |
|                 | F1         |      0.364      |   0.182   |   0.182    |     0.200      |     0.125     |
| AKE | P | 0.838 | 0.791 | 0.870 | 0.879 | 0.667 |
|                 | R          |      0.886      |   0.486   |   0.286    |     0.414      |     0.257     |
|                 | F1         |      0.861      |   0.602   |   0.430    |     0.563      |     0.371     |
| RSI | P | 0.000 | 0.067 | 0.000 | 0.100 | 0.000 |
|                 | R          |      0.000      |   0.143   |   0.000    |     0.143      |     0.000     |
|                 | F1         |      0.000      |   0.091   |   0.000    |     0.118      |     0.000     |
| RO | P | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
|                 | R          |      0.000      |   0.000   |   0.000    |     0.000      |     0.000     |
|                 | F1         |      0.000      |   0.000   |   0.000    |     0.000      |     0.000     |
| SE | P | 0.778 | 0.583 | 0.500 | 0.636 | 0.385 |
|                 | R          |      0.368      |   0.368   |   0.316    |     0.368      |     0.263     |
|                 | F1         |      0.500      |   0.452   |   0.387    |     0.467      |     0.312     |
| NCC | P | 0.500 | 0.333 | 0.333 | 0.333 | 0.200 |
|                 | R          |      0.400      |   0.200   |   0.200    |     0.200      |     0.200     |
|                 | F1         |      0.444      |   0.250   |   0.250    |     0.250      |     0.200     |
| Macro-F1        |            |      0.451      |   0.301   |   0.255    |     0.305      |     0.190     |

***Note:*** P = Precision; R = Recall.

