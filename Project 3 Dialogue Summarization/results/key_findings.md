### Test-set results (819 dialogues)

| method | rouge1 | rouge2 | rougeL | rougeLsum | avg_words | ms_per_dialogue |
|---|---|---|---|---|---|---|
| lead3_baseline | 31.34 | 8.68 | 24.21 | 29.67 | 23.60 | 0.00 |
| greedy | 36.24 | 12.99 | 29.39 | 32.90 | 16.20 | 83.00 |
| beam4 | 37.49 | 14.09 | 30.42 | 34.02 | 15.30 | 136.10 |
| beam4_lp2.0 | 37.71 | 13.71 | 30.00 | 34.09 | 17.70 | 135.90 |

### Key findings

- Decoding strategy selected on validation: **beam4**. Test ROUGE-1 / ROUGE-2 / ROUGE-L = **37.49 / 14.09 / 30.42**.
- Versus the Lead-3 baseline (ROUGE-L 24.21): +6.21 ROUGE-L (95% bootstrap CI of the paired difference: +5.08 to +7.23).
- Generated summaries average 15.3 words; reference summaries average 20.0.
- By dialogue length, mean ROUGE-L is 40.7 on the shortest quartile and 23.7 on the longest.
- Two-speaker chats score 30.8 ROUGE-L; chats with 3+ speakers score 29.0.
- 49.5% of summaries contain a capitalized name not found in the dialogue (heuristic); speaker recall is 55.9%.
- Training took 44 min on Tesla T4 (14,731 training dialogues); the best checkpoint was at step 4500 (validation ROUGE-L 28.99).

### Figures

![Training curves](figures/training_curves.png)

![Test ROUGE](figures/test_rouge.png)

![Error analysis](figures/error_breakdown.png)

Example summaries: [results/sample_summaries.md](results/sample_summaries.md)
