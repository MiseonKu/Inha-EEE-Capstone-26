# Figures and Tables

| Paper artifact | Repo file | Source command | Source data | Expected metric |
|---|---|---|---|---|
| 1-link contact sheet | `ml/final_results/figures/1link_paper_figure_contact_sheet.png` | `make figures` | 1-link final package | 584 clean rows |
| 2-link contact sheet | `ml/final_results/figures/2link_paper_figure_contact_sheet.png` | `make figures` | 2-link final package | 5,000 clean rows |
| 2-link alignment residual | `ml/final_results/figures/2link_alignment_residual.png` | `make figures` | 2-link manifest + labels | Kabsch RMSE 51.4663 mm |
| 2-link model comparison | `ml/final_results/figures/2link_model_comparison_summary.png` | `make figures` | 2-link summary table | mean 3D error 0.7101 mm |
| Main result table | `ml/final_results/tables/main_results.csv` | `make tables` | `ml/final_results/expected/expected_metrics.json` | matches README table |
