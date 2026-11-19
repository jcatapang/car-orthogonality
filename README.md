# Orthogonality Is Not Disentanglement — code and data

Everything needed to reproduce every number, table and figure in the paper.

## Repository layout
```
car_audit.py           co-activation audit (orthant law)
car_audit2.py          conditional entanglement, Bias in Bios projection audit, LEACE and Σ⁻¹ conditions
car_steer.py           steering with readouts at later layers (equal-norm controls)
car_token_steer.py     next-token test
car_bbq2.py            BBQ behavioral test;  car_bbq_diag.py  its diagnostic
analysis/              scripts that turn results into the paper's macros, tables and figures
results/               every result file used in the paper
CAR_scale_runs.ipynb   Colab notebook for the 7B, 14B and Llama-3.1-8B runs (one 80 GB A100)
```
The analysis commands below name result files without a directory; run them from `results/` (or prefix
`results/`) and pass an output directory for the generated `numbers_*.tex`, `tables/` and figures.

## Setup
```bash
pip install -r requirements.txt
```
A CUDA GPU is recommended (an 8 GB laptop GPU suffices for all models up to 3B in bfloat16).

## 1. Original co-activation audit (orthant law; first-generation tables)
```bash
python car_audit.py --models roberta-base gpt2 EleutherAI/pythia-160m --layers all --out car_audit_results.json
python car_audit.py --models roberta-large bert-base-uncased answerdotai/ModernBERT-base gpt2-medium \
    EleutherAI/pythia-410m Qwen/Qwen2.5-1.5B --layers stride3 --seeds 3 --out car_audit_results_full.json
```
`python car_audit.py --synthetic` runs the pipeline on synthetic activations in seconds (no GPU, no download).

## 2. Conditional entanglement, Bias in Bios audit, LEACE and Σ⁻¹ filters
```bash
python car_audit2.py --models roberta-base gpt2 EleutherAI/pythia-160m --datasets stereoset factorial \
    --template-cells --layers stride2 --seeds 3 --out car_audit2_G_core.json
python car_audit2.py --models roberta-large bert-base-uncased answerdotai/ModernBERT-base gpt2-medium \
    EleutherAI/pythia-410m --datasets stereoset factorial --template-cells --layers stride3 --seeds 3 --out car_audit2_G_small.json
python car_audit2.py --models Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct --half --datasets stereoset factorial \
    --template-cells --layers stride4 --seeds 3 --out car_audit2_G_qwen.json
python car_audit2.py --models roberta-base gpt2 EleutherAI/pythia-160m bert-base-uncased answerdotai/ModernBERT-base \
    roberta-large gpt2-medium EleutherAI/pythia-410m --datasets bios --layers stride3 --seeds 3 --out car_audit2_G_bios_small.json
python car_audit2.py --models Qwen/Qwen2.5-1.5B --half --datasets bios --layers stride4 --seeds 3 --out car_audit2_G_bios_qwen.json
python car_audit2.py --models Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct --half --bs 8 --datasets bios --layers stride4 \
    --seeds 3 --out car_audit2_G_bios_qwen3b.json
```
The earlier result files in `results/` (`car_audit2_results*.json`, `_C_`, `_D_`, `_factorial_tpl*`) come from the
same script with the dataset and `--template-cells` options named in their file names.
Conditions per probe kind (logistic regression and difference of means): original, Euclidean, Σ-, within-cell
Σ_W- and Ledoit–Wolf Gram–Schmidt, LEACE applied to the readout, the Σ⁻¹ filter, shuffled-label probes and random
Σ-orthonormal directions. `--bs` lowers the extraction batch size for 3B models on long texts.

### 7B and 14B models (one 80 GB A100; `CAR_scale_runs.ipynb` runs all of these on Colab)
```bash
for M in 7B 14B; do
  python car_audit2.py --models Qwen/Qwen2.5-$M --half --bs 32 --datasets bios --layers frac6 --seeds 3 --out car_audit2_G_bios_${M,,}.json
  python car_audit2.py --models Qwen/Qwen2.5-$M --half --bs 32 --datasets stereoset factorial --template-cells --layers frac6 --seeds 3 --out car_audit2_G_${M,,}.json
done
python car_steer.py --models Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B --half --out car_steer_v3_big.json
python car_audit2.py --models meta-llama/Llama-3.1-8B --half --bs 32 --datasets bios --layers frac6 --seeds 3 --out car_audit2_G_bios_llama8b.json
```
`--layers frac6` takes six evenly spaced hidden states from the first to the last.

## 3. Steering, next-token and BBQ tests
```bash
python car_steer.py --models gpt2 gpt2-medium EleutherAI/pythia-160m EleutherAI/pythia-410m Qwen/Qwen2.5-1.5B --out car_steer_v3.json
python car_steer.py --models Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct --half --out car_steer_v3_3b.json
python car_token_steer.py --models gpt2 gpt2-medium EleutherAI/pythia-160m EleutherAI/pythia-410m Qwen/Qwen2.5-1.5B --out car_token_steer.json
python car_token_steer.py --models Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct --half --out car_token_steer_3b.json
python car_bbq2.py --models Qwen/Qwen2.5-1.5B-Instruct --half --out car_bbq2_results.json
python car_bbq2.py --models Qwen/Qwen2.5-3B-Instruct --half --bs 4 --out car_bbq2_results_3b.json
python car_bbq_diag.py --model Qwen/Qwen2.5-1.5B-Instruct --half --n-amb 200 --out car_bbq_diag.json
```
Steering vectors: the gender probe, its Euclidean and Σ-orthogonalizations, the pattern Σw^Σ, and Euclidean and
random vectors rescaled to the norm of Σw^Σ (equal-norm controls). `--tiny-test` checks a pipeline on a random model.

## 4. Paper numbers, tables and figures
```bash
python analysis/gen_numbers.py paper/ car_audit_results.json car_audit_results_full.json car_audit_results_phase3.json
python analysis/make_figs.py paper/
python analysis/gen_cond.py paper/ car_audit2_results.json car_audit2_results_qwen3b.json \
    --both car_audit2_C_*.json car_audit2_D_*.json car_audit2_G_7b.json car_audit2_G_14b.json \\
    car_audit2_G_bios_7b.json car_audit2_G_bios_14b.json car_audit2_G_bios_llama8b.json --bios car_audit2_G_bios_*.json
python analysis/gen_e.py paper/ car_audit2_G_core.json car_audit2_G_small.json car_audit2_G_qwen.json car_audit2_G_7b.json car_audit2_G_14b.json
python analysis/gen_steer.py paper/ car_steer_v3.json car_steer_v3_3b.json car_steer_v3_big.json
python analysis/gen_token.py paper/ car_token_steer.json car_token_steer_3b.json
python analysis/gen_bbq.py paper/ car_bbq2_results.json car_bbq2_results_3b.json
python analysis/make_hero.py paper/ paper/case_bios.json
```
Each script writes LaTeX macros (`numbers_*.tex`) and `tables/*.tex`; the manuscript contains no hand-typed results.
All result files used in the paper are in `results/`.

## Factorial template set
Defined in `car_audit.py` (`FACT_TEMPLATES`, `FACT_GENDER`, `FACT_RACE`, `FACT_PROF`): 24 templates fully crossing
gender × race × profession, so the three labels are independent by construction.
