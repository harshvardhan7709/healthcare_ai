# Diabetic Retinopathy Screening Pipeline

This workspace contains a deployable training scaffold for diabetic-retinopathy grading from fundus images and CSV labels.

Expected data layout:

```text
train.csv                 # id_code,diagnosis
train_images/<id_code>.png
test.csv                  # id_code
test_images/<id_code>.png
```

Diagnosis labels follow the **International Clinical Diabetic Retinopathy (ICDR) Disease Severity Scale**:

```text
Level 0: No apparent retinopathy (Non-referable; annual rescreening)
Level 1: Mild non-proliferative DR (Microaneurysms only; Non-referable; rescreen in 6–12 months)
Level 2: Moderate non-proliferative DR (More than MAs, less than severe NPDR; Referable DR; 1–3 month referral)
Level 3: Severe non-proliferative DR (4-2-1 criteria without PDR signs; Referable DR; prompt 2–4 week referral)
Level 4: Proliferative DR (Neovascularization / vitreous hemorrhage; Referable DR; urgent 24–48 hour referral)
```

## What The Pipeline Handles

- Image quality assessment for focus, illumination, contrast, and field-of-view coverage.
- Adaptive preprocessing for gradeable/borderline images:
  - fundus field cropping
  - illumination normalization
  - CLAHE contrast enhancement
  - light denoising for borderline captures
- Rejection of ungradeable images with recapture feedback.
- International Clinical DR severity grading on the 0–4 scale with hallmark clinical lesion criteria.
- Referable DR screening for Level 2+ with validation threshold calibration targeting clinically acceptable sensitivity (>90%) and specificity (>85%).
- Detailed screening reports with severity level, ICDR label, hallmark findings, referable risk, calibrated threshold, referral urgency timeline, and recommended clinical actions.

## Run A Quality Audit

```bash
python scripts/quality_report.py --csv train.csv --image-dir train_images --output outputs/quality_report.csv
```

The report includes `quality_status`, `quality_score`, metric columns, rejection reasons, and recapture feedback.

## Train Now With Installed Dependencies

The current environment has OpenCV and scikit-learn available, so this baseline can train without installing PyTorch:

```bash
python scripts/train_sklearn.py --csv train.csv --image-dir train_images --output outputs/sklearn_dr_model.joblib
```

For a quick smoke test:

```bash
python scripts/train_sklearn.py --limit 64 --n-estimators 40
```

The baseline defaults to `--n-jobs 1` so it also runs in restricted Windows environments. On an unrestricted workstation, add `--n-jobs -1` to use all cores.

## Predict / Screen New Images

```bash
python scripts/predict_sklearn.py --model outputs/sklearn_dr_model.joblib --csv test.csv --image-dir test_images --output outputs/screening_predictions.csv
```

To force predictions even for rejected images when creating a competition-style submission:

```bash
python scripts/predict_sklearn.py --model outputs/sklearn_dr_model.joblib --csv test.csv --image-dir test_images --output outputs/screening_predictions.csv --submission-output outputs/submission.csv --predict-rejected
```

## Streamlit Verification App

Run the upload-and-verify interface:

```bash
streamlit run streamlit_app.py
```

The app checks the uploaded fundus image, shows the enhanced preview, and returns a DR grade when the saved model bundle is available.

If structure overlays are enabled in the sidebar, the same app also shows heuristic localization and segmentation for optic disc, fovea, vessels, microaneurysms, exudates, hemorrhage burden, and neovascularization suspicion.

The severity model reports the International Clinical DR scale levels 0-4 and a separate referable-DR decision for Level 2+.

## Train A CNN With PyTorch

For CPU training, install the optional dependencies from `requirements.txt`. For
NVIDIA GPU training, install the CUDA wheels from `requirements-gpu.txt`:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-gpu.txt
```

The script automatically uses CUDA when it is visible. To require NVIDIA GPU
training and GPU validation explicitly, run:

```bash
python scripts/train_torch.py --device cuda --csv train.csv --image-dir train_images --model simple_cnn --epochs 10
```

With torchvision installed, you can use transfer learning:

```bash
python scripts/train_torch.py --device cuda --model resnet18 --pretrained --epochs 20 --batch-size 16
```

To force NVIDIA GPU training when CUDA is available:

```bash
python scripts/train_torch.py --device cuda --multi-gpu --num-workers 2
```

The CNN script caches preprocessed images in `outputs/processed_train_images`,
places training batches and holdout validation batches on the selected device,
and saves the best checkpoint by quadratic weighted kappa. Check availability
before training with `python -c "import torch; print(torch.cuda.is_available())"`.

## Clinical Benchmark On Referable DR (Level 2+)

The pipeline calibrates the referable DR threshold on holdout validation data to enforce clinically acceptable screening standards:
- **Target Clinical Benchmarks**: Sensitivity > 90% (to prevent missed vision-threatening disease), Specificity > 85% (to prevent excessive false referrals).
- **Validation Results Achieved**:
  - **Sensitivity**: **95.74%** (45/47 referable cases captured, only 2 missed)
  - **Specificity**: **90.14%** (64/71 non-referable cases correctly ruled out)
  - **ROC AUC**: **0.9655**
  - **Quadratic Weighted Kappa (QWK)**: **0.7559**
  - **Calibrated Cutoff**: `referable_threshold = 0.438`

## Clinical Deployment Note

This code is a training scaffold and quality-control prototype, not a cleared medical device. Before clinical use, validate image-quality thresholds and model performance on local cameras, operators, demographics, disease prevalence, and referral workflows.
