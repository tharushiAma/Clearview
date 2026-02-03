# run_ablations.ps1
# Runs 2 key ablations:
#  A) No synthetic data (keep sampler ON)
#  B) No sampler (keep synthetic ON)
# For each: trains baseline (msr_strength=0.0) and MSR model (msr_strength=0.3)
# Then evaluates and writes outputs/eval/ablation_summary.csv

$ErrorActionPreference = "Stop"

# -------------------------
# Config
# -------------------------
$SEED = 42
$BATCH = 8
$EPOCHS = 2
$LR = "2e-5"
$DROPOUT = 0.3
$TEXT_COL = "text_clean"

$TRAIN_ORIG = "data/splits/train.parquet"
$TRAIN_AUG = "data/splits/train_aug.parquet"
$VAL_PATH = "data/splits/val.parquet"

$TRAIN_PY = "src/models/train_roberta_improved.py"
$EVAL_PY = "src/evaluation/evaluate_and_log.py"

# output dirs for training
$OUT_A1 = "outputs/abl_no_synth_baseline"
$OUT_A2 = "outputs/abl_no_synth_msr"
$OUT_B1 = "outputs/abl_no_sampler_baseline"
$OUT_B2 = "outputs/abl_no_sampler_msr"

# eval dirs
$EVAL_A1 = "outputs/eval/abl_no_synth_baseline"
$EVAL_A2 = "outputs/eval/abl_no_synth_msr"
$EVAL_B1 = "outputs/eval/abl_no_sampler_baseline"
$EVAL_B2 = "outputs/eval/abl_no_sampler_msr"

New-Item -ItemType Directory -Force -Path "outputs/eval" | Out-Null

function Run-Train {
    param(
        [string]$Name,
        [string]$TrainArgs
    )
    Write-Host "`n===================="
    Write-Host "TRAIN: $Name"
    Write-Host "===================="
    $flatArgs = $TrainArgs -replace "`r?`n", " "
    $cmd = "python $TRAIN_PY $flatArgs"
    Write-Host $cmd
    iex $cmd
}

function Run-Eval {
    param(
        [string]$Name,
        [string]$Ckpt,
        [string]$OutDir,
        [string]$MsrStrength
    )
    Write-Host "`n--------------------"
    Write-Host "EVAL: $Name"
    Write-Host "--------------------"
    $cmd = "python $EVAL_PY --val_path $VAL_PATH --text_col $TEXT_COL --ckpt $Ckpt --out_dir $OutDir --msr_strength $MsrStrength --save_predictions"
    Write-Host $cmd
    iex $cmd
}

function Read-Metrics {
    param(
        [string]$MetricsJson
    )
    if (!(Test-Path $MetricsJson)) {
        throw "Missing metrics file: $MetricsJson"
    }
    return Get-Content $MetricsJson -Raw | ConvertFrom-Json
}

# -------------------------
# ABLATION A: NO SYNTHETIC (Sampler ON)
# -------------------------
Run-Train "A1 NoSynth + Baseline (MSR OFF)" "
  --train_path $TRAIN_ORIG
  --val_path $VAL_PATH
  --text_col $TEXT_COL
  --batch_size $BATCH
  --epochs $EPOCHS
  --lr $LR
  --dropout $DROPOUT
  --msr_strength 0.0
  --conflict_weight 0.5
  --null_weight 0.2
  --use_sampler
  --patience 1
  --out_dir $OUT_A1
  --seed $SEED
"

Run-Train "A2 NoSynth + MSR (MSR ON)" "
  --train_path $TRAIN_ORIG
  --val_path $VAL_PATH
  --text_col $TEXT_COL
  --batch_size $BATCH
  --epochs $EPOCHS
  --lr $LR
  --dropout $DROPOUT
  --msr_strength 0.3
  --conflict_weight 0.5
  --null_weight 0.2
  --use_sampler
  --patience 1
  --out_dir $OUT_A2
  --seed $SEED
"

# -------------------------
# ABLATION B: NO SAMPLER (Synthetic ON)
# -------------------------
Run-Train "B1 Synth + Baseline (NoSampler, MSR OFF)" "
  --train_aug_path $TRAIN_AUG
  --val_path $VAL_PATH
  --use_synthetic
  --text_col $TEXT_COL
  --batch_size $BATCH
  --epochs $EPOCHS
  --lr $LR
  --dropout $DROPOUT
  --msr_strength 0.0
  --conflict_weight 0.5
  --null_weight 0.2
  --patience 1
  --out_dir $OUT_B1
  --seed $SEED
"

Run-Train "B2 Synth + MSR (NoSampler, MSR ON)" "
  --train_aug_path $TRAIN_AUG
  --val_path $VAL_PATH
  --use_synthetic
  --text_col $TEXT_COL
  --batch_size $BATCH
  --epochs $EPOCHS
  --lr $LR
  --dropout $DROPOUT
  --msr_strength 0.3
  --conflict_weight 0.5
  --null_weight 0.2
  --patience 1
  --out_dir $OUT_B2
  --seed $SEED
"

# -------------------------
# EVALUATE ALL 4
# -------------------------
Run-Eval "A1 NoSynth Baseline" "$OUT_A1/best_model.pt" $EVAL_A1 "0.0"
Run-Eval "A2 NoSynth MSR"      "$OUT_A2/best_model.pt" $EVAL_A2 "0.3"
Run-Eval "B1 NoSampler Base"   "$OUT_B1/best_model.pt" $EVAL_B1 "0.0"
Run-Eval "B2 NoSampler MSR"    "$OUT_B2/best_model.pt" $EVAL_B2 "0.3"

# -------------------------
# BUILD SUMMARY CSV
# -------------------------
$rows = @()

function Add-Row {
    param([string]$Tag, [string]$EvalDir)
    $path = "$EvalDir/report.json"
    if (!(Test-Path $path)) {
        Write-Host "Warning: Missing report at $path"
        return $null
    }
    $m = Get-Content $path -Raw | ConvertFrom-Json

    # Map from nested JSON structure
    $row = [PSCustomObject]@{
        tag                         = $Tag
        absa_overall_macro_f1_after = $m.absa.overall_macro_f1
        conflict_macro_f1           = $m.conflict.conf_f1_macro
        mixed_f1                    = $m.conflict.mixed_f1
        separation                  = $m.conflict.separation
        msr_total_error_reduction   = $m.msr_error_reduction.total_reduction
        stayingpower_f1_macro       = $m.absa.per_aspect.stayingpower.f1_macro
        texture_f1_macro            = $m.absa.per_aspect.texture.f1_macro
        smell_f1_macro              = $m.absa.per_aspect.smell.f1_macro
        price_f1_macro              = $m.absa.per_aspect.price.f1_macro
        colour_f1_macro             = $m.absa.per_aspect.colour.f1_macro
        shipping_f1_macro           = $m.absa.per_aspect.shipping.f1_macro
        packing_f1_macro            = $m.absa.per_aspect.packing.f1_macro
    }
    return $row
}

$rows += Add-Row "A1_no_synth_baseline" $EVAL_A1
$rows += Add-Row "A2_no_synth_msr"      $EVAL_A2
$rows += Add-Row "B1_no_sampler_base"   $EVAL_B1
$rows += Add-Row "B2_no_sampler_msr"    $EVAL_B2

$summaryPath = "outputs/eval/ablation_summary.csv"
$rows | Export-Csv -NoTypeInformation $summaryPath
Write-Host "`n✅ Saved ablation summary to: $summaryPath"

# Also print it
$rows | Format-Table -AutoSize
