# run_final_ablations_4class.ps1
# 2x2 Matrix: [No/Yes Sampler] x [No/Yes Synthetic]
# For each corner: Baseline (0.0) vs MSR (0.3)

$ErrorActionPreference = "Continue"

$SEED = 42
$BATCH = 16
$EPOCHS = 3
$NULL_W = 0.2

$TRAIN_AUG = "data/splits/train_aug.parquet"
$TRAIN_PY = "src/models/train_roberta_improved.py"
$EVAL_PY = "src/evaluation/evaluate_and_log.py"
$VAL_PATH = "data/splits/val.parquet"

$ABL_DIR = "outputs/ablations_4class"
New-Item -ItemType Directory -Force -Path $ABL_DIR | Out-Null

function Start-Ablation {
    param($Tag, $UseSampler, $UseSynthetic)
    
    $SamplerFlag = if ($UseSampler) { "--use_sampler" } else { "" }
    $SynthFlag = if ($UseSynthetic) { "--use_synthetic --train_aug_path $TRAIN_AUG" } else { "" }
    
    foreach ($Msr in @(0.0, 0.3)) {
        $MsrTag = if ($Msr -eq 0.0) { "base" } else { "msr" }
        $FullTag = "${Tag}_${MsrTag}"
        $TrainDir = "$ABL_DIR/train_$FullTag"
        $EvalDir = "$ABL_DIR/eval_$FullTag"
        
        Write-Host "`n>>> RUNNING: $FullTag (Sampler=$UseSampler, Synth=$UseSynthetic, MSR=$Msr)" -ForegroundColor Cyan
        
        # Train
        $trainCmd = "python $TRAIN_PY --msr_strength $Msr --out_dir $TrainDir --null_weight $NULL_W --epochs $EPOCHS --batch_size $BATCH $SamplerFlag $SynthFlag --seed $SEED"
        Write-Host $trainCmd
        Invoke-Expression $trainCmd
        
        # Eval
        $evalCmd = "python $EVAL_PY --ckpt $TrainDir/best_model.pt --val_path $VAL_PATH --out_dir $EvalDir --msr_strength $Msr --save_predictions"
        Write-Host $evalCmd
        Invoke-Expression $evalCmd
    }
}

# 1. Base (No Sampler, No Synth)
Start-Ablation -Tag "1_base" -UseSampler $false -UseSynthetic $false
Start-Ablation -Tag "2_sampler" -UseSampler $true -UseSynthetic $false
Start-Ablation -Tag "3_synth" -UseSampler $false -UseSynthetic $true
Start-Ablation -Tag "4_full" -UseSampler $true -UseSynthetic $true

Write-Host "`n✅ All Ablation Runs Complete!" -ForegroundColor Green
