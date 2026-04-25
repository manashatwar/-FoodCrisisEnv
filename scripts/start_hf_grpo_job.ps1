[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceSpaceRepo,

    [Parameter(Mandatory = $true)]
    [string]$OutputModelRepo,

    [ValidateSet("manual", "trl")]
    [string]$Mode = "manual",

    [string]$Flavor = "a10g-small",
    [string]$Image = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    [string]$Model = "Qwen/Qwen2.5-1.5B-Instruct",
    [int]$Episodes = 30,
    [int]$GroupSize = 4,
    [double]$LearningRate = 5e-6,
    [double]$Temperature = 0.9,
    [int]$Seed = 7,
    [string]$Timeout = "6h",
    [string]$Namespace = "",
    [string]$SavePath = "/tmp/grpo_trained_model",
    [switch]$LoRA,
    [switch]$PrintOnly
)

$pythonExe = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Expected Python at $pythonExe"
}

function Quote-Bash([string]$Value) {
    return "'" + $Value.Replace("'", "'""'""'") + "'"
}

$trainArgs = @(
    "--mode", $Mode,
    "--model", $Model,
    "--episodes", "$Episodes",
    "--group-size", "$GroupSize",
    "--learning-rate", "$LearningRate",
    "--temperature", "$Temperature",
    "--seed", "$Seed",
    "--device", "cuda",
    "--save-path", $SavePath,
    "--push-to-hub-repo", $OutputModelRepo
)

if ($LoRA) {
    $trainArgs += "--lora"
}

$quotedTrainArgs = ($trainArgs | ForEach-Object { Quote-Bash $_ }) -join " "

$bashCommand = @(
    "python -m pip install --upgrade pip",
    "python -m pip install -e /workspace accelerate datasets huggingface_hub peft transformers trl",
    "python /workspace/train_grpo.py $quotedTrainArgs"
) -join " && "

$hfArgs = @(
    "-m", "huggingface_hub.cli.hf",
    "jobs", "run",
    "--flavor", $Flavor,
    "--timeout", $Timeout,
    "--secrets", "HF_TOKEN",
    "-v", "hf://spaces/${SourceSpaceRepo}:/workspace:ro",
    "-d"
)

if ($Namespace) {
    $hfArgs += @("--namespace", $Namespace)
}

$hfArgs += @(
    $Image,
    "bash",
    "-lc",
    $bashCommand
)

Write-Host "Submitting HF job with flavor $Flavor using source space $SourceSpaceRepo"
Write-Host "Output model repo: $OutputModelRepo"

if ($PrintOnly) {
    $rendered = @($pythonExe) + $hfArgs
    Write-Host ""
    Write-Host ($rendered -join " ")
    exit 0
}

& $pythonExe @hfArgs
