param(
    [Parameter(Mandatory=$false)]
    [string]$RunId,

    [Parameter(Mandatory=$false)]
    [switch]$WithCkpts,

    [Parameter(Mandatory=$false)]
    [string]$HostName = "ps@219.230.66.29",

    [Parameter(Mandatory=$false)]
    [string]$RemoteLogsRoot = "/home/ps/Code/yqy/DINO-WM/jepawm_logs",

    [Parameter(Mandatory=$false)]
    [string]$LocalResultsRoot = "E:\jepa-wms\experiments\results",

    [Parameter(Mandatory=$false)]
    [switch]$Help
)

if ($Help -or [string]::IsNullOrWhiteSpace($RunId)) {
    Write-Host "Usage: .\pull_run.ps1 -RunId <run_id> [-WithCkpts]"
    Write-Host "Pulls small run artifacts from lab01 into E:\jepa-wms\experiments\results\<run_id>."
    Write-Host "Use -WithCkpts to also pull jepa-e40..jepa-e49 checkpoint files."
    exit $(if ($Help) { 0 } else { 2 })
}

$ErrorActionPreference = "Stop"

$remoteRun = "$RemoteLogsRoot/$RunId"
$localRun = Join-Path $LocalResultsRoot $RunId
New-Item -ItemType Directory -Force -Path $localRun | Out-Null

function Test-RemotePath {
    param([string]$Path)
    & ssh $HostName "test -e '$Path'"
    return ($LASTEXITCODE -eq 0)
}

function Copy-RemotePath {
    param(
        [string]$RemotePath,
        [string]$LocalPath,
        [switch]$Required
    )
    if (Test-RemotePath $RemotePath) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LocalPath) | Out-Null
        & scp -r "${HostName}:$RemotePath" "$LocalPath"
        if ($LASTEXITCODE -ne 0) {
            throw "scp failed for $RemotePath"
        }
    } elseif ($Required) {
        throw "required remote artifact missing: $RemotePath"
    } else {
        Write-Warning "missing optional remote artifact: $RemotePath"
    }
}

$smallFiles = @(
    "config.yaml",
    "params-pretrain.yaml",
    "launcher.sh",
    "git_commit.txt",
    "log_r0.csv",
    "launch.log",
    "eval_last10/summary.csv"
)

foreach ($file in $smallFiles) {
    Copy-RemotePath -RemotePath "$remoteRun/$file" -LocalPath (Join-Path $localRun $file)
}

if (Test-RemotePath "$remoteRun/eval_last10") {
    Copy-RemotePath -RemotePath "$remoteRun/eval_last10" -LocalPath (Join-Path $localRun "eval_last10")
}

if ($WithCkpts) {
    foreach ($epoch in 40..49) {
        $ckpt = "jepa-e$epoch.pth.tar"
        Copy-RemotePath -RemotePath "$remoteRun/$ckpt" -LocalPath (Join-Path $localRun $ckpt)
    }
}

Write-Host "Pulled artifacts for $RunId into $localRun"
