param(
  [string]$SourcePublic = "public",
  [string]$SourceModel = "model",
  [string]$Dest = "docs"
)

Write-Host "Preparing GitHub Pages site..." -ForegroundColor Cyan

if (Test-Path $Dest) {
  Write-Host "Removing existing '$Dest'..." -ForegroundColor DarkYellow
  Remove-Item -Recurse -Force $Dest
}

Write-Host "Copying UI from '$SourcePublic' to '$Dest'..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path $Dest | Out-Null
Copy-Item -Recurse -Force "$SourcePublic\*" $Dest

if (-Not (Test-Path $SourceModel)) {
  Write-Warning "Model folder '$SourceModel' not found. Run training/export to create TF.js model (model/model.json)."
} else {
  Write-Host "Copying model from '$SourceModel' to '$Dest\model'..." -ForegroundColor Cyan
  New-Item -ItemType Directory -Path "$Dest\model" -Force | Out-Null
  Copy-Item -Recurse -Force "$SourceModel\*" "$Dest\model"
}

Write-Host "Done. Configure GitHub Pages to serve from 'main' branch /docs." -ForegroundColor Green
