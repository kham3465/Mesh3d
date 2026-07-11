<#
.SYNOPSIS
    Build Docker image tu Dockerfile va push len AWS ECR.

.EXAMPLE
    # Full args
    .\implement.ps1 \
    -Dockerfile Dockerfile \
    -BuildContext "." \
    -ImageTag v14 \
    -AwsRegion ap-southeast-1 \
    -RepoName view-app \
    -BuildArgs @{ APP_ENV = "prod"; PORT = "8080" }

    # Dạng cơ bản dùng luôn:
    .\implement.ps1 -BuildArgs @{ APP_ENV = "prod"; PORT = "8080" }
#>

param(
    [Parameter(Position = 0)]
    [string]$ImageTag = "latest",

    [string]$AwsRegion = "ap-southeast-1",

    [string]$RepoName = "view-app",

    [string]$Dockerfile = "Dockerfile",

    [string]$BuildContext = ".",

    [hashtable]$BuildArgs,

    # --- Deploy len EC2 sau khi push (tuy chon) ---
    # Truyen -SshHost + -SshKey de tu dong goi deploy_ec2.ps1 sau khi push xong.
    [string]$SshHost = "13.213.16.139",

    [string]$SshKey = (Join-Path $PSScriptRoot "iot-ec2-key.pem"),

    [string]$SshUser = "ec2-user",

    # Duong dan file compose tren EC2. Bo trong -> deploy_ec2.ps1 tu tim.
    [string]$RemoteComposePath = "",

    # Ten service app trong compose (kem --no-deps). Bo trong -> toan bo compose.
    [string]$ServiceName = "view-service"
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# Tra ve Account ID neu xac thuc AWS thanh cong, nguoc lai tra ve $null.
function Get-AwsAccountId {
    # Ha EAP cuc bo: tranh loi NativeCommandError khi redirect stderr cua aws (WinPS 5.1).
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    $id = aws sts get-caller-identity --query Account --output text 2>$null
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -eq 0 -and $id) { return "$id".Trim() }
    return $null
}

# Hoi nguoi dung nhap Access Key + Secret Key, set credentials cho phien hien tai.
function Request-AwsCredentials {
    Write-Host "Hay cung cap thong tin de xac thuc:" -ForegroundColor Yellow
    $accessKey = Read-Host "AWS Access Key ID"
    $secretSecure = Read-Host "AWS Secret Access Key" -AsSecureString
    $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secretSecure)
    $secretKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)

    # Set cho phien PowerShell hien tai (khong ghi ra dia).
    $env:AWS_ACCESS_KEY_ID     = $accessKey.Trim()
    $env:AWS_SECRET_ACCESS_KEY = $secretKey
    $env:AWS_DEFAULT_REGION    = $AwsRegion
}

# 0. Kiem tra Docker
Write-Step "Kiem tra Docker"
docker info | Out-Null
if (-not $?) { throw "Docker chua chay. Hay mo Docker Desktop roi thu lai." }

# 1. Dam bao da xac thuc AWS - lap hoi key cho den khi nhap dung
Write-Step "Kiem tra xac thuc AWS"
$accountId = Get-AwsAccountId
while (-not $accountId) {
    Request-AwsCredentials
    $accountId = Get-AwsAccountId
    if (-not $accountId) {
        Write-Host "Xac thuc that bai. Vui long kiem tra va nhap lai." -ForegroundColor Red
    }
}
Write-Host "Da xac thuc AWS - Account ID: $accountId" -ForegroundColor Green

$ecrUri = "$accountId.dkr.ecr.$AwsRegion.amazonaws.com"

# 2. Dang nhap Docker vao ECR
Write-Step "Dang nhap Docker vao ECR ($ecrUri)"
aws ecr get-login-password --region $AwsRegion | docker login --username AWS --password-stdin $ecrUri
if (-not $?) { throw "Dang nhap ECR that bai." }

$localRef  = "${RepoName}:${ImageTag}"
$remoteRef = "$ecrUri/${RepoName}:${ImageTag}"

if (-not (Test-Path $Dockerfile)) { throw "Khong tim thay Dockerfile: '$Dockerfile'" }
Write-Host "Dung Dockerfile: $Dockerfile | context: $BuildContext" -ForegroundColor Gray


Write-Step "Kiem tra / tao ECR repository '$RepoName'"
$ErrorActionPreference = 'SilentlyContinue'
aws ecr describe-repositories --repository-names $RepoName --region $AwsRegion 2>$null | Out-Null
$repoExists = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = 'Stop'
if (-not $repoExists) {
    aws ecr create-repository --repository-name $RepoName --region $AwsRegion | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Tao repository that bai." }
    Write-Host "Da tao repository moi." -ForegroundColor Green
} else {
    Write-Host "Repository da ton tai - tiep tuc." -ForegroundColor Gray
}

# Build image
Write-Step "Build image: $localRef"
$buildCmd = @("build", "-t", $localRef, "-f", $Dockerfile)
if ($BuildArgs) {
    foreach ($key in $BuildArgs.Keys) {
        $buildCmd += @("--build-arg", "$key=$($BuildArgs[$key])")
    }
}
$buildCmd += $BuildContext
docker @buildCmd
if (-not $?) { throw "Build image that bai." }

# Tag tro ve ECR
Write-Step "Tag image: $remoteRef"
docker tag $localRef $remoteRef
if (-not $?) { throw "Tag image that bai." }

# Push
Write-Step "Push image len ECR"
docker push $remoteRef
if (-not $?) { throw "Push image that bai." }

Write-Step "HOAN TAT"
Write-Host "Image da day len: $remoteRef" -ForegroundColor Green
Write-Host ""
Write-Host "--- Copy dong duoi vao docker-compose.yml ---" -ForegroundColor Cyan
Write-Host "    image: $remoteRef" -ForegroundColor Yellow
Write-Host "---------------------------------------------" -ForegroundColor Cyan

# Copy image link vao clipboard cho tien dan vao docker-compose
try { $remoteRef | Set-Clipboard; Write-Host "(Da copy image link vao clipboard)" -ForegroundColor Gray } catch {}

# --- Tu dong deploy len EC2 (neu co truyen -SshHost va -SshKey) ---
if ($SshHost -and $SshKey) {
    Write-Step "Goi deploy_ec2.ps1 de trien khai len EC2"
    $deployScript = Join-Path $PSScriptRoot "deploy_ec2.ps1"
    if (-not (Test-Path $deployScript)) { throw "Khong tim thay deploy_ec2.ps1 canh implement.ps1: '$deployScript'" }

    $deployArgs = @{
        ImageRef  = $remoteRef
        SshHost   = $SshHost
        SshKey    = $SshKey
        SshUser   = $SshUser
        AwsRegion = $AwsRegion
    }
    if ($RemoteComposePath) { $deployArgs.RemoteComposePath = $RemoteComposePath }
    if ($ServiceName)       { $deployArgs.ServiceName       = $ServiceName }

    & $deployScript @deployArgs
    if (-not $?) { throw "deploy_ec2.ps1 that bai." }
} else {
    Write-Host "(Bo qua deploy EC2 - khong co -SshHost/-SshKey. Chay deploy_ec2.ps1 thu cong neu can.)" -ForegroundColor Gray
}