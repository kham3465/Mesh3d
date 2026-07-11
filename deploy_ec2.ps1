<#
.SYNOPSIS
    Cap nhat image link vao docker-compose.yml tren EC2 (qua SSH) roi pull + up -d.
    Dung sau khi da build & push image len ECR bang implement.ps1.

.EXAMPLE
    # Truyen thang image ref
    .\deploy_ec2.ps1 -ImageRef 123.dkr.ecr.ap-southeast-1.amazonaws.com/my-iot-project:v14 `
                     -SshHost 1.2.3.4 -SshKey C:\keys\ec2.pem

.EXAMPLE
    # Khong truyen -ImageRef -> tu lay tu clipboard (implement.ps1 da copy san)
    .\deploy_ec2.ps1 -SshHost ec2-1-2-3-4.compute.amazonaws.com -SshKey C:\keys\ec2.pem -SshUser ubuntu
#>

param(
    # Image link day du. Bo trong -> lay tu clipboard.
    [string]$ImageRef = (Get-Clipboard),

    [Parameter(Mandatory = $true)]
    [string]$SshHost,

    [Parameter(Mandatory = $true)]
    [string]$SshKey,

    [string]$SshUser = "ec2-user",

    # Duong dan file compose tren EC2. Bo trong -> script tu do tim (auto-discovery).
    [string]$RemoteComposePath = "",

    # Ten service app trong compose. Truyen vao de pull/up CHI service nay
    # (kem --no-deps) -> dam bao nginx & cac service khac KHONG bi anh huong.
    # Bo trong -> ap dung cho toan bo compose (canh bao: co the dung den nginx).
    [string]$ServiceName = "",

    [string]$AwsRegion = "ap-southeast-1"
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# --- Kiem tra dau vao ---
$ImageRef = "$ImageRef".Trim()
if (-not $ImageRef -or $ImageRef -notmatch '^\S+/\S+:\S+$') {
    throw "ImageRef khong hop le: '$ImageRef'. Mong doi dang <registry>/<repo>:<tag>."
}
if (-not (Test-Path $SshKey)) { throw "Khong tim thay SSH key: '$SshKey'" }

# Chay 1 doan bash tren EC2 qua SSH, tra ve stdout.
# Luu y: KHONG pipe script qua stdin. Windows PowerShell 5.1 khi pipe chuoi vao stdin
# cua native command se chen BOM (EF BB BF) va doi sang CRLF -> bash bao loi
# "﻿{: command not found" / "syntax error". Thay vao do base64-encode (UTF-8/LF) roi
# truyen nhu argument, decode + chay tren remote -> mien nhiem BOM/CRLF.
function Invoke-Ssh($script) {
    $clean = $script -replace "`r`n", "`n"
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($clean))
    return (ssh -i $SshKey -o StrictHostKeyChecking=accept-new "$SshUser@$SshHost" "echo $b64 | base64 -d | bash -s")
}

# --- Auto-discovery: tu tim file compose tren EC2 neu khong duoc chi dinh ---
if (-not $RemoteComposePath) {
    Write-Step "Tu do tim docker-compose tren EC2"
    $discovery = @'
{
  ls -1 ~/docker-compose.yml ~/docker-compose.yaml ~/compose.yml ~/compose.yaml 2>/dev/null
  find ~ /opt /srv /home -maxdepth 4 \( -name 'docker-compose.y*ml' -o -name 'compose.y*ml' \) 2>/dev/null
} | sort -u
'@
    $candidates = @(Invoke-Ssh $discovery | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    if ($LASTEXITCODE -ne 0) { throw "Khong ket noi duoc EC2 (ssh exit $LASTEXITCODE). Kiem tra SshHost/SshKey/SshUser va Security Group cong 22." }

    if ($candidates.Count -eq 0) {
        throw "Khong tim thay file compose nao tren EC2. Hay truyen -RemoteComposePath."
    } elseif ($candidates.Count -eq 1) {
        $RemoteComposePath = $candidates[0]
        Write-Host "Tim thay: $RemoteComposePath" -ForegroundColor Green
    } else {
        Write-Host "Tim thay nhieu file compose:" -ForegroundColor Yellow
        $candidates | ForEach-Object { Write-Host "  $_" }
        throw "Co nhieu file compose. Hay chon 1 bang -RemoteComposePath."
    }
}

# Registry (phan truoc dau '/' dau tien) - dung de docker login tren EC2.
$ecrRegistry = ($ImageRef -split '/')[0]
# Repo khong kem tag (bo phan ':tag' cuoi cung) - dung lam moc thay the trong YAML.
$repoNoTag = $ImageRef -replace ':[^:/]*$', ''
# Escape dau '.' cho sed ERE (dung '|' lam delimiter nen khong can escape '/').
$repoSed = $repoNoTag -replace '\.', '\.'

# Chuong trinh awk: thay dong "image:" NGAY TRONG service block duoc chi dinh,
# BAT KE image/app name cu la gi -> cho phep DOI APP NAME (repo moi != repo cu).
# Dung single-quote here-string ('@...@') de PowerShell KHONG noi suy '$0', '$svc'...
$awkProgram = @'
function indent(s){ if (match(s, /^ +/)) return RLENGTH; return 0 }
BEGIN { inblk=0; svcind=-1; done=0 }
{
  line=$0
  if (match(line, /^[ \t]*[A-Za-z0-9._-]+:[ \t]*(#.*)?$/)) {
    ind=indent(line)
    name=line; sub(/^[ \t]*/,"",name); sub(/:.*/,"",name)
    if (inblk && ind <= svcind) inblk=0
    if (name == svc) { inblk=1; svcind=ind }
    print line; next
  }
  if (inblk) { ind=indent(line); if (line ~ /[^ \t]/ && ind <= svcind) inblk=0 }
  if (inblk && !done && match(line, /^[ \t]*image:[ \t]/)) {
    ind=indent(line); pad=""; for (i=0;i<ind;i++) pad=pad " "
    print pad "image: " img; done=1; next
  }
  print line
}
END { if (!done) exit 3 }
'@
$awkB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($awkProgram -replace "`r`n","`n")))

# Gioi han lenh chi vao service app (kem --no-deps) de KHONG dung den nginx & service khac.
if ($ServiceName) {
    $pullLine = "docker compose -f `"`$COMPOSE`" pull $ServiceName"
    $upLine   = "docker compose -f `"`$COMPOSE`" up -d --no-deps $ServiceName"
    $scopeMsg = "service '$ServiceName' (nginx & service khac KHONG bi anh huong)"
} else {
    $pullLine = "docker compose -f `"`$COMPOSE`" pull"
    $upLine   = "docker compose -f `"`$COMPOSE`" up -d"
    $scopeMsg = "TOAN BO compose (canh bao: co the dung den nginx)"
}

# --- Buoc cap nhat dong image trong YAML ---
if ($ServiceName) {
    # Thay image NGAY TRONG service block -> DOI APP NAME (repo moi) van chay.
    $imageStep = @"
echo $awkB64 | base64 -d > /tmp/upd_img.awk
awk -v svc="$ServiceName" -v img="$ImageRef" -f /tmp/upd_img.awk "`$COMPOSE" > "`$COMPOSE.tmp" && mv "`$COMPOSE.tmp" "`$COMPOSE" || { echo "Khong tim thay service '$ServiceName' (hoac thieu dong image) trong `$COMPOSE" >&2; rm -f "`$COMPOSE.tmp"; exit 1; }
echo "--- image sau khi cap nhat ---"
grep -nF "image: $ImageRef" "`$COMPOSE" || { echo "Xac minh that bai: khong thay '$ImageRef'" >&2; exit 1; }
"@
} else {
    # Fallback (khong truyen -ServiceName): thay theo repo cu -> repo moi PHAI trung repo cu.
    $imageStep = @"
sed -i -E "s|image:[[:space:]]*${repoSed}:[^[:space:]]*|image: ${ImageRef}|g" "`$COMPOSE"
echo "--- image sau khi cap nhat ---"
grep -nE "image:[[:space:]]*${repoSed}:" "`$COMPOSE" || { echo "Khong tim thay dong image khop voi ${repoNoTag}. Hay truyen -ServiceName de doi app name." >&2; exit 1; }
"@
}

Write-Step "Trien khai len EC2: $SshUser@$SshHost"
Write-Host "Image moi : $ImageRef"      -ForegroundColor Gray
Write-Host "Compose   : $RemoteComposePath" -ForegroundColor Gray
Write-Host "Pham vi   : $scopeMsg" -ForegroundColor Gray

# --- Script chay tren EC2 (bash). Luu y: `$ = bien bash (khong cho PowerShell mo rong) ---
$remoteScript = @"
set -e
COMPOSE="$RemoteComposePath"
COMPOSE=`$(eval echo "`$COMPOSE")              # no '~' thanh duong dan that
if [ ! -f "`$COMPOSE" ]; then echo "Khong tim thay `$COMPOSE" >&2; exit 1; fi

# 1. Sao luu truoc khi sua
cp "`$COMPOSE" "`$COMPOSE.bak.`$(date +%Y%m%d%H%M%S)"

# 2. Cap nhat dong image (theo service block neu co -ServiceName, nguoc lai theo repo cu)
${imageStep}

# 3. Dang nhap ECR tren EC2 (can IAM role hoac aws configure san)
aws ecr get-login-password --region ${AwsRegion} | docker login --username AWS --password-stdin ${ecrRegistry}

# 4. Pull image moi va khoi dong lai (chi service app neu co chi dinh)
cd "`$(dirname "`$COMPOSE")"
${pullLine}
${upLine}
docker compose -f "`$COMPOSE" ps

# 5. Deploy thanh cong (set -e: toi day nghia la khong loi) -> xoa cac file backup .bak.*
rm -f "`$COMPOSE".bak.*
echo "Da don dep cac file backup `$COMPOSE.bak.*"
"@

# --- Gui script qua SSH, chay bang bash (qua Invoke-Ssh de tranh BOM/CRLF) ---
Invoke-Ssh $remoteScript
if ($LASTEXITCODE -ne 0) { throw "Trien khai tren EC2 that bai (exit $LASTEXITCODE)." }

Write-Step "HOAN TAT"
Write-Host "Da cap nhat va deploy '$ImageRef' tren $SshHost" -ForegroundColor Green
