# 제출 DB 덤프(data/db/council_vectors.dump)를 PostgreSQL+pgvector 컨테이너에 복원합니다. (Windows PowerShell)
#   powershell -ExecutionPolicy Bypass -File scripts\restore_db.ps1            # 처음 복원
#   powershell -ExecutionPolicy Bypass -File scripts\restore_db.ps1 -Replace   # 기존 테이블을 지우고 다시 복원
# HNSW 인덱스를 다시 만들기 때문에 수십 분 걸릴 수 있습니다.
param([switch]$Replace)
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Dump = Join-Path $Root "data\db\council_vectors.dump"
$User = "council"; $Db = "council_vectors"; $Table = "council_arctic_raw_v1"

if (-not (Test-Path $Dump)) { throw "덤프 파일이 없습니다: $Dump" }

# 1) 무결성 확인
$Expected = (Get-Content "$Dump.sha256" -Raw).Split(" ")[0].Trim()
Write-Host "SHA-256 확인 중..."
$Actual = (Get-FileHash $Dump -Algorithm SHA256).Hash.ToLower()
if ($Actual -ne $Expected) { throw "체크섬 불일치: 파일이 손상되었습니다." }

# 2) 컨테이너 시작·대기
docker compose up -d pgvector | Out-Null
for ($i = 0; $i -lt 60; $i++) {
    docker compose exec -T pgvector pg_isready -U $User -d $Db 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 1
}

# 3) 기존 테이블 확인
$Existing = docker compose exec -T pgvector psql -At -U $User -d $Db -c "SELECT COALESCE(to_regclass('public.$Table')::text, '');"
if ($Existing -and -not $Replace) { throw "이미 $Table 테이블이 있습니다. 교체하려면 -Replace를 붙이세요." }

# 4) 복원 (덤프는 docker-compose.yml에서 /dump로 마운트됨)
$Args = @("-U", $User, "-d", $Db, "--no-owner", "--no-privileges", "--exit-on-error", "--jobs=4")
if ($Replace) { $Args += @("--clean", "--if-exists") }
Write-Host "복원 중... (HNSW 인덱스 재생성 포함)"
docker compose exec -T pgvector pg_restore @Args /dump/council_vectors.dump
if ($LASTEXITCODE -ne 0) { throw "pg_restore 실패" }

$Count = docker compose exec -T pgvector psql -At -U $User -d $Db -c "SELECT count(*) FROM public.$Table;"
Write-Host "=== 복원 완료: $Table $Count 행 ==="
