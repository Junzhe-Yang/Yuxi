param(
    [string]$ArchiveName = "Yuxi-review-20260813.zip",
    [string]$HandoffReadmeName = "Yuxi-handoff-20260813-README.md"
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$archivePath = Join-Path $PSScriptRoot $ArchiveName
$partialPath = "$archivePath.partial"
$handoffReadme = Join-Path $PSScriptRoot $HandoffReadmeName
$archiveRoot = [System.IO.Path]::GetFileNameWithoutExtension($ArchiveName)

if (Test-Path -LiteralPath $archivePath) {
    throw "Archive already exists: $archivePath"
}
if (Test-Path -LiteralPath $partialPath) {
    throw "Partial archive already exists: $partialPath"
}
if (-not (Test-Path -LiteralPath $handoffReadme)) {
    throw "Handoff README is missing: $handoffReadme"
}

$excludedDirectoryNames = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
@(
    ".git",
    ".agents",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".mypy_cache",
    ".cache",
    "cache",
    ".tox",
    ".nox",
    ".playwright",
    ".playwright-cli",
    ".playwright-mcp",
    ".vscode",
    ".idea",
    ".codex",
    ".cursor",
    ".claude",
    ".vibe",
    ".qoder",
    ".trae",
    ".sisyphus",
    ".taskr",
    "dist",
    "build",
    "tmp",
    "temp",
    "models",
    "saves",
    "saves_dev",
    "deliverables"
) | ForEach-Object { [void]$excludedDirectoryNames.Add($_) }

function Test-ExcludedFile {
    param([System.IO.FileInfo]$File)

    $name = $File.Name
    $lowerName = $name.ToLowerInvariant()

    if ($lowerName -in @(".ds_store", "thumbs.db", ".coverage")) {
        return $true
    }
    if ($lowerName -match "\.(pyc|pyo|log|tmp|temp|bak|swp|swo|db|sqlite|sqlite3|pem|key|pfx|p12)$") {
        return $true
    }
    if ($lowerName -match "\.log\.") {
        return $true
    }
    if ($lowerName.EndsWith("~")) {
        return $true
    }
    if ($lowerName -eq ".envrc") {
        return $true
    }
    if ($lowerName -match "^\.env($|\.)" -and $lowerName -notmatch "(example|template|sample)") {
        return $true
    }
    if ($lowerName -match "(^|[._-])(secret|credentials?|private)([._-]|$)") {
        return $true
    }

    return $false
}

function Get-PackageFiles {
    param([string]$Directory)

    foreach ($item in Get-ChildItem -LiteralPath $Directory -Force) {
        if ($item.PSIsContainer) {
            if (
                $excludedDirectoryNames.Contains($item.Name) -or
                $item.Name -match "^\.(uv|ruff|pytest|mypy)-cache($|-)"
            ) {
                continue
            }
            if ($item.FullName -eq (Join-Path $projectRoot "docker\volumes")) {
                continue
            }
            Get-PackageFiles -Directory $item.FullName
            continue
        }

        if (-not (Test-ExcludedFile -File $item)) {
            $item
        }
    }
}

function Get-ProjectRelativePath {
    param([string]$Path)

    $rootPrefix = $projectRoot.TrimEnd("\") + "\"
    if (-not $Path.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside project root: $Path"
    }
    return $Path.Substring($rootPrefix.Length)
}

$files = @(
    Get-PackageFiles -Directory $projectRoot |
        Sort-Object { Get-ProjectRelativePath -Path $_.FullName }
)

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$fileStream = $null
$archive = $null
try {
    $fileStream = [System.IO.File]::Open(
        $partialPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
    $archive = [System.IO.Compression.ZipArchive]::new(
        $fileStream,
        [System.IO.Compression.ZipArchiveMode]::Create,
        $false,
        [System.Text.Encoding]::UTF8
    )

    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
        $archive,
        $handoffReadme,
        "$archiveRoot/HANDOFF_README.md",
        [System.IO.Compression.CompressionLevel]::Optimal
    ) | Out-Null

    foreach ($file in $files) {
        $relativePath = Get-ProjectRelativePath -Path $file.FullName
        $entryPath = "$archiveRoot/" + $relativePath.Replace("\", "/")
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive,
            $file.FullName,
            $entryPath,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }

    $listEntry = $archive.CreateEntry(
        "$archiveRoot/PACKAGE_FILELIST.txt",
        [System.IO.Compression.CompressionLevel]::Optimal
    )
    $listStream = $listEntry.Open()
    $writer = [System.IO.StreamWriter]::new(
        $listStream,
        [System.Text.UTF8Encoding]::new($false)
    )
    try {
        $writer.WriteLine("HANDOFF_README.md")
        $writer.WriteLine("PACKAGE_FILELIST.txt")
        foreach ($file in $files) {
            $relativePath = Get-ProjectRelativePath -Path $file.FullName
            $writer.WriteLine($relativePath.Replace("\", "/"))
        }
    }
    finally {
        $writer.Dispose()
        $listStream.Dispose()
    }
}
finally {
    if ($null -ne $archive) {
        $archive.Dispose()
    }
    if ($null -ne $fileStream) {
        $fileStream.Dispose()
    }
}

Move-Item -LiteralPath $partialPath -Destination $archivePath

$sourceBytes = ($files | Measure-Object -Property Length -Sum).Sum
$archiveInfo = Get-Item -LiteralPath $archivePath
Write-Output "PACKAGE_PATH=$archivePath"
Write-Output "SOURCE_FILE_COUNT=$($files.Count)"
Write-Output "SOURCE_BYTES=$sourceBytes"
Write-Output "ARCHIVE_BYTES=$($archiveInfo.Length)"
