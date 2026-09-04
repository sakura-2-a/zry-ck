# 课设答辩.pptx -> PDF（PowerPoint COM 自动化，32 = ppSaveAsPDF）
$ErrorActionPreference = "Stop"
$src = "C:\Users\Administrator\Desktop\轴承缺陷项目\docs\课设答辩.pptx"
$dst = "C:\Users\Administrator\Desktop\轴承缺陷项目\docs\课设答辩.pdf"

$ppt = New-Object -ComObject PowerPoint.Application
try {
    # Open(FileName, ReadOnly, Untitled, WithWindow)
    $pres = $ppt.Presentations.Open($src, $true, $true, $false)
    # 32 = ppSaveAsPDF
    $pres.SaveAs($dst, 32)
    $pres.Close()
    Write-Output "PDF saved: $dst"
} finally {
    $ppt.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt) | Out-Null
}
