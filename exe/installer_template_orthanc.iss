#define MyAppName "ImageCLASP_Orthanc"
#define MyAppVersion "1.0"
#define MyAppPublisher "UCL Institute of Cardiovascular Science"
#define MyAppExeName "ImageCLASP_Orthanc.exe"
#define OrthancExeName "orthanc.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-47A8-90AB-CDEF12345678}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=installer
OutputBaseFilename=ImageCLASP_Orthanc_Setup
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\ImageCLASP\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

Source: "..\dist\Orthanc\*"; DestDir: "{app}\Orthanc"; Flags: ignoreversion recursesubdirs createallsubdirs

Source: "VC_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{group}\Orthanc Console"; Filename: "{app}\Orthanc\{#OrthancExeName}"

[Run]
Filename: "{tmp}\VC_redist.x64.exe"; Parameters: "/install /passive /norestart"; Flags: waituntilterminated runhidden

; Install Orthanc as a Windows service
Filename: "sc.exe"; Parameters: "create Orthanc binPath= \"{app}\Orthanc\{#OrthancExeName}\" start= auto"; Flags: runhidden waituntilterminated
Filename: "sc.exe"; Parameters: "start Orthanc"; Flags: runhidden waituntilterminated

; Launch main app
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent