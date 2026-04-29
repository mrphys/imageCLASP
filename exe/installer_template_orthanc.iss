#define MyAppName "ImageCLASP"
#define MyAppVersion "1.0"
#define MyAppPublisher "UCL Institute of Cardiovascular Science"
#define MyAppExeName "ImageCLASP.exe"

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
Source: "OrthancInstaller.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "VC_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Install VC++ runtime
Filename: "{tmp}\VC_redist.x64.exe"; Parameters: "/install /passive /norestart"; Flags: waituntilterminated runhidden

; Install Orthanc first (must support silent install)
Filename: "{tmp}\OrthancInstaller.exe"; Parameters: "/S /DIR=""{app}\Orthanc"""; Flags: waituntilterminated

; Create Orthanc Windows service
Filename: "sc.exe"; Parameters: "create Orthanc binPath= ""{app}\Orthanc\orthanc.exe"" start= auto"; Flags: runhidden waituntilterminated

; Start Orthanc service
Filename: "sc.exe"; Parameters: "start Orthanc"; Flags: runhidden waituntilterminated

; Launch main application (does NOT affect Orthanc server)
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppExeName}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Stop and remove Orthanc service on uninstall
Filename: "sc.exe"; Parameters: "stop Orthanc"; Flags: runhidden waituntilterminated
Filename: "sc.exe"; Parameters: "delete Orthanc"; Flags: runhidden waituntilterminated