; ============================================================================
;  SENTINEL MEDICAL AI — Custom NSIS installer hooks
;  Bilingual: text shown in Russian (primary) + English (in parens)
; ============================================================================

!macro customInit
  ; Medical disclaimer in Russian + English
  MessageBox MB_OKCANCEL|MB_ICONINFORMATION \
    "Sentinel Medical AI — вспомогательный инструмент для лицензированных клиник.$\r$\n\
     (Sentinel Medical AI is an assistive radiology tool for licensed clinics.)$\r$\n$\r$\n\
     • Результаты ИИ требуют подтверждения врачом-рентгенологом.$\r$\n\
       (AI results require physician review.)$\r$\n$\r$\n\
     • НЕ используйте как единственное основание для диагноза.$\r$\n\
       (Do NOT use as the sole basis for diagnosis.)$\r$\n$\r$\n\
     • Данные пациентов остаются на этом ПК — не передаются в интернет.$\r$\n\
       (Patient data stays on this machine — never sent to cloud.)$\r$\n$\r$\n\
     Продолжить установку? (Continue with installation?)" \
    IDOK +2
    Quit
!macroend

!macro customInstall
  ; Create user data dirs
  CreateDirectory "$APPDATA\Sentinel Medical AI"
  CreateDirectory "$APPDATA\Sentinel Medical AI\models"
  CreateDirectory "$APPDATA\Sentinel Medical AI\dicom"
  CreateDirectory "$APPDATA\Sentinel Medical AI\reports"
  CreateDirectory "$APPDATA\Sentinel Medical AI\corrections"
  CreateDirectory "$APPDATA\Sentinel Medical AI\logs"

  ; Write default config
  FileOpen $0 "$APPDATA\Sentinel Medical AI\config.ini" w
  FileWrite $0 "[paths]$\r$\n"
  FileWrite $0 "models=$APPDATA\Sentinel Medical AI\models$\r$\n"
  FileWrite $0 "dicom=$APPDATA\Sentinel Medical AI\dicom$\r$\n"
  FileWrite $0 "reports=$APPDATA\Sentinel Medical AI\reports$\r$\n"
  FileWrite $0 "corrections=$APPDATA\Sentinel Medical AI\corrections$\r$\n"
  FileWrite $0 "logs=$APPDATA\Sentinel Medical AI\logs$\r$\n"
  FileWrite $0 "$\r$\n[server]$\r$\n"
  FileWrite $0 "host=127.0.0.1$\r$\n"
  FileWrite $0 "port=8000$\r$\n"
  FileClose $0

  ; Russian-first activation prompt
  MessageBox MB_OK|MB_ICONINFORMATION \
    "Sentinel Medical AI установлен!$\r$\n\
     (Sentinel Medical AI installed!)$\r$\n$\r$\n\
     ДАЛЕЕ / NEXT STEPS:$\r$\n$\r$\n\
     1. Запустите Sentinel из меню Пуск.$\r$\n\
        (Launch Sentinel from the Start Menu.)$\r$\n$\r$\n\
     2. При первом запуске нажмите 'Активировать лицензию'.$\r$\n\
        (On first run, click 'Activate License'.)$\r$\n$\r$\n\
     3. Отправьте отпечаток ПК на: shakhzodbatirjonov@gmail.com$\r$\n\
        (Send the Machine Fingerprint to that email.)$\r$\n$\r$\n\
     4. Сохраните полученный license.dat в:$\r$\n\
        (Save the license.dat file into:)$\r$\n\
        $APPDATA\Sentinel Medical AI\$\r$\n$\r$\n\
     До активации работает ДЕМО-режим (10 анализов/день).$\r$\n\
     (Until activated, runs in DEMO mode — 10 analyses/day.)"
!macroend

!macro customUnInstall
  MessageBox MB_YESNO|MB_ICONQUESTION \
    "Также удалить данные пациентов, лицензию и отчёты?$\r$\n\
     (Also remove patient data, license, and reports?)$\r$\n$\r$\n\
     Это действие необратимо. (This cannot be undone.)$\r$\n$\r$\n\
     Выберите НЕТ — данные сохранятся (безопасный выбор).$\r$\n\
     (Choose NO to keep your data — safe option.)" \
    IDNO +2
    RMDir /r "$APPDATA\Sentinel Medical AI"
!macroend
