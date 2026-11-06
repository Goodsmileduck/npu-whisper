# Windows GPU diagnostic — 2026-10-05

Read-only investigation of the real inference failure at **14:45:44**, with all
times in **America/Sao_Paulo (UTC−03:00)**. Collected approximately 14:55–14:58.
No configuration, drivers, event channels, permissions, or application state were
changed. No inference was started and no dump was opened or copied.

## Conclusion

The application failure coincides with an operating-system GPU watchdog event.
This provides evidence beyond the generic OpenCL `CL_OUT_OF_RESOURCES` message:
Windows created a WATCHDOG kernel dump in the same second, then recorded
`LiveKernelEvent 141`. Microsoft defines 0x141 as
`VIDEO_ENGINE_TIMEOUT_DETECTED` (a video engine did not respond in time; a live
dump code, not a real bug check/BSOD):
<https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x141---video-engine-timeout-detected>.

An earlier real failure at 14:19 was classified by Windows as
`LKD_0x141_Tdr:C_IMAGE_igdkmdn64.sys_MTL_OCL_PAGEFAULT`, explicitly naming the Intel
graphics kernel driver and an OpenCL page-fault classification. That supports
investigation of the Intel GPU/OpenCL/OpenVINO path. **It does not establish
whether the defect originates in the driver, a generated kernel, or their
interaction.** The 14:45 report had no populated fault bucket when inspected, so
the earlier page-fault classification must not be presented as a confirmed
classification of the later event.

## Event correlation

| Time | Source / event ID | Evidence |
| --- | --- | --- |
| 14:19:28 | Application log file | Earlier real `GPU CL_OUT_OF_RESOURCES`; old handler attempted to reload GPU. |
| 14:19:28.337 | Microsoft-Windows-WerKernel/Operational, 1001 | WATCHDOG requested a kernel dump; `STATUS_SUCCESS`. |
| 14:19:31.532 | Application / Application Error, 1000 | `python.exe` terminated; `ucrtbase.dll`, exception `0xc0000409`. Reliability records also contain this crash. |
| 14:19:33.551 | Microsoft-Windows-WerKernel/Operational, 1002 | `WATCHDOG-20261005-1419.dmp` submission completed. |
| 14:19:38.871 and 14:20:09.108 | Application / Windows Error Reporting, 1001 | Earlier `LiveKernelEvent`, `P1: 141`. |
| 14:25:57–14:26:03 | Application log file | New app session; GPU model loaded; warmup succeeded. |
| 14:26:21 and 14:29:48 | Application log file | Successful GPU inference on 2.9 s and 29.8 s audio. |
| 14:45:43 | Application log file | Recording completed, 2.9 s. Audio telemetry reported zero capture overflows. |
| 14:45:44 | Application log file | GPU inference raised `CL_OUT_OF_RESOURCES`; application disabled further dictation. |
| **14:45:44.895** | **Microsoft-Windows-WerKernel/Operational, 1001** | **WATCHDOG kernel dump creation completed successfully, in the same second as the app failure.** Event record 34. |
| 14:45:51.584 | Microsoft-Windows-WerKernel/Operational, 1001 | WATCHDOG mini dump creation completed successfully. Event record 35. |
| 14:45:51.721 | Microsoft-Windows-WerKernel/Operational, 1002 | `WATCHDOG-20261005-1445.dmp` submission completed successfully. Event record 36. |
| 14:45:58.184 | Application / Windows Error Reporting, 1001 | New `LiveKernelEvent`, `P1: 141`, referencing the **14:45** dump; fault bucket empty. |
| 14:45:59.962 | Application / Windows Error Reporting, 1001 | Classification of the **earlier 14:19 dump**, not the 14:45 dump: `LKD_0x141_Tdr:C_IMAGE_igdkmdn64.sys_MTL_OCL_PAGEFAULT`. |
| 14:46:28.308 | Application / Windows Error Reporting, 1001 | Follow-up of the 14:45 event: code 141; bucket still empty. |

The simulated application failures at 14:23–14:24 were excluded. The earlier
14:19 event is independently corroborated by Windows records and is a real event.
The Python crash after that earlier GPU reload does not, by itself, prove that
reloading caused termination, but the sequence is consistent with the reason
the current handler now prevents further GPU calls.

## Separate Intel Graphics Software installation problem

`Intel-IGS-Logs/Service` contains **958 error events (ID 14)** between 14:15:00 and
14:55:00: two messages per failed attempt, approximately every five seconds.
The 479 pairs report `Failed to start process` / `Process failed to start`, with
`Win32Exception` and `The system cannot find the file specified` for:

```text
C:\Program Files\WindowsApps\AppUp.IntelArcSoftware_26.32.2604.0_x64__8j3eq9eme6ctt\VFS\ProgramFilesX64\Intel\Intel Graphics Software\IntelGraphicsSoftware.Service.Graphics.exe
```

The installed AppX package is `AppUp.IntelArcSoftware 26.32.2604.0`.
`IntelGraphicsSoftwareService` itself reports `Running`, with automatic startup,
and points at the main `IntelGraphicsSoftware.Service.exe` in that package.
Thus the recorded failures concern its child graphics process. They predate and
continue after the inference failures. Repairing/checking this installation is
justified independently, but there is no evidence establishing it as the cause
of the OpenCL timeout.

## Negative findings and limitations

- No System/Application warning/error in the 14:25–14:55 window directly identified
  Display recovery, WHEA, Intel/NVIDIA reset, or resource exhaustion. WER records
  above have informational severity, illustrating why severity-only searches
  would miss the relevant evidence.
- No same-day resource-exhaustion event was found. Resource-Exhaustion-Detector
  and Resolver channels last wrote on October 2 and October 4 respectively.
  This does **not** prove GPU allocations had sufficient memory at failure time.
- Intel NPU Kmd and LevelZero operational channels are disabled. They were left
  unchanged. This incident occurred on the GPU inference path.
- Repeated `IntelTraceAgentService` ID 257 / SupportAssist power notifications
  occurred before and after the incident, but no causal connection is established.
- Windows ACLs denied reading the corresponding `Report.wer` files and listing
  `C:\Windows\LiveKernelReports\WATCHDOG`. No elevation or ACL modifications were
  attempted. Dump names and successful creation/submission are established by
  readable event-log records, not by examining the dump files.
- Reliability records showed the earlier Python crash but did not yet expose
  the later kernel event. The event logs do contain it.
- The event records cannot uniquely identify the faulty OpenCL operation or
  distinguish a driver defect from an OpenVINO kernel defect. A controlled
  reproduction and/or authorized dump analysis would be required for that.

## Sources examined

- `Get-WinEvent`: System, Application, Microsoft-Windows-WerKernel/Operational,
  Microsoft-Windows-WER-PayloadHealth/Operational, Intel-IGS-Logs/Service, and
  channel inventories for Intel, graphics, driver-framework, WER, and resource
  exhaustion logs.
- `Get-CimInstance Win32_ReliabilityRecords` (same-day records).
- `Get-AppxPackage` and `Get-CimInstance Win32_Service` (Intel graphics package).
- Local dictation log: only operational timing and failure evidence retained
  here; dictated content omitted.
