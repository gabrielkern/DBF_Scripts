# QGroundControl & ArduPilot Telemetry Troubleshooting Log

**Date:** October 12, 2025
**Hardware:** MATEKSYS H743-WLITE Flight Controller
**Firmware:** ArduPlane V4.6.2
**Issue:** BIN log files filled with zeros, EKF errors, arming failures

---

## Initial Problem Statement

### Primary Symptoms
1. **Log files corrupted**: .BIN files created with appropriate size but containing all zeros
2. **Duplicate firmware detection**: QGroundControl showing "Detected [2]: ArduPilot ChibiOS, ArduPilot ChibiOS"
3. **Compass issues**: All three compasses showing "Not installed"
4. **Missing parameters**: Error message showing missing params: `FS_OPTIONS`, `FS_GCS_TIMEOUT`, `FS_GCS_ENABLE`
5. **Arming failures**: Consistent PreArm check failures preventing arming

### Configuration
- Logging configured for all flight data to .BIN files on every startup
- LOG_DISARMED enabled (logging with or without arming)

---

## Root Cause Analysis

### Initial Hypothesis: Firmware Corruption
**Evidence:**
- Duplicate firmware detection
- Missing critical parameters
- Zero-filled log files despite correct file sizes

**Reasoning:**
The duplicate firmware listing combined with missing parameters strongly suggested corrupted or incomplete firmware installation, which would cascade through all subsystems including logging.

### Attempted Solution 1: Firmware Reflash via Betaflight Configurator

**Why Betaflight Configurator:**
- Native macOS support (avoiding Windows VM USB passthrough issues)
- Can flash ArduPilot firmware despite the name
- Well-tested for STM32 boards
- No Java dependency

**Process:**
1. Downloaded Betaflight Configurator (native Mac app)
2. Downloaded latest ArduPlane firmware: `arducopter_with_bl.hex` from firmware.ardupilot.org
3. Put H743-WLITE into DFU mode (hold BOOT button while connecting USB)
4. Flashed firmware using Betaflight Configurator

**Result:** Firmware successfully flashed, but issues persisted.

---

## Post-Reflash Symptoms

After firmware reflash, the following errors remained:

```
PreArm: Waiting for RC
PreArm: AHRS: waiting for home
PreArm: Compass not healthy
PreArm: AHRS: EKF3 Roll/Pitch inconsistent 14 deg
```

**Key Observation:** Sensors (pitch, roll, yaw) were now responding correctly in QGroundControl.

---

## Breakthrough: Calibration Discovery

### Issue: Uncalibrated IMU After Fresh Firmware Flash
Fresh firmware flash resets ALL calibrations to defaults, causing EKF3 (Extended Kalman Filter) to fail initialization.

**Solution Applied:**
1. Performed accelerometer calibration (6-position calibration in QGroundControl)
2. Performed compass calibration (though H743-WLITE has no onboard compass)

**Result:** EKF3 Roll/Pitch error CLEARED! New error state:

```
PreArm: Waiting for RC
PreArm: AHRS: waiting for home
PreArm: Compass not healthy
PreArm: AHRS: EKF3 not started
```

---

## Mystery Solved: FS_OPTIONS Parameter

### Discovery
**FS_OPTIONS does NOT exist in ArduPlane** - it's ArduCopter-only.

**Explanation:**
- ArduPlane uses different failsafe parameters: `FS_SHORT_ACTN`, `FS_LONG_ACTN`, `FS_GCS_ENABL`
- QGroundControl's error about missing FS_OPTIONS was a red herring
- The "duplicate firmware" message was likely QGC confusion about firmware type expectations

**Verdict:** This was NEVER actually a problem. ArduPlane 4.6.2 has correct parameters.

---

## Remaining Issues & Solutions

### Issue 1: EKF3 Not Started

**Root Cause:**
Extended Kalman Filter requires a yaw source (compass OR GPS) to initialize. The H743-WLITE has:
- ❌ No onboard compass
- ❌ No GPS module connected
- ❌ Compass not disabled in parameters

**Why This Matters:**
No EKF3 → No attitude estimation → Cannot arm → **No valid flight data to log** → Zero-filled BIN files

**Solution:**
Configure compass-less operation with GPS-based yaw:

```
COMPASS_ENABLE = 0          # Disables compass entirely
COMPASS_USE = 0             # Don't use compass 1
COMPASS_USE2 = 0            # Don't use compass 2
COMPASS_USE3 = 0            # Don't use compass 3
EK3_SRC1_YAW = 8            # Use GSF (GPS-based yaw estimation)
ARMING_CHECK = 16382        # Disable compass check, enable all others
```

For bench testing WITHOUT GPS (temporary, dangerous for flight):
```
EK3_SRC1_POSXY = 0          # No horizontal position source
EK3_SRC1_VELXY = 0          # No horizontal velocity source
EK3_SRC1_POSZ = 1           # Use barometer for altitude
EK3_SRC1_VELZ = 0           # No vertical velocity source
ARMING_CHECK = 0            # DISABLE ALL PREARM CHECKS (BENCH ONLY!)
```

⚠️ **WARNING:** Never fly with `ARMING_CHECK = 0`

---

### Issue 2: SD Card Format

**Hardware:** Lexar 32GB microSDHC UHS-I (U1, Class 10, V10, A1)

**Problem:**
Mac formats 32GB cards as **exFAT** by default. ArduPilot requires **FAT32** for reliable logging.

**Symptoms of Wrong Format:**
- Files created with correct size but zero data
- Intermittent write failures
- Log corruption

**Solution - Format to FAT32 on Mac:**
1. Backup any data from SD card
2. Open Disk Utility (Applications → Utilities)
3. Select SD card (physical device, not partition)
4. Click **Erase**
5. Format: **MS-DOS (FAT)** ← This is FAT32
6. Scheme: **Master Boot Record**
7. Click Erase

**ArduPilot SD Card Requirements:**
- ✅ FAT32: Fully supported, required
- ⚠️ exFAT: May work but causes issues
- ❌ FAT16: Not supported

---

## Known H743 Hardware Issues (Research Findings)

### 1. SD Card Slot Defects
Multiple reports of Matek H743 boards having defective SD card slots:
- "MatekH743 Slim V3 Sd card issue and FCB not initializing"
- "Possible bug on 4.1.3 with Matek H743-MINI - SD card related"
- FC won't initialize when SD card inserted, showing "failed data logging" errors

### 2. H7 Processor Memory Corruption
- H7 series processors can enter unrecoverable state
- Symptoms: Never exit bootloader, freeze during initialization
- Believed to be memory corruption from interrupted flash writes
- Requires complete reflash via STM32CubeProgrammer

### 3. IMU Inconsistency Issues
- "Accels Inconsistent MatekH743" reported by multiple users
- Issues persist even after temperature calibration
- May indicate hardware-level IMU defects on some boards

### 4. Soft-Bricking Tendency
- Matek H743 boards can "soft brick" after use
- Gets stuck in startup mode
- Requires reflash with STM32CubeProgrammer to recover

---

## Diagnostic Steps Performed

### Phase 1: Firmware Recovery
- ✅ Complete firmware erase and reflash using Betaflight Configurator
- ✅ Verified firmware version: ArduPlane V4.6.2
- ✅ Confirmed sensors responding (pitch/roll/yaw visible)

### Phase 2: Calibration
- ✅ Accelerometer calibration (6-position)
- ✅ Compass calibration attempt
- ✅ EKF3 Roll/Pitch error cleared after calibration

### Phase 3: Parameter Analysis
- ✅ Confirmed FS_OPTIONS doesn't exist in ArduPlane (not a bug)
- ✅ Identified missing compass as EKF3 blocker
- ✅ Identified SD card format as likely logging issue

---

## Recommended Next Steps

### Priority 1: Enable Logging (Bench Testing)
1. Format SD card to FAT32
2. Set `COMPASS_ENABLE = 0`
3. Set `LOG_DISARMED = 1`
4. Set `ARMING_CHECK = 0` (temporarily for bench testing)
5. Reboot, let run 60 seconds
6. Check BIN file contents in hex editor

### Priority 2: Add GPS Module (For Full Functionality)
- Connect GPS module via UART
- Configure serial port for GPS protocol
- EKF3 will use GPS-based yaw estimation (GSF)
- Provides home position for "waiting for home" error

### Priority 3: Configure RC Receiver
- Connect RC receiver to appropriate UART
- Configure `SERIALx_PROTOCOL` for receiver type
- Perform RC calibration in QGroundControl

### Priority 4: Restore Safety Checks
- Set `ARMING_CHECK` back to 1 or 16382 (all except compass)
- Verify all PreArm checks pass before flight
- Test arming and disarming

---

## Tools & Resources Used

### Software
- **QGroundControl**: Ground control station (native macOS)
- **Betaflight Configurator**: Firmware flashing (native macOS)
- **STM32CubeProgrammer**: Emergency recovery tool (available for macOS)

### Firmware Sources
- ArduPilot firmware: https://firmware.ardupilot.org/
- H743-WLITE specific: https://firmware.ardupilot.org/Plane/latest/MatekH743-WLITE/

### Documentation
- ArduPilot Plane docs: https://ardupilot.org/plane/
- H743-WLITE manual: http://www.mateksys.com/downloads/Manual/H743-WLITE_Manual.pdf
- ArduPilot Discourse forums: https://discuss.ardupilot.org/

### Diagnostic Tools
- **H2testw** (or F3 for Mac): SD card integrity testing
- **Disk Utility**: SD card formatting (macOS built-in)
- **MAVLink Inspector** (in QGC): Real-time sensor monitoring

---

## Key Learnings

### 1. Fresh Firmware = No Calibrations
**Always perform full sensor calibration suite after firmware flash:**
- Accelerometer (mandatory)
- Compass (if present)
- RC calibration (if receiver connected)
- ESC calibration (for motors)

### 2. ArduPlane ≠ ArduCopter
Different vehicle types use different parameter sets:
- FS_OPTIONS: ArduCopter only
- ArduPlane uses FS_SHORT_ACTN/FS_LONG_ACTN instead

### 3. H743-WLITE Has NO Compass
- This is by design, not a defect
- Requires either external compass OR compass-less configuration
- Compass-less mode requires GPS for yaw estimation (GSF)

### 4. Mac VM USB Issues
**Avoid using Windows VM for firmware flashing:**
- USB passthrough can hang/disconnect during flash
- Use native Mac tools instead:
  - Betaflight Configurator (primary)
  - STM32CubeProgrammer (emergency recovery)
  - QGroundControl (updates only, not initial flash)

### 5. SD Card Format Matters
- macOS defaults to exFAT for 32GB+ cards
- ArduPilot requires FAT32 for reliable logging
- Always use "MS-DOS (FAT)" format in Disk Utility

### 6. EKF3 Requirements
Extended Kalman Filter needs:
- Calibrated IMU (accelerometer/gyroscope)
- Yaw source (compass OR GPS with GSF)
- Without these, no attitude estimation = no arming = no logging

---

## Troubleshooting Decision Tree

```
BIN logs filled with zeros?
├─ Is firmware showing correctly in QGC?
│  ├─ NO → Reflash firmware via Betaflight Configurator
│  └─ YES → Continue
│
├─ Are sensors responding in QGC (pitch/roll/yaw)?
│  ├─ NO → Perform accelerometer calibration
│  └─ YES → Continue
│
├─ Does EKF3 error exist?
│  ├─ YES → Check compass/GPS configuration
│  │       ├─ No compass? Set COMPASS_ENABLE=0
│  │       └─ No GPS? Add GPS or disable EKF3 position sources
│  └─ NO → Continue
│
├─ Is SD card formatted as FAT32?
│  ├─ NO → Reformat to FAT32 (MS-DOS FAT in Disk Utility)
│  └─ YES → Continue
│
└─ Still zeros after all above?
   └─ Likely defective SD card slot (known H743 issue)
      └─ Try different SD card brand/model
      └─ Consider board RMA if multiple cards fail
```

---

## Status Summary

### ✅ Working
- Firmware: ArduPlane V4.6.2 properly installed
- IMU: Calibrated, sensors responding correctly
- Hardware: Board functional, USB communication stable

### ⚠️ Needs Configuration
- Compass: Disabled (not present on H743-WLITE)
- GPS: Not yet connected (needed for EKF3)
- RC: Not yet connected (needed for arming)
- SD Card: Needs FAT32 format verification

### ❓ To Be Tested
- Logging functionality after FAT32 format
- EKF3 initialization with proper compass/GPS config
- Arming sequence with all PreArm checks

---

## Contact & Support Resources

### ArduPilot Community
- Forums: https://discuss.ardupilot.org/
- Discord: ArduPilot Discord Server
- GitHub Issues: https://github.com/ArduPilot/ardupilot/issues

### Mateksys Support
- Website: https://www.mateksys.com/
- Product page: https://www.mateksys.com/?portfolio=h743-wlite

### QGroundControl
- Website: http://qgroundcontrol.com/
- GitHub: https://github.com/mavlink/qgroundcontrol

---

## Appendix: Parameter Reference

### Critical Parameters for H743-WLITE Setup

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `COMPASS_ENABLE` | 0 | Disable compass (not present on board) |
| `COMPASS_USE` | 0 | Don't use compass 1 |
| `COMPASS_USE2` | 0 | Don't use compass 2 |
| `COMPASS_USE3` | 0 | Don't use compass 3 |
| `EK3_SRC1_YAW` | 8 | Use GSF (GPS-based yaw) |
| `LOG_DISARMED` | 1 | Log even when disarmed |
| `ARMING_CHECK` | 16382 | All checks except compass |
| `AHRS_EKF_TYPE` | 3 | Use EKF3 (default) |

### Temporary Bench Testing (NO FLIGHT)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `ARMING_CHECK` | 0 | Disable all safety checks |
| `EK3_SRC1_POSXY` | 0 | No horizontal position |
| `EK3_SRC1_VELXY` | 0 | No horizontal velocity |

⚠️ **Restore normal values before any flight operations!**

---

*Document last updated: October 12, 2025*
