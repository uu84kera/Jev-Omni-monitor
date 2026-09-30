# Jev-Omni-monitor: Anamoly Detection based on Jev-Omni
A general monitoring system to detect suspicious action through video.
Jev-Omni is a open-source image, audio, video and text classifier pretrained on large general data.

### Architecture
Camera
   ↓
3-frame temporal window
   ↓
Motion filter
   ↓
Jev-Omni
   ↓
suspicious   /     safe 
   ↓                 ↓
Strong VLM        discard
   ↓
What happened?
   ↓
Event classification
   ↓
Severity
   ↓
Policy engine
   ├── log
   ├── save
   ├── notify
   └── emergency




### Resources
## Data

## Server
