# Jev-Omni-monitor: Anamoly Detection based on Jev-Omni
A general monitoring system to detect suspicious action through video.
Jev-Omni is a open-source image, audio, video and text classifier pretrained on large general data.

### Architecture
Camera

   ↓
   
Frame Sampling 

   ↓
   
Lightweight filtering

   ↓
   
Jev-Omni Classifier

   ↓
   
suspicious   /     safe 

   ↓                 ↓
   
Strong VLM        Ignore

   ↓
   
semantic analysis

   ↓
   
Event + Severity

   ↓
   
Alert / store / action




### Resources
## Data

## Server
