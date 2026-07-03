"""
VISION-PRO OMNI | Made By Frosts | V0.5.0
------------------------------------------
Ghost Build — Bone ESP · 3-D Face Mask · Emotion · Voice Control

KEYBOARD:
    H  - Toggle Full HUD          K  - ESP Style (Bone / Standard)
    Q  - Hands    W  - Body       E  - Face     R  - Objects
    M  - Mirror   J  - Joints     Y  - Air Draw
    F  - Cycle Face Mask  (Off / Wire / Solid / Neon / Skull / Blackout)
    X  - Cycle Pen Color  Z  - Zoom Pulse
    1/2/3 - Filters (Gray / Invert / Reset)
    UP/DOWN - Pen Thickness       C  - Clear All
    D  - Hold to Talk (push-to-talk voice command)
    ESC - Quit

VOICE COMMANDS (say these out loud, while holding D):
    "Jarvis hide face"    -> Blackout mask on
    "Jarvis show face"    -> Mask off
    "Jarvis wire"         -> Wire mask
    "Jarvis solid"        -> Solid mask
    "Jarvis neon"         -> Neon mask
    "Jarvis skull"        -> Skull mask
    "Jarvis hide hud"     -> Hide HUD
    "Jarvis show hud"     -> Show HUD
    "Jarvis mirror"       -> Toggle mirror
    "Jarvis draw"         -> Toggle draw mode
    "Jarvis clear"        -> Clear canvas
    "Jarvis body on"      -> Enable body tracking
    "Jarvis body off"     -> Disable body tracking
    "Jarvis objects on"   -> Enable object detection
    "Jarvis objects off"  -> Disable object detection
    "Jarvis invert"       -> Invert filter
    "Jarvis grayscale"    -> Grayscale filter
    "Jarvis reset filter" -> Reset filters
    "Jarvis zoom"         -> Zoom pulse
    "Jarvis bone"         -> Bone ESP style
    "Jarvis standard"     -> Standard body style

INTERACTION:
    2-Hand Pinch  -> Create Box
    1-Hand Pinch  -> Drag Box
    FIST 0.5s     -> Wipe Workspace
"""

import cv2, mediapipe as mp, time, os, math, threading, collections
import numpy as np, urllib.request
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Voice control -- graceful fallback if not installed
try:
    import speech_recognition as sr
    import sounddevice  # noqa: ensures sounddevice backend is available
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("[VOICE] pip install SpeechRecognition sounddevice")

# ── Colours ──────────────────────────────────────────────────────────────────
C_GOLD   = (255, 185,   0)
C_GREEN  = (  0, 255, 150)
C_PINK   = (255,   0, 200)
C_RED    = ( 50,  50, 255)
C_CYAN   = (  0, 210, 255)
C_WHITE  = (255, 255, 255)
C_GREY   = ( 80,  80,  80)
FONT     = cv2.FONT_HERSHEY_DUPLEX
PEN_COLS = [C_GREEN, C_GOLD, C_CYAN, C_PINK, C_WHITE]

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "hand":   ("hand_v040.task",
               "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"),
    "pose":   ("pose_v040.task",
               "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
    "face":   ("face_v040.task",
               "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
    "object": ("object_v040.tflite",
               "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"),
}

# ── Bone-ESP skeleton definition ──────────────────────────────────────────────
# MediaPipe pose landmark indices
# Head=0  Neck≈(11+12)/2  LShoulder=11  RShoulder=12
# LElbow=13 RElbow=14  LWrist=15 RWrist=16
# LHip=23  RHip=24  LKnee=25 RKnee=26  LAnkle=27 RAnkle=28
BONE_SEGS = [
    # Spine  (nose → mid-shoulder → mid-hip)
    "SPINE",
    # Arms
    (11, 13), (13, 15),   # L upper / lower arm
    (12, 14), (14, 16),   # R upper / lower arm
    # Legs
    (23, 25), (25, 27),   # L thigh / shin
    (24, 26), (26, 28),   # R thigh / shin
    # Shoulder bar & hip bar — drawn via "BARS"
    "BARS",
]

# ── Face-mask triangle tessellation (468-landmark mesh subset) ────────────────
# Full MediaPipe canonical face mesh triangulation
_RAW = """
0 267 269,0 269 270,0 270 409,0 409 291,0 291 375,0 375 321,0 321 405,
0 405 314,0 314 17,0 17 84,0 84 181,0 181 91,0 91 146,0 146 61,
10 338 297,10 297 332,10 332 284,10 284 251,10 251 389,10 389 356,
10 356 454,10 454 323,10 323 361,10 361 288,10 288 397,10 397 365,
10 365 379,10 379 378,10 378 400,10 400 377,10 377 152,10 152 148,
10 148 176,10 176 149,10 149 150,10 150 136,10 136 172,10 172 58,
10 58 132,10 132 93,10 93 234,10 234 127,10 127 162,10 162 21,
10 21 54,10 54 103,10 103 67,10 67 109,10 109 10,
21 162 54,54 162 103,103 162 67,67 162 109,109 162 10,
152 377 400,152 400 378,152 378 379,152 379 365,152 365 397,
152 397 288,152 288 361,152 361 323,152 323 454,152 454 356,
152 356 389,152 389 251,152 251 284,152 284 332,152 332 297,
152 297 338,338 10 297
"""
FACE_TRIS = []
for tok in _RAW.replace('\n','').split(','):
    parts = tok.strip().split()
    if len(parts)==3:
        try: FACE_TRIS.append((int(parts[0]),int(parts[1]),int(parts[2])))
        except: pass

SKULL_EDGES = [
    (10,151),(151,9),(9,8),(8,168),(168,6),(6,197),(197,195),(195,5),(5,4),(4,1),
    (33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),(173,33),
    (362,398),(398,384),(384,385),(385,386),(386,387),(387,388),(388,466),(466,362),
    (61,39),(39,37),(37,0),(0,267),(267,269),(269,270),(270,409),(409,291),
    (78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),(402,318),(318,324),(324,308),
]
MASK_MODES = ["OFF","WIRE","SOLID","NEON","SKULL","BLACKOUT"]

# Face oval landmark indices (MediaPipe) — used for the black-out convex hull
FACE_OVAL_IDX = [
    10,338,297,332,284,251,389,356,454,323,361,288,
    397,365,379,378,400,377,152,148,176,149,150,136,
    172,58,132,93,234,127,162,21,54,103,67,109
]


# ─────────────────────────────────────────────────────────────────────────────
class VisionProOmni:
    def __init__(self):
        # Capture at 16:9 native — use 640x360 for inference, display at 1280x720
        # This avoids the squish from 4:3 → 16:9 stretch
        self.CAP_W, self.CAP_H = 640, 360
        self.W,     self.H     = 1280, 720

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.CAP_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAP_H)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.detectors  = {}

        # Feature toggles
        self.enabled   = {"hand": True, "pose": False, "face": True,
                          "object": False, "eye": False}
        self.pose_mode  = 1        # 0=standard connections, 1=Bone ESP
        self.show_hud   = True
        self.show_joints= True
        self.draw_mode  = False
        self.mirror_mode= True

        # Mask
        self.mask_idx  = 0
        self.hue_t     = 0

        # Pen
        self.strokes, self.active_stroke = [], []
        self.pen_idx   = 0
        self.thickness = 5
        self.boxes, self.active_box = [], None
        self.drag_idx, self.drag_off = -1, (0,0)

        # Filters / effects
        self.f_gray, self.f_invert = False, False
        self.zoom_pulse, self.zoom_t = False, 0

        # State
        self.fps          = 0.0
        self.curr_gest    = "SEARCHING"
        self.gest_conf    = 0.0
        self.fist_t       = 0
        self.flash, self.flash_exp = "", 0

        # Emotion — rule-based from facial geometry (no blendshapes dependency)
        self.emotion      = "NEUTRAL"
        self._ebuf        = collections.deque(maxlen=15)

        # Async detection results (thread-safe)
        self._face_res    = None
        self._pose_res    = None
        self._obj_res     = None
        self._mp_img_lock = threading.Lock()
        self._mp_img_cur  = None
        self._ts_cur      = 0

        # Frame-skip counters + cached results (prevents flash on skipped frames)
        self._pose_skip = 0
        self._obj_skip  = 0
        self._last_pose = []
        self._last_objs = []

        # Voice command state
        self._voice_cmd   = ""           # last recognised command (thread writes, main reads)
        self._voice_lock  = threading.Lock()
        self._voice_label = ""           # shown in HUD
        self._voice_label_exp = 0
        self._mic_status  = "STARTING" if VOICE_AVAILABLE else "NO MIC LIB"

        # Push-to-talk (hold D) — no "Jarvis" wake word needed while held
        self._ptt_held       = False   # main thread sets this each frame from key state
        self._ptt_last_seen  = 0.0     # timestamp of last 'D' keydown event seen by main loop
        self._ptt_release_grace = 0.12 # seconds — OS key-repeat gap tolerance while held
        self._ptt_lock        = threading.Lock()
        self._ptt_recording   = False  # listener thread sets True/False as it starts/stops capture

        print("[FROSTS] V0.5.0 ready  |  F=Mask  K=BoneESP  Q/W/E/R=toggle  | VOICE ON" if VOICE_AVAILABLE else "[FROSTS] V0.5.0 ready  |  Voice disabled — install SpeechRecognition")

    # ── model download / cache ────────────────────────────────────────────────
    def _model_path(self, key):
        fname, url = MODELS[key]
        path = os.path.join(self.script_dir, fname)
        if not os.path.exists(path):
            print(f"[DL] {key} → {fname}")
            urllib.request.urlretrieve(url, path)
            print(f"[DL] {key} done")
        return path

    def _det(self, key):
        if key in self.detectors:
            return self.detectors[key]
        p = self._model_path(key)
        bo = BaseOptions(model_asset_path=p, delegate=BaseOptions.Delegate.CPU)
        try:
            if key == "hand":
                o = vision.HandLandmarkerOptions(bo,
                        running_mode=vision.RunningMode.VIDEO,
                        num_hands=2,
                        min_hand_detection_confidence=0.55,
                        min_hand_presence_confidence=0.55,
                        min_tracking_confidence=0.5)
                self.detectors[key] = vision.HandLandmarker.create_from_options(o)
            elif key == "pose":
                o = vision.PoseLandmarkerOptions(bo,
                        running_mode=vision.RunningMode.VIDEO,
                        min_pose_detection_confidence=0.55,
                        min_tracking_confidence=0.5)
                self.detectors[key] = vision.PoseLandmarker.create_from_options(o)
            elif key == "face":
                o = vision.FaceLandmarkerOptions(bo,
                        running_mode=vision.RunningMode.VIDEO,
                        num_faces=1,
                        min_face_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                        output_face_blendshapes=True)
                self.detectors[key] = vision.FaceLandmarker.create_from_options(o)
            elif key == "object":
                o = vision.ObjectDetectorOptions(bo,
                        running_mode=vision.RunningMode.VIDEO,
                        score_threshold=0.45)
                self.detectors[key] = vision.ObjectDetector.create_from_options(o)
            return self.detectors[key]
        except Exception as e:
            print(f"[DET ERR] {key}: {e}")
            return None

    # ── gesture solver ────────────────────────────────────────────────────────
    def _gesture(self, lms):
        tips=[8,12,16,20]; mcps=[5,9,13,17]
        ext=[lms[t].y < lms[m].y for t,m in zip(tips,mcps)]
        th=lms[4]
        th_ext = math.hypot(th.x-lms[5].x, th.y-lms[5].y) > 0.08
        pinch  = math.hypot(th.x-lms[8].x, th.y-lms[8].y)
        if not any(ext) and not th_ext:              return "FIST",     0.99
        if pinch < 0.04:                             return "PINCH",    0.98
        if ext[0] and not any(ext[1:]):              return "POINTING", 0.95
        if ext[0] and ext[1] and not ext[2] and not ext[3]: return "PEACE",0.96
        if ext[0] and not ext[1] and not ext[2] and ext[3]: return "ROCK", 0.95
        if ext[1] and not ext[0] and not ext[2] and not ext[3]: return "MID",0.99
        if all(ext) and th_ext:                      return "OPEN",     0.94
        return "NEUTRAL", 0.2

    # ── emotion from face geometry (no blendshapes API needed) ───────────────
    def _emotion_geo(self, lms):
        # mouth openness: lip distance / face height
        top_lip  = lms[13]
        bot_lip  = lms[14]
        chin     = lms[152]
        nose_tip = lms[1]
        face_h   = abs(chin.y - lms[10].y) + 1e-6

        mouth_open = abs(top_lip.y - bot_lip.y) / face_h

        # brow raise: distance brow to eye
        l_brow = lms[65]; l_eye = lms[159]
        r_brow = lms[295]; r_eye = lms[386]
        brow_raise = ((abs(l_brow.y-l_eye.y) + abs(r_brow.y-r_eye.y))/2) / face_h

        # mouth corners vs centre: smile
        l_corner = lms[61]; r_corner = lms[291]
        mouth_mid_y = (top_lip.y + bot_lip.y) / 2
        smile = ((mouth_mid_y - l_corner.y) + (mouth_mid_y - r_corner.y)) / 2 / face_h

        # brow furrow: inner brow distance
        l_inner = lms[55]; r_inner = lms[285]
        furrow = abs(l_inner.x - r_inner.x) / (abs(lms[234].x - lms[454].x) + 1e-6)

        if mouth_open > 0.06:   e = "SURPRISED"
        elif smile > 0.012:     e = "HAPPY"
        elif furrow < 0.28:     e = "ANGRY"
        elif brow_raise > 0.07: e = "WORRIED"
        else:                   e = "NEUTRAL"
        self._ebuf.append(e)
        return collections.Counter(self._ebuf).most_common(1)[0][0]

    # ── glass panel ──────────────────────────────────────────────────────────
    def _glass(self, img, x, y, w, h, a=0.6, col=(100,100,100)):
        ov = img.copy()
        cv2.rectangle(ov,(x,y),(x+w,y+h),(10,10,10),-1)
        cv2.addWeighted(ov,a,img,1-a,0,img)
        cv2.rectangle(img,(x,y),(x+w,y+h),col,1,cv2.LINE_AA)

    # ── Bone ESP body ─────────────────────────────────────────────────────────
    def _draw_bone_esp(self, frame, lms, col=C_GREEN):
        W, H = self.W, self.H
        def pt(i): return (int(lms[i].x*W), int(lms[i].y*H))
        def vis(i): return lms[i].visibility > 0.4

        # Mid-shoulder & mid-hip virtual points
        lsh, rsh = pt(11), pt(12)
        lhp, rhp = pt(23), pt(24)
        mid_sh = ((lsh[0]+rsh[0])//2, (lsh[1]+rsh[1])//2)
        mid_hp = ((lhp[0]+rhp[0])//2, (lhp[1]+rhp[1])//2)
        nose   = pt(0)

        thick = 2
        # Head circle
        head_r = max(12, abs(mid_sh[1] - nose[1])//2)
        head_c = (nose[0], nose[1])
        cv2.circle(frame, head_c, head_r, col, thick, cv2.LINE_AA)

        # Spine line
        cv2.line(frame, mid_sh, mid_hp, col, thick, cv2.LINE_AA)
        # Neck
        cv2.line(frame, (nose[0], nose[1]+head_r), mid_sh, col, thick, cv2.LINE_AA)

        # Shoulder bar
        cv2.line(frame, lsh, rsh, col, thick, cv2.LINE_AA)
        # Hip bar
        cv2.line(frame, lhp, rhp, col, thick, cv2.LINE_AA)

        # Arms
        for side in [(11,13,15),(12,14,16)]:
            sh,el,wr = side
            if vis(sh) and vis(el): cv2.line(frame, pt(sh), pt(el), col, thick, cv2.LINE_AA)
            if vis(el) and vis(wr): cv2.line(frame, pt(el), pt(wr), col, thick, cv2.LINE_AA)
            # Fist dot at wrist
            if vis(wr):
                cv2.circle(frame, pt(wr), 4, col, -1, cv2.LINE_AA)

        # Legs
        for side in [(23,25,27),(24,26,28)]:
            hp,kn,an = side
            if vis(hp) and vis(kn): cv2.line(frame, pt(hp), pt(kn), col, thick, cv2.LINE_AA)
            if vis(kn) and vis(an): cv2.line(frame, pt(kn), pt(an), col, thick, cv2.LINE_AA)
            if vis(an):
                cv2.circle(frame, pt(an), 4, col, -1, cv2.LINE_AA)

        # ESP bounding box corners
        xs = [int(l.x*W) for l in lms if l.visibility>0.4]
        ys = [int(l.y*H) for l in lms if l.visibility>0.4]
        if xs and ys:
            x1,y1,x2,y2 = min(xs)-20,min(ys)-20,max(xs)+20,max(ys)+20
            L=22
            for (px,py,sx,sy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(px,py),(px+L*sx,py),col,2,cv2.LINE_AA)
                cv2.line(frame,(px,py),(px,py+L*sy),col,2,cv2.LINE_AA)
            # Health bar
            cv2.rectangle(frame,(x1,y2+10),(x2,y2+16),(40,40,40),-1)
            cv2.rectangle(frame,(x1,y2+10),(x1+int((x2-x1)*0.92),y2+16),col,-1)
            cv2.putText(frame,"TARGET",(x1,y1-6),FONT,0.38,col,1)

    # ── Standard skeleton ─────────────────────────────────────────────────────
    def _draw_skeleton(self, frame, lms, conns, col):
        W,H = self.W,self.H
        for c in conns:
            try:
                p1=(int(lms[c.start].x*W),int(lms[c.start].y*H))
                p2=(int(lms[c.end  ].x*W),int(lms[c.end  ].y*H))
                cv2.line(frame,p1,p2,col,2,cv2.LINE_AA)
            except: pass
        if self.show_joints:
            for lm in lms:
                cv2.circle(frame,(int(lm.x*W),int(lm.y*H)),3,C_WHITE,-1)

    # ── 3-D face mask ─────────────────────────────────────────────────────────
    def _draw_mask(self, frame, lms):
        mode = MASK_MODES[self.mask_idx]
        if mode == "OFF": return

        W,H = self.W,self.H
        # Build 2-D pixel array and depth array from landmarks
        px = np.array([(lm.x*W, lm.y*H) for lm in lms], dtype=np.float32)
        dz = np.array([lm.z for lm in lms], dtype=np.float32)

        # ── BLACKOUT: solid black fill over face oval, no alpha ───────────────
        if mode == "BLACKOUT":
            oval_pts = px[FACE_OVAL_IDX].astype(np.int32)
            hull = cv2.convexHull(oval_pts)
            cv2.fillConvexPoly(frame, hull, (0, 0, 0), cv2.LINE_AA)
            # Subtle cyan border so it doesn't look like a glitch
            cv2.polylines(frame, [hull], True, C_CYAN, 1, cv2.LINE_AA)
            return
        # Normalise depth
        z0,z1 = dz.min(), dz.max()
        zr = max(z1-z0, 1e-5)

        self.hue_t = (self.hue_t + 1) % 180

        if mode in ("SOLID","NEON","SKULL"):
            overlay = frame.copy()
            for tri in FACE_TRIS:
                try:
                    a,b,c = tri
                    verts = px[[a,b,c]].astype(np.int32)
                    # Clip to frame bounds
                    if (verts<0).any() or (verts[:,0]>=W).any() or (verts[:,1]>=H).any():
                        continue
                    depth = (dz[a]+dz[b]+dz[c])/3
                    norm  = 1.0 - (depth-z0)/zr   # 1=close,0=far
                    if mode == "SOLID":
                        v = int(60 + 150*norm)
                        cv2.fillPoly(overlay,[verts],(0,v,int(v*0.6)))
                    elif mode == "NEON":
                        hue = int((self.hue_t + norm*50) % 180)
                        hsv = np.uint8([[[hue,230,int(160+95*norm)]]])
                        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)[0][0]
                        cv2.fillPoly(overlay,[verts],(int(bgr[0]),int(bgr[1]),int(bgr[2])))
                    elif mode == "SKULL":
                        v = int(40+120*norm)
                        cv2.fillPoly(overlay,[verts],(v,v,v))
                except: pass
            cv2.addWeighted(overlay, 0.52, frame, 0.48, 0, frame)

        if mode in ("WIRE","SKULL"):
            ecol = C_CYAN if mode=="WIRE" else (30,80,30)
            for tri in FACE_TRIS:
                try:
                    verts = px[list(tri)].astype(np.int32)
                    cv2.polylines(frame,[verts],True,ecol,1,cv2.LINE_AA)
                except: pass

        if mode == "SKULL":
            for (a,b) in SKULL_EDGES:
                try:
                    p1=(int(px[a,0]),int(px[a,1]))
                    p2=(int(px[b,0]),int(px[b,1]))
                    cv2.line(frame,p1,p2,(0,200,0),1,cv2.LINE_AA)
                except: pass

        if mode == "WIRE":
            for i,(x,y) in enumerate(px):
                norm = 1-(dz[i]-z0)/zr
                r = max(1,int(2*norm))
                cv2.circle(frame,(int(x),int(y)),r,C_CYAN,-1,cv2.LINE_AA)

    # ── HUD ──────────────────────────────────────────────────────────────────
    def _hud(self, frame):
        if not self.show_hud: return
        W,H = self.W,self.H

        # Top bar
        self._glass(frame,0,0,W,44,0.72,C_GOLD)
        cv2.putText(frame,"FROSTS OMNI | V0.4.0",(18,30),FONT,0.6,C_WHITE,1)
        fps_col = C_GREEN if self.fps>25 else (C_GOLD if self.fps>15 else C_RED)
        cv2.putText(frame,f"FPS {int(self.fps)}",(W-105,30),FONT,0.55,fps_col,1)

        # Left sidebar
        self._glass(frame,14,56,210,310,0.52,C_GOLD)
        cv2.putText(frame,"SYSTEMS",(28,82),FONT,0.45,C_GOLD,1)
        rows=[("HAND","hand"),("BODY","pose"),("FACE","face"),("OBJ","object")]
        for i,(label,key) in enumerate(rows):
            on = self.enabled[key]
            col= C_GREEN if on else C_GREY
            cv2.putText(frame,f"[{'ON ' if on else 'OFF'}] {label}",(28,108+i*36),FONT,0.38,col,1)
        cv2.putText(frame,f"POSE: {'BONE' if self.pose_mode else 'STD'}",(28,108+4*36),FONT,0.38,C_CYAN,1)
        cv2.putText(frame,f"MASK: {MASK_MODES[self.mask_idx]}",(28,108+5*36),FONT,0.38,C_CYAN,1)
        pc=PEN_COLS[self.pen_idx]
        cv2.rectangle(frame,(28,108+6*36),(78,108+6*36+14),pc,-1)
        cv2.putText(frame,"PEN",(84,108+6*36+12),FONT,0.36,pc,1)

        # Right gesture box
        g_col = C_RED if self.curr_gest in ("FIST","MID") else C_WHITE
        self._glass(frame,W-250,56,234,148,0.58,g_col)
        cv2.putText(frame,f"POSE: {self.curr_gest}",(W-242,90),FONT,0.5,g_col,1)
        cv2.rectangle(frame,(W-242,106),(W-26,116),(40,40,40),-1)
        cv2.rectangle(frame,(W-242,106),(W-242+int(216*self.gest_conf),116),C_GREEN,-1)
        e_col = {"HAPPY":C_GOLD,"ANGRY":C_RED,"SURPRISED":C_CYAN,
                 "WORRIED":(180,100,255)}.get(self.emotion,C_WHITE)
        cv2.putText(frame,f"MOOD: {self.emotion}",(W-242,140),FONT,0.45,e_col,1)
        cv2.putText(frame,f"DRAW: {'ON' if self.draw_mode else 'OFF'}",(W-242,170),FONT,0.38,C_WHITE,1)
        cv2.putText(frame,f"MIRR: {'ON' if self.mirror_mode else 'OFF'}",(W-242,196),FONT,0.38,C_WHITE,1)

        # Flash
        if self.flash and time.time()<self.flash_exp:
            (tw,th),_ = cv2.getTextSize(self.flash,FONT,1.0,2)
            cv2.putText(frame,self.flash,((W-tw)//2,H//2),FONT,1.0,C_RED,2,cv2.LINE_AA)

        # Voice mic status bar (bottom left)
        if VOICE_AVAILABLE:
            ptt_active = self._ptt_is_held()
            if ptt_active:
                mic_col = C_RED          # actively recording — hard to miss
            elif self._mic_status == "LISTENING":
                mic_col = C_GREEN
            else:
                mic_col = C_GOLD
            self._glass(frame,14,H-38,236,28,0.6,mic_col)
            label = "MIC: ● REC (D)" if ptt_active else f"MIC: {self._mic_status}"
            cv2.putText(frame,label,(22,H-18),FONT,0.38,mic_col,1)
        # Last voice label (bottom centre)
        if self._voice_label and time.time()<self._voice_label_exp:
            lbl = f">> {self._voice_label}"
            (lw,_),_ = cv2.getTextSize(lbl,FONT,0.55,1)
            self._glass(frame,(W-lw)//2-8,H-50,lw+16,32,0.7,C_CYAN)
            cv2.putText(frame,lbl,((W-lw)//2,H-28),FONT,0.55,C_CYAN,1)

    # ── Push-to-talk key-state helper ─────────────────────────────────────────
    def _mark_ptt_seen(self):
        """Call every time the main loop sees the 'D' keycode — refreshes the
        held-down timestamp. OpenCV doesn't give us key-up events, only repeat
        events while a key is physically down, so 'still held' = 'seen recently'."""
        self._ptt_last_seen = time.time()

    def _ptt_is_held(self):
        return (time.time() - self._ptt_last_seen) < self._ptt_release_grace

    # ── zoom pulse ────────────────────────────────────────────────────────────
    def _zoom(self, frame):
        t=time.time()-self.zoom_t
        if t>0.8: self.zoom_pulse=False; return frame
        s=1+0.05*math.sin(t*math.pi*7)*(1-t/0.8)
        cx,cy=self.W//2,self.H//2
        M=cv2.getRotationMatrix2D((cx,cy),0,s)
        return cv2.warpAffine(frame,M,(self.W,self.H))


    # ── Voice command listener (runs in background thread) ────────────────────
    def _voice_listen(self):
        if not VOICE_AVAILABLE:
            return
        try:
            import sounddevice as sd
            import queue as _q
            import numpy as _np

            RATE       = 16000
            CHUNK      = 512          # small blocks for low latency
            SILENCE_DB = 500          # RMS threshold — below = silence (auto/VAD mode)
            SPEECH_SEC = 3.5          # max recording window in seconds (auto/VAD mode)
            SILENCE_SEC= 0.7          # silence after speech triggers recognition (auto mode)
            PTT_MAX_SEC= 12.0         # safety cap on a single push-to-talk hold
            MAX_BYTES  = int(RATE * SPEECH_SEC * 2)
            PTT_MAX_BYTES = int(RATE * PTT_MAX_SEC * 2)

            recognizer = sr.Recognizer()
            recognizer.energy_threshold        = 400
            recognizer.dynamic_energy_threshold= False
            recognizer.pause_threshold         = 0.5

            audio_q = _q.Queue()
            def sd_cb(indata, frames, t, status):
                audio_q.put(bytes(indata))

            def recognize_and_dispatch(raw_bytes, require_jarvis):
                """Send buffered audio to Google, push result into the
                rolling command buffer. require_jarvis=False is the
                push-to-talk path — recognised text is accepted directly,
                no wake word needed, by faking one onto the front so the
                existing _handle_voice() matching logic Just Works."""
                if not raw_bytes:
                    return
                audio = sr.AudioData(raw_bytes, RATE, 2)
                try:
                    text = recognizer.recognize_google(audio).lower()
                    with self._voice_lock:
                        if require_jarvis:
                            prev = self._voice_cmd
                            self._voice_cmd = (prev + " " + text).strip()[-120:]
                        else:
                            self._voice_cmd = ("jarvis " + text)[-120:]
                    tag = "PTT" if not require_jarvis else "AUTO"
                    print(f"[VOICE·{tag}] heard: {text}")
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    if "Connection" not in str(e):
                        print(f"[VOICE] err: {e}")

            self._mic_status = "PTT: HOLD D"
            with sd.RawInputStream(samplerate=RATE, blocksize=CHUNK,
                                   dtype="int16", channels=1, callback=sd_cb):
                buf            = b""        # auto/VAD buffer
                ptt_buf        = b""        # push-to-talk buffer
                recording      = False      # auto/VAD recording flag
                ptt_was_held   = False      # edge-detect for PTT press/release
                silence_frames = 0
                SILENCE_CHUNKS = int(SILENCE_SEC * RATE / CHUNK)

                while True:
                    chunk = audio_q.get()
                    ptt_held = self._ptt_is_held()

                    # ── Edge detection: PTT press / release ─────────────────
                    if ptt_held and not ptt_was_held:
                        # D just pressed — start a clean capture; drop
                        # anything the VAD path had buffered so the two
                        # modes never bleed into each other.
                        ptt_buf = b""
                        buf = b""; recording = False; silence_frames = 0
                        with self._ptt_lock:
                            self._ptt_recording = True
                        self._mic_status = "PTT: RECORDING"
                    if not ptt_held and ptt_was_held:
                        # D just released — recognize whatever we captured.
                        with self._ptt_lock:
                            self._ptt_recording = False
                        self._mic_status = "PTT: HOLD D"
                        captured, ptt_buf = ptt_buf, b""
                        recognize_and_dispatch(captured, require_jarvis=False)
                    ptt_was_held = ptt_held

                    if ptt_held:
                        # While D is held: raw capture, no VAD, no silence
                        # cutoff, no wake word required.
                        ptt_buf += chunk
                        if len(ptt_buf) >= PTT_MAX_BYTES:
                            captured, ptt_buf = ptt_buf, b""
                            recognize_and_dispatch(captured, require_jarvis=False)
                            self._mic_status = "PTT: RECORDING"
                        continue  # skip the VAD/"Jarvis" path entirely while held

                    # ── Existing always-on VAD + wake-word path (D not held) ──
                    arr = _np.frombuffer(chunk, dtype=_np.int16)
                    rms = float(_np.sqrt(_np.mean(arr.astype(_np.float32)**2)))

                    if rms > SILENCE_DB:
                        recording = True
                        silence_frames = 0
                        buf += chunk
                    elif recording:
                        buf += chunk
                        silence_frames += 1
                        if silence_frames >= SILENCE_CHUNKS or len(buf) >= MAX_BYTES:
                            captured, buf = buf, b""
                            recording = False; silence_frames = 0
                            recognize_and_dispatch(captured, require_jarvis=True)
        except Exception as e:
            self._mic_status = f"ERR"
            print(f"[VOICE] listener failed: {e}")

    # ── Voice command dispatcher ──────────────────────────────────────────────
    def _handle_voice(self):
        with self._voice_lock:
            cmd = self._voice_cmd
            # Clear only if command was acted on (we clear below after match)
        if not cmd:
            return

        # ── Fuzzy keyword matcher ──────────────────────────────────────────
        import difflib
        def near(word, threshold=0.72):
            """True if any token in cmd is close enough to word."""
            tokens = cmd.split()
            for tok in tokens:
                if difflib.SequenceMatcher(None, tok, word).ratio() >= threshold:
                    return True
            return False

        def phrase(words, threshold=0.72):
            """True if all words in the phrase have a fuzzy match somewhere in cmd."""
            return all(near(w, threshold) for w in words.split())

        # Must contain 'jarvis' (or sound-alike) somewhere in the rolling buffer
        if not near("jarvis", 0.70):
            return

        # Clear the buffer now that we have a valid Jarvis command window
        with self._voice_lock:
            self._voice_cmd = ""

        def flash(msg):
            self._voice_label     = msg
            self._voice_label_exp = time.time() + 2.5
            self.flash     = msg
            self.flash_exp = time.time() + 1.8

        # ── Face / mask ───────────────────────────────────────────────────
        # "jarvis face" → toggle blackout (fully on/off, not just the mask)
        if phrase("face") and not phrase("hide") and not phrase("show")                 and not any(phrase(m) for m in ["wire","solid","neon","skull","blackout"]):
            if self.mask_idx == MASK_MODES.index("BLACKOUT"):
                self.mask_idx = MASK_MODES.index("OFF")
                self.enabled["face"] = False; flash("FACE OFF")
            else:
                self.mask_idx = MASK_MODES.index("BLACKOUT")
                self.enabled["face"] = True; flash("FACE HIDDEN")
        elif (phrase("hide face") or phrase("hide") and phrase("face"))                 or (near("hyde") and phrase("face"))                 or (near("pie") and phrase("face"))                 or (near("side") and phrase("face"))                 or (near("hunt") and phrase("face")):
            self.mask_idx = MASK_MODES.index("BLACKOUT")
            self.enabled["face"] = True; flash("FACE HIDDEN")
        elif phrase("show face") or (phrase("show") and phrase("face")):
            self.mask_idx = MASK_MODES.index("OFF"); flash("FACE VISIBLE")
        elif near("wire"):
            self.mask_idx = MASK_MODES.index("WIRE")
            self.enabled["face"] = True; flash("MASK: WIRE")
        elif near("solid"):
            self.mask_idx = MASK_MODES.index("SOLID")
            self.enabled["face"] = True; flash("MASK: SOLID")
        elif near("neon"):
            self.mask_idx = MASK_MODES.index("NEON")
            self.enabled["face"] = True; flash("MASK: NEON")
        elif near("skull"):
            self.mask_idx = MASK_MODES.index("SKULL")
            self.enabled["face"] = True; flash("MASK: SKULL")
        elif near("blackout") or phrase("black out"):
            self.mask_idx = MASK_MODES.index("BLACKOUT")
            self.enabled["face"] = True; flash("MASK: BLACKOUT")
        # ── HUD ───────────────────────────────────────────────────────────
        elif phrase("hide hud") or (near("hide") and near("hud")):
            self.show_hud = False; flash("HUD OFF")
        elif phrase("show hud") or (near("show") and near("hud")):
            self.show_hud = True;  flash("HUD ON")
        # ── Mirror ────────────────────────────────────────────────────────
        elif near("mirror"):
            self.mirror_mode ^= 1
            flash("MIRROR " + ("ON" if self.mirror_mode else "OFF"))
        # ── Drawing ───────────────────────────────────────────────────────
        elif near("draw"):
            self.draw_mode ^= 1
            flash("DRAW " + ("ON" if self.draw_mode else "OFF"))
        elif near("clear") or near("wipe"):
            self.strokes=[]; self.boxes=[]; flash("CANVAS CLEARED")
        # ── Body ──────────────────────────────────────────────────────────
        elif near("body") and near("on"):
            self.enabled["pose"] = True;  flash("BODY ON")
        elif near("body") and near("off"):
            self.enabled["pose"] = False; flash("BODY OFF")
        elif near("bone"):
            self.pose_mode = 1; flash("BONE ESP")
        elif near("standard"):
            self.pose_mode = 0; flash("STANDARD BODY")
        # ── Objects ───────────────────────────────────────────────────────
        elif near("object") and near("on"):
            self.enabled["object"] = True;  flash("OBJECTS ON")
        elif near("object") and near("off"):
            self.enabled["object"] = False; flash("OBJECTS OFF")
        # ── Hands ─────────────────────────────────────────────────────────
        elif near("hand") and near("on"):
            self.enabled["hand"] = True;  flash("HANDS ON")
        elif near("hand") and near("off"):
            self.enabled["hand"] = False; flash("HANDS OFF")
        # ── Filters ───────────────────────────────────────────────────────
        elif near("invert"):
            self.f_invert ^= 1
            flash("INVERT " + ("ON" if self.f_invert else "OFF"))
        elif near("grayscale") or near("greyscale") or near("gray") or near("grey"):
            self.f_gray ^= 1
            flash("GRAY " + ("ON" if self.f_gray else "OFF"))
        elif near("reset") and near("filter"):
            self.f_gray = self.f_invert = False; flash("FILTERS RESET")
        # ── Effects ───────────────────────────────────────────────────────
        elif near("zoom"):
            self.zoom_pulse=True; self.zoom_t=time.time(); flash("ZOOM")
        # ── Joints ────────────────────────────────────────────────────────
        elif near("joint") and near("on"):
            self.show_joints = True;  flash("JOINTS ON")
        elif near("joint") and near("off"):
            self.show_joints = False; flash("JOINTS OFF")

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        prev=time.time(); t0_ms=int(time.time()*1000)

        # Start voice listener in background
        self._voice_stop = None
        if VOICE_AVAILABLE:
            vt = threading.Thread(target=self._voice_listen, daemon=True)
            vt.start()

        while True:
            ok,raw=self.cap.read()
            if not ok: break

            # Upscale from capture res to display res (fast nearest-neighbour)
            frame=cv2.resize(raw,(self.W,self.H),interpolation=cv2.INTER_LINEAR)
            if self.mirror_mode: frame=cv2.flip(frame,1)
            if self.f_gray: frame=cv2.cvtColor(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
            if self.f_invert: frame=cv2.bitwise_not(frame)
            if self.zoom_pulse: frame=self._zoom(frame)

            # Build MP image from the SMALL capture frame (saves CPU)
            small=cv2.flip(raw,1) if self.mirror_mode else raw
            if self.f_gray: small=cv2.cvtColor(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
            rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
            mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            ts=int(time.time()*1000)-t0_ms

            pinch_pts,frame_gest,tip=[],  "NEUTRAL", None

            # ── HANDS (every frame — fast) ───────────────────────────────────
            hd=self._det("hand")
            if self.enabled["hand"] and hd:
                hr=hd.detect_for_video(mp_img,ts)
                if hr.hand_landmarks:
                    for i,lms in enumerate(hr.hand_landmarks):
                        g,cf=self._gesture(lms)
                        if i==0: self.curr_gest,self.gest_conf,frame_gest=g,cf,g
                        # Scale landmarks to display size
                        def pt(lm): return (int(lm.x*self.W),int(lm.y*self.H))
                        tx,ty=pt(lms[8])
                        if g=="PINCH":    pinch_pts.append((tx,ty))
                        if g=="POINTING": tip=(tx,ty)

                        # Fingertip glow
                        for fi in [4,8,12,16,20]:
                            cv2.circle(frame,pt(lms[fi]),7,C_GOLD,-1,cv2.LINE_AA)
                            cv2.circle(frame,pt(lms[fi]),12,C_GOLD,1,cv2.LINE_AA)

                        self._draw_skeleton(frame,lms,
                            vision.HandLandmarksConnections.HAND_CONNECTIONS,C_GOLD)

                        # Label
                        hn=hr.handedness[i][0].display_name if hr.handedness else ""
                        wx,wy=pt(lms[0]); wy+=20
                        cv2.putText(frame,f"{hn} {g}",(wx-30,wy),FONT,0.35,C_GOLD,1)

                        # Fist wipe timer
                        if frame_gest=="FIST" and i==0:
                            if not self.fist_t: self.fist_t=time.time()
                            prog=min((time.time()-self.fist_t)/0.5,1.0)
                            ctr=pt(lms[0])
                            cv2.ellipse(frame,ctr,(42,42),-90,0,int(prog*360),C_RED,3,cv2.LINE_AA)
                            if prog>=1:
                                self.boxes,self.strokes,self.fist_t=[],[],0
                                self.flash,self.flash_exp="WIPED",time.time()+1
                        elif i==0: self.fist_t=0

            # ── POSE (every 2nd frame, cache result to avoid flash) ──────────
            self._pose_skip=(self._pose_skip+1)%2
            pd=self._det("pose")
            if self.enabled["pose"] and pd:
                if self._pose_skip==0:
                    pr=pd.detect_for_video(mp_img,ts)
                    self._last_pose=pr.pose_landmarks if pr and pr.pose_landmarks else []
                for plms in (self._last_pose or []):
                    if self.pose_mode==1:
                        self._draw_bone_esp(frame,plms,C_GREEN)
                    else:
                        self._draw_skeleton(frame,plms,
                            vision.PoseLandmarksConnections.POSE_LANDMARKS,C_GREEN)
            else:
                self._last_pose=[]

            # ── FACE (every frame — fast on lite) ────────────────────────────
            fd=self._det("face")
            if self.enabled["face"] and fd:
                fr=fd.detect_for_video(mp_img,ts)
                if fr and fr.face_landmarks:
                    for fi2,flms in enumerate(fr.face_landmarks):
                        self._draw_mask(frame,flms)
                        # Emotion (geometry-based — no blendshapes needed)
                        self.emotion=self._emotion_geo(flms)
                        # Subtle contour
                        if MASK_MODES[self.mask_idx]=="OFF":
                            self._draw_skeleton(frame,flms,
                                vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                                (60,60,60))

            # ── OBJECTS (every 3rd frame, cached) ────────────────────────────
            self._obj_skip=(self._obj_skip+1)%3
            od=self._det("object")
            if self.enabled["object"] and od:
                sx=self.W/self.CAP_W; sy=self.H/self.CAP_H
                if self._obj_skip==0:
                    ore=od.detect_for_video(mp_img,ts)
                    self._last_objs=ore.detections if ore else []
                for d in (self._last_objs or []):
                    b=d.bounding_box; cat=d.categories[0]
                    x1o=int(b.origin_x*sx); y1o=int(b.origin_y*sy)
                    x2o=int((b.origin_x+b.width)*sx); y2o=int((b.origin_y+b.height)*sy)
                    cv2.rectangle(frame,(x1o,y1o),(x2o,y2o),C_GREEN,2,cv2.LINE_AA)
                    lbl=f"{cat.category_name} {cat.score:.0%}"
                    self._glass(frame,x1o,y1o-20,len(lbl)*9,18,0.7,C_GREEN)
                    cv2.putText(frame,lbl,(x1o+3,y1o-7),FONT,0.36,C_GREEN,1)
            else:
                self._last_objs=[]

            # ── BOX interaction ───────────────────────────────────────────────
            if len(pinch_pts)==2:
                cv2.rectangle(frame,pinch_pts[0],pinch_pts[1],C_WHITE,2)
                self.active_box=[pinch_pts[0],pinch_pts[1],C_GOLD]
            elif len(pinch_pts)==1 and not self.draw_mode:
                pt2=pinch_pts[0]
                if self.drag_idx==-1:
                    for idx,(b1,b2,c) in enumerate(self.boxes):
                        if min(b1[0],b2[0])<pt2[0]<max(b1[0],b2[0]) and \
                           min(b1[1],b2[1])<pt2[1]<max(b1[1],b2[1]):
                            self.drag_idx=idx; self.drag_off=(pt2[0]-b1[0],pt2[1]-b1[1]); break
                else:
                    b1,b2,c=self.boxes[self.drag_idx]
                    bw,bh=b2[0]-b1[0],b2[1]-b1[1]
                    np1=(pt2[0]-self.drag_off[0],pt2[1]-self.drag_off[1])
                    self.boxes[self.drag_idx]=[np1,(np1[0]+bw,np1[1]+bh),c]
            else:
                if self.active_box: self.boxes.append(self.active_box); self.active_box=None
                self.drag_idx=-1

            # ── Drawing ───────────────────────────────────────────────────────
            pc=PEN_COLS[self.pen_idx]
            if self.draw_mode and tip and frame_gest=="POINTING":
                self.active_stroke.append((tip,pc))
            elif self.active_stroke:
                self.strokes.append(self.active_stroke); self.active_stroke=[]

            for b in self.boxes: cv2.rectangle(frame,b[0],b[1],b[2],3,cv2.LINE_AA)
            for s in self.strokes:
                for j in range(1,len(s)): cv2.line(frame,s[j-1][0],s[j][0],s[j][1],self.thickness,cv2.LINE_AA)
            for j in range(1,len(self.active_stroke)):
                cv2.line(frame,self.active_stroke[j-1][0],self.active_stroke[j][0],pc,self.thickness,cv2.LINE_AA)

            # ── Voice commands ───────────────────────────────────────────────
            self._handle_voice()

            # ── FPS ──────────────────────────────────────────────────────────
            now=time.time(); self.fps=1/(now-prev+1e-9); prev=now
            self._hud(frame)
            cv2.imshow("VISION-PRO OMNI | Frosts v0.4.0",frame)

            # ── Keys ─────────────────────────────────────────────────────────
            k=cv2.waitKey(1)&0xFF
            if k==ord('d'):
                self._mark_ptt_seen()   # refresh "D is held" timestamp every time we see it
            if   k==27:       break
            elif k==ord('q'): self.enabled["hand"]  ^=1
            elif k==ord('w'): self.enabled["pose"]  ^=1
            elif k==ord('e'): self.enabled["face"]  ^=1
            elif k==ord('r'): self.enabled["object"]^=1
            elif k==ord('k'): self.pose_mode        ^=1
            elif k==ord('h'): self.show_hud         ^=1
            elif k==ord('m'): self.mirror_mode      ^=1
            elif k==ord('j'): self.show_joints      ^=1
            elif k==ord('y'): self.draw_mode        ^=1
            elif k==ord('c'): self.strokes=[]; self.boxes=[]
            elif k==ord('x'): self.pen_idx=(self.pen_idx+1)%len(PEN_COLS)
            elif k==ord('z'): self.zoom_pulse=True; self.zoom_t=time.time()
            elif k==ord('f'):
                self.mask_idx=(self.mask_idx+1)%len(MASK_MODES)
                self.flash=f"MASK: {MASK_MODES[self.mask_idx]}"; self.flash_exp=time.time()+1
            elif k==ord('1'): self.f_gray  ^=1
            elif k==ord('2'): self.f_invert^=1
            elif k==ord('3'): self.f_gray=self.f_invert=False
            elif k==82:       self.thickness=min(20,self.thickness+1)
            elif k==84:       self.thickness=max(1, self.thickness-1)

        self.cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    VisionProOmni().run()