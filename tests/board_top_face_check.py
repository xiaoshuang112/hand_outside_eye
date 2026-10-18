#!/usr/bin/env python3
import ctypes
import json
import os
import sys

APP = "/home/sunrise/Desktop/ai-det-feilun"
IMG = "/home/sunrise/work/zhanhui/input/Image_20260930111936842.bmp"
LIB = os.path.join(APP, "libai-det.so")

lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
lib.AI_det_init.argtypes = [ctypes.c_char_p]
lib.AI_det_init.restype = ctypes.c_int
lib.AI_det.argtypes = [
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
]
lib.AI_det.restype = ctypes.c_int
lib.AI_det_close.restype = ctypes.c_int

code = lib.AI_det_init(APP.encode())
print("AI_det_init", code)
if code != 0:
    sys.exit(1)


def call(req):
    img = ctypes.c_char_p()
    w = ctypes.c_int(0)
    h = ctypes.c_int(0)
    out = ctypes.c_char_p()
    n = ctypes.c_int(0)
    rc = lib.AI_det(
        ctypes.byref(img), ctypes.byref(w), ctypes.byref(h),
        json.dumps(req).encode(), ctypes.byref(out), ctypes.byref(n), None)
    text = ""
    if out.value and n.value > 0:
        text = ctypes.string_at(out, n.value).decode("utf-8", "replace")
    return rc, text


rc, text = call({
    "xialiao_type": "top_face",
    "test_mode": 1,
    "color_image": IMG,
    "ref_depth": 850,
})
print("top_face rc", rc)
print(text[:4000])
data = json.loads(text) if text else {}
ok = (
    rc == 0
    and data.get("ret-code") == 0
    and isinstance(data.get("pts_lunzi"), list)
    and len(data.get("pts_lunzi") or []) > 0
    and len(data.get("angle_deg") or []) == len(data.get("pts_lunzi") or [])
)
print("top_face_ok", ok, "count", data.get("top_face_count"), "pts", len(data.get("pts_lunzi") or []))

rc2, text2 = call({
    "xialiao_type": "feilun_up",
    "test_mode": 1,
    "color_image": os.path.join(APP, "preview.jpg"),
    "ref_depth": 850,
    "clean_mode": True,
})
print("feilun_up rc", rc2)
print(text2[:1500])

lib.AI_det_close()
sys.exit(0 if ok else 6)
