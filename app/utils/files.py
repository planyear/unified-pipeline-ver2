import os, tempfile, shutil
from fastapi import UploadFile

def get_ext(filename: str) -> str:
    return os.path.splitext(filename.lower())[-1].lstrip(".")

def save_upload_to_tmp(upload: UploadFile) -> str:
    fd, path = tempfile.mkstemp(prefix="incoming_", suffix=f"_{upload.filename}")
    with os.fdopen(fd, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path
