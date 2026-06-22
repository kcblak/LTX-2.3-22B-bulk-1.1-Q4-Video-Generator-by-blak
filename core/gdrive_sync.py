from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class GDriveSync:
    def __init__(self, folder_id: Optional[str], shared_drive_id: Optional[str], sa_json_path: Optional[str]):
        self.folder_id = folder_id
        self.shared_drive_id = shared_drive_id
        self.sa_json_path = sa_json_path
        self._service = None
        self.available = False

        if not sa_json_path or not Path(sa_json_path).is_file():
            return
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            import json
            with open(sa_json_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/drive"]
            )
            self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
            self.available = True
        except Exception:
            pass

    def upload(self, file_path: Path, folder_id: Optional[str] = None) -> Optional[str]:
        if not self.available or self._service is None:
            return None
        try:
            from googleapiclient.http import MediaFileUpload
            filename = file_path.name
            metadata = {"name": filename}
            target = folder_id or self.folder_id
            if target:
                metadata["parents"] = [target]
            media = MediaFileUpload(str(file_path), mimetype="video/mp4", resumable=True)
            kwargs = {"body": metadata, "media_body": media, "fields": "id"}
            if self.shared_drive_id:
                kwargs.update({"supportsAllDrives": True})
            file = self._service.files().create(**kwargs).execute()
            return file.get("id")
        except Exception:
            return None
