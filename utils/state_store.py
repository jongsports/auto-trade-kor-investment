"""재시작을 넘겨 살아남아야 하는 상태의 파일 저장소.

`data/state/` 디렉터리에 JSON 으로 저장한다. 서버에서는 이 디렉터리가 통째로
볼륨 마운트되므로 (개별 파일 마운트가 아님) 임시 파일 + rename 으로 원자적
교체가 가능하다. 쓰기 도중 프로세스가 죽어도 이전 내용이 남는다.
"""
import json
import logging
import os
import tempfile
from typing import Any

import config

logger = logging.getLogger("auto_trade.state")


def _path(name: str) -> str:
    return os.path.join(str(config.STATE_DIR), f"{name}.json")


def save_state(name: str, data: Any) -> bool:
    path = _path(name)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=f".{name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return True
    except Exception as e:
        logger.error(f"[상태저장] {name} 저장 실패: {e}")
        return False


def load_state(name: str, default: Any) -> Any:
    path = _path(name)
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return json.loads(content) if content else default
    except Exception as e:
        logger.error(f"[상태저장] {name} 로드 실패 — 기본값 사용: {e}")
        return default
