import asyncio
import logging
import re
import ssl

import aiohttp
import certifi

import config

logger = logging.getLogger("auto_trade.notifier")

TELEGRAM_MAX_LENGTH = 4096
_TAG = re.compile(r"</?(b|i|u|s|code|pre)>")


def split_message(message: str, limit: int = TELEGRAM_MAX_LENGTH) -> list:
    """줄 단위로 잘라 limit 이하의 조각으로 나눈다."""
    chunks, current = [], ""
    for line in message.split("\n"):
        while len(line) > limit:               # 한 줄이 한도를 넘는 경우
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class AsyncTelegramNotifier:
    """비동기 텔레그램 알림 전송 클래스 (채널/개인봇 공용)"""
    def __init__(self):
        self.enabled = config.TELEGRAM_ENABLED
        self.token = config.TELEGRAM_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

        # SSL Context 설정 (macOS 등에서 인증서 라이브러리 문제 해결)
        try:
            self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        except Exception as e:
            logger.warning(f"SSL 컨텍스트 생성 중 경고: {e}. 기본 설정 사용 시도.")
            self.ssl_context = None

    async def send_message(self, message: str) -> bool:
        if not self.enabled or not self.token or not self.chat_id:
            logger.debug("텔레그램 알림이 비활성화되어 있거나 토큰/채널ID가 없습니다.")
            return False

        try:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                ok = True
                for chunk in split_message(message):
                    ok = await self._send_chunk(session, chunk) and ok
                return ok
        except Exception as e:
            logger.error(f"텔레그램 메시지 전송 중 예외 발생: {type(e).__name__}: {e}")
            return False

    async def _send_chunk(self, session: aiohttp.ClientSession, text: str) -> bool:
        """HTML 로 보내고, 파싱 오류(400)면 태그를 떼고 일반 텍스트로 다시 보낸다.

        장애 알림에는 예외 메시지처럼 `<`, `&` 가 든 문자열이 섞인다. HTML 파싱에
        실패했다고 알림 자체를 버리면 정작 장애가 났을 때 알림이 오지 않는다.
        """
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
        for attempt in range(3):
            try:
                async with session.post(self.api_url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        return True
                    body = await response.text()
                    if response.status == 400 and "parse_mode" in payload:
                        payload = {"chat_id": self.chat_id, "text": _TAG.sub("", text)}
                        continue
                    if response.status == 429:
                        retry_after = 1
                        try:
                            retry_after = int((await response.json())["parameters"]["retry_after"])
                        except Exception:
                            pass
                        await asyncio.sleep(min(retry_after, 10))
                        continue
                    logger.error(f"텔레그램 전송 실패 (상태 코드: {response.status}): {body[:200]}")
                    return False
            except asyncio.TimeoutError:
                logger.warning(f"텔레그램 API 타임아웃 (시도 {attempt + 1}/3)")
        logger.error("텔레그램 메시지 전송 최종 실패")
        return False
