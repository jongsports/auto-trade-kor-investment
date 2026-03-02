import logging
import aiohttp
import config
import asyncio
import ssl
import certifi

logger = logging.getLogger("auto_trade.notifier")

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
        
    async def send_message(self, message: str):
        if not self.enabled or not self.token or not self.chat_id:
            logger.debug("텔레그램 알림이 비활성화되어 있거나 토큰/채널ID가 없습니다.")
            return False
            
        try:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                payload = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                }
                async with session.post(self.api_url, json=payload, timeout=5) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"텔레그램 채널 메시지 전송 실패 (상태 코드: {response.status}): {error_text}")
                        return False
                    return True
        except asyncio.TimeoutError:
            logger.error("텔레그램 API 타임아웃 발생 (채널 메시지 전송 실패)")
            return False
        except Exception as e:
            logger.error(f"텔레그램 메시지 전송 중 예외 발생: {e}")
            return False
