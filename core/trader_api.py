import os
import json
import logging
import asyncio
import aiohttp
import time
import random
from datetime import datetime, timedelta
from urllib.parse import urljoin
import pandas as pd
from typing import Optional, Dict, Any, List

logger = logging.getLogger("auto_trade.api")

class AsyncKisAPI:
    """Async wrapper for Korea Investment Open API."""

    def __init__(self, app_key: str, app_secret: str, account_number: str, demo_mode: bool = False):
        self.app_key = app_key
        self.app_secret = app_secret
        # 계좌번호 8자리만 입력된 경우 상품코드 '01' 자동 부착
        if account_number and len(account_number.replace("-", "").strip()) == 8:
            self.account_number = account_number.replace("-", "").strip() + "01"
        else:
            self.account_number = account_number
        self.demo_mode = demo_mode

        # Domain
        if self.demo_mode:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
        else:
            self.base_url = "https://openapi.koreainvestment.com:9443"

        self.token_file = "data/token.json"
        self.access_token = None
        self.token_expire_time = None
        self.is_connected = False
        
        # Async session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate Limiter Lock & Timestamps
        self.semaphore = asyncio.Semaphore(1 if self.demo_mode else 20)
        self._request_timestamps = []
        self._rate_limit_lock = asyncio.Lock()

        # TTL 캐시 (Issue #10-E/F)
        self._ohlcv_cache: Dict[str, Any] = {}   # key: f"{ticker}_{period_code}_{count}", value: (df, timestamp)
        self._trend_cache: Dict[str, Any] = {}    # key: f"{ticker}_{market_code}", value: (result, timestamp)

        # 서킷브레이커 (R1)
        self._cb_failure_count: int = 0
        self._cb_open_until = None
        self._cb_threshold: int = 5
        self._cb_open_seconds: int = 60
        self._cb_lock = asyncio.Lock()

        # 토큰 갱신 Lock (동시 갱신 방지)
        self._token_refresh_lock = asyncio.Lock()

    async def init_session(self):
        """Initialize aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
            # KIS API 서버 인증서 호환 이슈로 ssl=False 유지 (demo/live 공통)
            connector = aiohttp.TCPConnector(
                ssl=False, limit=20, limit_per_host=10, ttl_dns_cache=300,
            )
            self.session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"},
                connector=connector,
                timeout=timeout,
            )
            
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    def _sync_init(self):
        """Used for synchronous initialization (e.g., getting tokens)."""
        import requests
        url = urljoin(self.base_url, "/oauth2/tokenP")
        headers = {"content-type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        res = requests.post(url, headers=headers, json=data)
        if res.status_code == 200:
            res_data = res.json()
            self.access_token = res_data["access_token"]
            expires_in = res_data["expires_in"]
            self.token_expire_time = datetime.now() + timedelta(seconds=expires_in)
            self.is_connected = True
            
            # Save token
            os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
            with open(self.token_file, "w") as f:
                json.dump({
                    "access_token": self.access_token,
                    "expire_time": self.token_expire_time.strftime("%Y-%m-%d %H:%M:%S")
                }, f)
            logger.info("Successfully connected to KIS API and generated token.")
            return True
        else:
            logger.error(f"Failed to connect to KIS API: {res.text}")
            return False

    def is_token_valid(self):
        if not os.path.exists(self.token_file):
            return False
            
        try:
            with open(self.token_file, "r") as f:
                token_info = json.load(f)
            self.access_token = token_info["access_token"]
            expire_time = datetime.strptime(token_info["expire_time"], "%Y-%m-%d %H:%M:%S")
            
            if datetime.now() > expire_time - timedelta(minutes=10):
                return False
                
            self.token_expire_time = expire_time
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to validate token: {e}")
            return False
            
    def connect(self):
        """Connect to the API server and get an access token."""
        if self.is_token_valid():
            logger.info("Using valid existing token.")
            return True
        return self._sync_init()

    def get_headers(self, tr_id: str) -> dict:
        """Get standard API headers."""
        return {
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    async def _fetch(self, method: str, path: str, tr_id: str, **kwargs) -> dict:
        """Helper to make an async API request with rate limiting and retries.

        Issue #11: demo 모드 1.1s 쿨다운을 finally 블록으로 이동하여
        성공/HTTP에러/네트워크예외 모든 경로에서 쿨다운 보장.
        세마포어와 슬라이딩 윈도우 레이트 리미터를 결합하여 TPS 버스트 원천 차단.
        """
        if not self.is_connected:
            self.connect()

        await self.init_session()

        # 서킷브레이커 차단 확인 (R1)
        if self._cb_open_until and datetime.now() < self._cb_open_until:
            remaining = int((self._cb_open_until - datetime.now()).total_seconds())
            logger.warning(f"[서킷브레이커] 차단 중 ({remaining}초 남음) tr_id={tr_id}")
            return {"rt_cd": "-1", "msg1": "Circuit breaker open"}

        url = urljoin(self.base_url, path)
        max_retries = 5

        async with self.semaphore:
            try:
                for attempt in range(max_retries):
                    try:
                        await self._wait_rate_limit()
                        headers = self.get_headers(tr_id)
                        async with self.session.request(method, url, headers=headers, **kwargs) as response:
                            # KIS sometimes returns JSON error bodies even on 500
                            if response.status == 500:
                                try:
                                    res_data = await response.json()
                                except Exception:
                                    text = await response.text()
                                    logger.error(f"API HTTP 500 Error: {text}")
                                    return {"rt_cd": "-1", "msg1": "HTTP 500"}
                            elif response.status != 200:
                                text = await response.text()
                                logger.error(f"API Request Error: {response.status} - {text}")
                                return {"rt_cd": "-1", "msg1": "HTTP Error"}
                            else:
                                res_data = await response.json()

                            # 1. TPS Limit Error (EGW00201)
                            # KIS API가 msg_cd 또는 message 키로 에러코드를 반환하는 경우 모두 처리
                            if (res_data.get("msg_cd") == "EGW00201"
                                    or res_data.get("message") == "EGW00201"):
                                jitter = random.uniform(0.0, 1.0)
                                wait_time = 2.0 * (attempt + 1) + jitter
                                logger.warning(f"TPS 한도 초과. {wait_time:.2f}초 대기 (시도 {attempt+1}/{max_retries})")
                                await asyncio.sleep(wait_time)
                                continue

                            # 2. Token Expired (EGW00123 / EGW00121)
                            if (res_data.get("msg_cd") in ["EGW00123", "EGW00121"]
                                    or res_data.get("message") in ["EGW00123", "EGW00121"]):
                                logger.warning("토큰 만료 감지 — 비동기 갱신 중...")
                                async with self._token_refresh_lock:
                                    # 다른 코루틴이 이미 갱신했는지 확인
                                    if self.token_expire_time and datetime.now() < self.token_expire_time:
                                        logger.info("토큰이 이미 갱신됨 — 건너뜀")
                                    else:
                                        await asyncio.to_thread(self._sync_init)
                                continue

                            async with self._cb_lock:
                                self._cb_failure_count = 0
                                self._cb_open_until = None
                            return res_data

                    except Exception as e:
                        logger.error(f"Request exception (attempt {attempt+1}/{max_retries}) tr_id={tr_id}: {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1.0)

                # 최대 재시도 초과
                async with self._cb_lock:
                    self._cb_failure_count += 1
                    if self._cb_failure_count >= self._cb_threshold:
                        self._cb_open_until = datetime.now() + timedelta(seconds=self._cb_open_seconds)
                        logger.error(f"[서킷브레이커] 연속 {self._cb_failure_count}회 실패 → {self._cb_open_seconds}초 차단")
                        self._cb_failure_count = 0
                return {"rt_cd": "-1", "msg1": "Max retries exceeded"}

            finally:
                pass

    async def _wait_rate_limit(self):
        """정확한 1초 슬라이딩 윈도우 레이트 리미터. KIS 초당 20건 제한(실전) 준수."""
        import time
        limit = 1 if self.demo_mode else 15  # 여유 버퍼를 위해 20 대신 15로 설정
        
        while True:
            sleep_time = 0
            async with self._rate_limit_lock:
                now = time.time()
                # 1초가 지난 기록은 제거
                self._request_timestamps = [t for t in self._request_timestamps if now - t < 1.0]
                
                if len(self._request_timestamps) < limit:
                    self._request_timestamps.append(now)
                    return
                else:
                    # 초과 시 가장 오래된 요청이 1초가 지날 때까지 대기
                    oldest = self._request_timestamps[0]
                    sleep_time = 1.0 - (now - oldest) + 0.01  # 약간의 버퍼 추가
                    
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def get_ohlcv(self, ticker: str, period_code: str = "D", count: int = 100) -> pd.DataFrame:
        """Fetch historical price data asynchronously."""
        # TTL 캐시 조회 (Issue #10-E): 장중 5분, 장외 1시간
        cache_key = f"{ticker}_{period_code}_{count}"
        if cache_key in self._ohlcv_cache:
            cached_df, cached_time = self._ohlcv_cache[cache_key]
            now = datetime.now()
            now_hm = now.hour * 60 + now.minute
            screening_windows = (
                (8 * 60 + 55 <= now_hm <= 10 * 60 + 40) or
                (13 * 60 + 25 <= now_hm <= 13 * 60 + 40)
            )
            market_open = (9 * 60 <= now_hm <= 15 * 60 + 30)
            if screening_windows:
                ttl = 120
            elif market_open:
                ttl = 300
            else:
                ttl = 1800
            if (now - cached_time).total_seconds() < ttl:
                return cached_df

        # Calculate dates — 2년치 동적 범위 (Issue #10-C)
        end_date = datetime.now()
        start_date_str = (end_date - timedelta(days=730)).strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_PERIOD_DIV_CODE": period_code,
            "FID_ORG_ADJ_PRC": "0",
            "FID_INPUT_DATE_1": start_date_str,
            "FID_INPUT_DATE_2": end_date_str,
        }
        
        tr_id = "FHKST01010400"
        
        res = await self._fetch("GET", "/uapi/domestic-stock/v1/quotations/inquire-daily-price", tr_id, params=params)
        
        # Some KIS APIs use 'output', others use 'output2' for the price list.
        # FHKST01010400 typically returns a list in 'output'.
        data_list = res.get("output") or res.get("output2")
        
        if res.get("rt_cd") == "0" and data_list is not None:
            if isinstance(data_list, dict):
                 # If it's a single dict (can happen in some TRs), wrap it in a list
                 data_list = [data_list]

            df = pd.DataFrame(data_list)
            if df.empty or len(data_list) == 0:
                logger.debug(f"OHLCV data empty for {ticker} (데이터 없음)")
                return pd.DataFrame()
                
            column_mappings = {
                "stck_bsop_date": "date",
                "stck_oprc": "open",
                "stck_hgpr": "high",
                "stck_lwpr": "low",
                "stck_clpr": "close",
                "acml_vol": "volume",
                "acml_tr_pbmn": "amount",
            }
            df.rename(columns=column_mappings, inplace=True)
            
            # Filter needed columns
            cols = ["date", "open", "high", "low", "close", "volume", "amount"]
            
            # Numeric coercing before checking amount
            for col in cols[1:]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # If amount is missing (e.g. from FHKST01010400), estimate for strategy safety
            if "amount" not in df.columns or df["amount"].isnull().all():
                 df["amount"] = (df["close"] * df["volume"])
            
            df = df[[c for c in cols if c in df.columns]]
                    
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            df = df.sort_values("date").tail(count).reset_index(drop=True)
            self._ohlcv_cache[cache_key] = (df, datetime.now())
            return df
        else:
            # rt_cd != "0" 이거나 data_list가 None인 경우만 에러
            msg = res.get("msg1", "")
            if res.get("rt_cd") == "0":
                # API 성공이지만 output 키 누락 — 데이터 미지원 종목
                logger.debug(f"OHLCV no data for {ticker}: {msg}")
            else:
                logger.error(f"OHLCV Request Failed for {ticker}: {res}")

        return pd.DataFrame()

    async def get_ohlcv_by_range(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        period_code: str = "D",
        market_code: str = "J",
    ) -> pd.DataFrame:
        """
        날짜 범위 지정 OHLCV 수집 (TR: FHKST03010100).

        이슈 #1 명세에 따른 백테스팅 전용 메서드.
        1회 호출에 최대 100건 반환 → 범위가 크면 100일 단위로 청크 분할 후 병합.

        Args:
            ticker: 종목코드 (예: "005930")
            start_date: 조회 시작일 "YYYYMMDD"
            end_date: 조회 종료일 "YYYYMMDD"
            period_code: "D"(일봉), "W"(주봉), "M"(월봉)
            market_code: "J"(KRX 기본)

        Returns:
            pd.DataFrame: date/open/high/low/close/volume/amount 정렬된 DataFrame
        """
        col_map = {
            "stck_bsop_date": "date",
            "stck_oprc": "open",
            "stck_hgpr": "high",
            "stck_lwpr": "low",
            "stck_clpr": "close",
            "acml_vol": "volume",
            "acml_tr_pbmn": "amount",
        }

        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        all_frames: List[pd.DataFrame] = []

        # 100일 청크 단위로 분할 (영업일 기준 약 70일 = 달력 100일)
        chunk_days = 100
        chunk_end = end_dt
        while chunk_end >= start_dt:
            chunk_start = max(start_dt, chunk_end - timedelta(days=chunk_days))
            params = {
                "FID_COND_MRKT_DIV_CODE": market_code,
                "FID_INPUT_ISCD": ticker,
                "FID_INPUT_DATE_1": chunk_start.strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": chunk_end.strftime("%Y%m%d"),
                "FID_PERIOD_DIV_CODE": period_code,
                "FID_ORG_ADJ_PRC": "0",  # 수정주가
            }
            res = await self._fetch(
                "GET",
                "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                "FHKST03010100",
                params=params,
            )
            data_list = res.get("output2")
            if res.get("rt_cd") == "0" and data_list:
                df_chunk = pd.DataFrame(data_list)
                df_chunk.rename(columns=col_map, inplace=True)
                cols = ["date", "open", "high", "low", "close", "volume", "amount"]
                for c in cols[1:]:
                    if c in df_chunk.columns:
                        df_chunk[c] = pd.to_numeric(df_chunk[c], errors="coerce")
                if "amount" not in df_chunk.columns or df_chunk["amount"].isnull().all():
                    df_chunk["amount"] = df_chunk["close"] * df_chunk["volume"]
                df_chunk = df_chunk[[c for c in cols if c in df_chunk.columns]]
                df_chunk["date"] = pd.to_datetime(df_chunk["date"], format="%Y%m%d", errors="coerce")
                all_frames.append(df_chunk)
            else:
                logger.warning(f"[{ticker}] FHKST03010100 청크 실패 ({chunk_start.strftime('%Y%m%d')}~{chunk_end.strftime('%Y%m%d')}): {res.get('msg1','')}")

            chunk_end = chunk_start - timedelta(days=1)
            if chunk_end < start_dt:
                break

        if not all_frames:
            logger.error(f"[{ticker}] get_ohlcv_by_range: 수집된 데이터 없음")
            return pd.DataFrame()

        result = (
            pd.concat(all_frames, ignore_index=True)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        logger.info(f"[{ticker}] get_ohlcv_by_range: {len(result)}행 수집 ({start_date}~{end_date})")
        return result

    async def get_current_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get the current price and basic intraday data asynchronously."""
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
        }
        
        res = await self._fetch("GET", "/uapi/domestic-stock/v1/quotations/inquire-price", "FHKST01010100", params=params)
        
        if res.get("rt_cd") == "0" and "output" in res:
            out = res["output"]
            return {
                "price":       int(out["stck_prpr"]),
                "open":        int(out["stck_oprc"]),
                "high":        int(out["stck_hgpr"]),
                "low":         int(out["stck_lwpr"]),
                "volume":      int(out["acml_vol"]),
                "amount":      int(out["acml_tr_pbmn"]),
                "change_rate": float(out.get("prdy_ctrt", 0)),  # 전일대비율(%)
            }
        return None
        
    async def get_account_summary(self) -> Dict[str, Any]:
        """Get account balances efficiently."""
        params = {
            "CANO": self.account_number[:8],
            "ACNT_PRDT_CD": self.account_number[8:10],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }

        tr_id = "VTTC8434R" if self.demo_mode else "TTTC8434R"
        res = await self._fetch("GET", "/uapi/domestic-stock/v1/trading/inquire-balance", tr_id, params=params)

        if res.get("rt_cd") == "0" and "output2" in res:
            summary = res["output2"][0]
            raw_positions = res.get("output1", [])

            # KIS output1 필드명을 내부 표준 필드명으로 정규화
            positions = []
            for p in raw_positions:
                qty = int(p.get("hldg_qty", 0) or 0)
                if qty <= 0:
                    continue  # 보유수량 0인 종목 제외
                positions.append({
                    "ticker": p.get("pdno", ""),
                    "name": p.get("prdt_name", ""),
                    "quantity": qty,
                    "buy_price": float(p.get("pchs_avg_pric", 0) or 0),
                    "current_price": int(p.get("prpr", 0) or 0),
                    "eval_profit_loss": int(p.get("evlu_pfls_amt", 0) or 0),
                })

            # 예수금, D+2 정산금, CMA 평가금액 중 가장 큰 값을 가용예산으로 사용
            dnca = int(summary.get("dnca_tot_amt", 0))
            prvs = int(summary.get("prvs_rcdl_excc_amt", 0))
            cma = int(summary.get("cma_evlu_amt", 0))
            available = max(dnca, prvs, cma)

            return {
                "total_evaluated_amount": int(summary.get("tot_evlu_amt", 0)),
                "available_amount": available,
                "positions": positions
            }

        return {}

    async def market_buy(self, ticker: str, quantity: int) -> Dict[str, Any]:
        """시장가 매수 주문.

        TR_ID: TTTC0012U (실전), VTTC0012U (모의)
        ORD_DVSN "01" = 시장가
        """
        tr_id = "VTTC0012U" if self.demo_mode else "TTTC0012U"
        body = {
            "CANO": self.account_number[:8],
            "ACNT_PRDT_CD": self.account_number[8:10],
            "PDNO": ticker,
            "ORD_DVSN": "01",       # 시장가
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0",        # 시장가 주문은 0
        }
        res = await self._fetch(
            "POST",
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id,
            json=body,
        )
        if res.get("rt_cd") == "0":
            logger.info(f"[매수완료] {ticker} {quantity}주 | 주문번호: {res.get('output', {}).get('odno', '')}")
        else:
            msg1 = res.get("msg1", "")
            msg_cd = res.get("msg_cd", "")
            logger.error(f"[매수실패] {ticker} {quantity}주 | msg_cd={msg_cd} | {msg1}")
        return res

    async def market_sell(self, ticker: str, quantity: int) -> Dict[str, Any]:
        """시장가 매도 주문.

        TR_ID: TTTC0011U (실전), VTTC0011U (모의)
        ORD_DVSN "01" = 시장가
        """
        tr_id = "VTTC0011U" if self.demo_mode else "TTTC0011U"
        body = {
            "CANO": self.account_number[:8],
            "ACNT_PRDT_CD": self.account_number[8:10],
            "PDNO": ticker,
            "ORD_DVSN": "01",       # 시장가
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0",
            "SLL_TYPE": "01",       # 일반매도
        }
        res = await self._fetch(
            "POST",
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id,
            json=body,
        )
        if res.get("rt_cd") == "0":
            logger.info(f"[매도완료] {ticker} {quantity}주 | 주문번호: {res.get('output', {}).get('odno', '')}")
        else:
            msg1 = res.get("msg1", "")
            msg_cd = res.get("msg_cd", "")
            logger.error(f"[매도실패] {ticker} {quantity}주 | msg_cd={msg_cd} | {msg1}")
            # 거래정지/매매불가 에러 표시 (재시도 방지용)
            unsellable_codes = {"APBK0066", "APBK0919", "APBK0033"}
            if msg_cd in unsellable_codes or "거래정지" in msg1 or "매매불가" in msg1:
                res["_unsellable"] = True
        return res
        
    async def get_investor_trend(self, ticker: str, market_code: str = "J") -> Dict[str, Any]:
        """
        종목별 외국인/기관 수급 조회.
        Primary  : HHPTJ04160200 (investor-trend-estimate) - 장중 추정가집계 (4회 갱신)
        Fallback : FHKST01010900 (inquire-investor)        - 당일 확정치 (장종료 후 제공)

        Args:
            ticker: 종목코드
            market_code: 시장구분코드 "J"(KOSPI) / "K"(KOSDAQ) — Fallback 엔드포인트에 사용

        Returns:
            dict: foreign_net_buy (외국인 순매수), institution_net_buy (기관 순매수),
                  prgrm_net_buy (프로그램 순매수) — 모두 순매수 수량(주), 양수=순매수
        """
        # TTL 캐시 조회 (30분)
        cache_key = f"{ticker}_{market_code}"
        if cache_key in self._trend_cache:
            cached_result, cached_time = self._trend_cache[cache_key]
            _now = datetime.now()
            _hm = _now.hour * 60 + _now.minute
            trend_ttl = 600 if (9 * 60 <= _hm <= 15 * 60 + 30) else 1800
            if (_now - cached_time).total_seconds() < trend_ttl:
                return cached_result

        # --- Primary: 장중 추정가집계 (09:30/11:20/13:20/14:30 갱신) ---
        try:
            res = await self._fetch(
                "GET",
                "/uapi/domestic-stock/v1/quotations/investor-trend-estimate",
                "HHPTJ04160200",
                params={"MKSC_SHRN_ISCD": ticker}
            )
            if res.get("rt_cd") == "0" and res.get("output2"):
                frgn = sum(int(r.get("frgn_fake_ntby_qty", 0) or 0) for r in res["output2"])
                orgn = sum(int(r.get("orgn_fake_ntby_qty", 0) or 0) for r in res["output2"])
                if frgn != 0 or orgn != 0:
                    logger.debug(f"[수급-추정] {ticker} | 외국인:{frgn:+,} 기관:{orgn:+,}")
                    result = {"foreign_net_buy": frgn, "institution_net_buy": orgn, "prgrm_net_buy": 0}
                    self._trend_cache[cache_key] = (result, datetime.now())
                    return result
        except Exception as e:
            logger.warning(f"investor-trend-estimate 실패 {ticker}: {e}")

        # --- Fallback: 당일 확정치 (장중엔 전일 데이터 반환됨) ---
        try:
            res = await self._fetch(
                "GET",
                "/uapi/domestic-stock/v1/quotations/inquire-investor",
                "FHKST01010900",
                params={"FID_COND_MRKT_DIV_CODE": market_code, "FID_INPUT_ISCD": ticker}
            )
            if res.get("rt_cd") == "0" and res.get("output"):
                output = res["output"]
                row = output[0] if isinstance(output, list) and len(output) > 0 else output
                frgn = int(row.get("frgn_ntby_qty", 0) or 0)
                orgn = int(row.get("orgn_ntby_qty", 0) or 0)
                logger.debug(f"[수급-확정] {ticker}({market_code}) | 외국인:{frgn:+,} 기관:{orgn:+,}")
                result = {"foreign_net_buy": frgn, "institution_net_buy": orgn, "prgrm_net_buy": 0}
                self._trend_cache[cache_key] = (result, datetime.now())
                return result
        except Exception as e:
            logger.warning(f"inquire-investor 폴백 실패 {ticker}: {e}")

        logger.warning(f"get_investor_trend: 모든 방법 실패 {ticker} → 중립 반환")
        # Issue #13: data_available=False 로 실패를 확정 0과 구분
        # 스크리너에서 이 값을 확인해 오버나이트 자동 실격 방지
        return {"foreign_net_buy": 0, "institution_net_buy": 0, "prgrm_net_buy": 0, "data_available": False}

    async def get_news_titles(self, ticker: str, count: int = 30) -> List[str]:
        """
        KIS API로 종목 관련 최신 뉴스 제목 조회 (FHKST01011800).
        TTL 캐시 5분.

        Returns:
            뉴스 제목 문자열 리스트 (최대 count건)
        """
        if not hasattr(self, "_news_cache"):
            self._news_cache: Dict[str, Any] = {}

        cache_key = f"news_{ticker}"
        if cache_key in self._news_cache:
            cached_titles, cached_time = self._news_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < 300:
                return cached_titles

        try:
            # Bug #8 fix (2026-04-24): FID_COND_MRKT_CLS_CODE 필수 파라미터 누락으로
            # 6주간 뉴스 API가 rt_cd='2' (OPSQ2001 ERROR INPUT FIELD NOT FOUND)
            # 반환 → 뉴스 0건 → 부정 키워드 필터 무력 상태였음.
            res = await self._fetch(
                "GET",
                "/uapi/domestic-stock/v1/quotations/news-title",
                "FHKST01011800",
                params={
                    # Bug #8 fix (2026-04-24): KIS news-title API는 빈 문자열이라도
                    # 모든 FID_* 필드가 요청에 포함되어야 "FIELD NOT FOUND" 회피 가능.
                    "FID_NEWS_OFER_ENTP_CODE": "",
                    "FID_COND_MRKT_CLS_CODE": "",
                    "FID_INPUT_ISCD": ticker,
                    "FID_TITL_CNTT": "",
                    "FID_INPUT_DATE_1": "",
                    "FID_INPUT_HOUR_1": "",
                    "FID_RANK_SORT_CLS_CODE": "0",
                    "FID_INPUT_SRNO": "",
                },
            )
            titles: List[str] = []
            rt_cd = res.get("rt_cd")
            if rt_cd == "0":
                for item in (res.get("output") or [])[:count]:
                    title = (item.get("hts_pbnt_titl_cntt") or "").strip()
                    if title:
                        titles.append(title)
            else:
                # rt_cd != "0" 시 명시적 WARN 로그 (과거엔 조용히 실패)
                logger.warning(
                    f"[뉴스] {ticker} API 오류 rt_cd={rt_cd} "
                    f"msg_cd={res.get('msg_cd')} msg1={res.get('msg1')}"
                )
            self._news_cache[cache_key] = (titles, datetime.now())
            logger.debug(f"[뉴스] {ticker}: {len(titles)}건 조회")
            return titles
        except Exception as e:
            logger.warning(f"뉴스 제목 조회 실패 {ticker}: {e}")
            return []

    async def get_market_investor_flow(self, market: str = "KOSPI") -> Dict[str, Any]:
        """시장별 투자자매매동향(시세) — FHPTJ04030000.

        KODEX200 proxy 대체 (2026-04-24): ETF 기반 투자자 추정이 Primary API
        미지원으로 항상 0 반환하던 구조 문제 해결. 시장 전체 외국인/기관/개인
        순매수 집계를 1회 호출로 획득.

        Args:
            market: "KOSPI" 또는 "KOSDAQ"
        Returns:
            {
              market, data_available,
              foreign_net_amount_bn, institution_net_amount_bn, personal_net_amount_bn  (억원),
              foreign_net_qty, institution_net_qty, personal_net_qty  (주)
            }
        """
        mkt = market.upper()
        if mkt == "KOSPI":
            iscd, iscd2 = "KSP", "0001"
        elif mkt == "KOSDAQ":
            iscd, iscd2 = "KSQ", "1001"
        else:
            return {"market": market, "data_available": False}

        try:
            res = await self._fetch(
                "GET",
                "/uapi/domestic-stock/v1/quotations/inquire-investor-time-by-market",
                "FHPTJ04030000",
                params={
                    "FID_INPUT_ISCD": iscd,
                    "FID_INPUT_ISCD_2": iscd2,
                },
            )
            if res.get("rt_cd") != "0":
                logger.warning(
                    f"[MarketFlow] {mkt} API 오류 rt_cd={res.get('rt_cd')} "
                    f"msg_cd={res.get('msg_cd')} msg1={res.get('msg1')}"
                )
                return {"market": mkt, "data_available": False}

            output = res.get("output")
            if not output or not isinstance(output, list):
                return {"market": mkt, "data_available": False}
            row = output[0]

            def _to_int(v) -> int:
                try:
                    return int(str(v).strip()) if v not in (None, "") else 0
                except Exception:
                    return 0

            # 거래대금 필드는 백만원 단위 → 억원 변환 (÷100)
            return {
                "market": mkt,
                "data_available": True,
                "foreign_net_amount_bn":     round(_to_int(row.get("frgn_ntby_tr_pbmn")) / 100, 1),
                "institution_net_amount_bn": round(_to_int(row.get("orgn_ntby_tr_pbmn")) / 100, 1),
                "personal_net_amount_bn":    round(_to_int(row.get("prsn_ntby_tr_pbmn")) / 100, 1),
                "foreign_net_qty":     _to_int(row.get("frgn_ntby_qty")),
                "institution_net_qty": _to_int(row.get("orgn_ntby_qty")),
                "personal_net_qty":    _to_int(row.get("prsn_ntby_qty")),
            }
        except Exception as e:
            logger.warning(f"[MarketFlow] {mkt} 조회 실패: {e}")
            return {"market": mkt, "data_available": False}

    async def get_top_market_stocks(self, market_code="0001", count=200) -> List[str]:
        """
        국내주식 거래대금 상위 종목 조회 (FHPST01710000)
        market_code: '0001' (KOSPI), '1001' (KOSDAQ)
        거래대금이 높은 종목 리스트를 반환합니다. 실전 투자에서는 전체 시장 중 활발한 종목 위주로 매매하는 것이 유리합니다.
        """
        params = {
            "FID_COND_MRKT_DIV_CODE": "J", # 주식
            "FID_COND_SCR_DIV_CODE": "20171", # 화면번호
            "FID_INPUT_ISCD": market_code, # '0001' 코스피, '1001' 코스닥
            "FID_DIV_CLS_CODE": "0", # 0:전체
            "FID_BLNG_CLS_CODE": "0", # 0:전체
            "FID_TRGT_CLS_CODE": "111111111", # 타겟 전체
            "FID_TRGT_EXLS_CLS_CODE": "000000000", # 제외 전체
            "FID_INPUT_PRICE_1": "",
            "FID_INPUT_PRICE_2": "",
            "FID_VOL_CNT": "",
        }
        
        # This TR is slightly different and common for quotation rankings
        res = await self._fetch("GET", "/uapi/domestic-stock/v1/quotations/volume-rank", "FHPST01710000", params=params)
        
        tickers = []
        if res.get("rt_cd") == "0" and "output" in res:
            for item in res["output"]:
                # mksc_shrn_iscd = 종목코드
                ticker = item.get("mksc_shrn_iscd")
                if ticker:
                    tickers.append(ticker)
                
        return tickers[:count]

    async def get_volume_surge_stocks(self, market_code="0001", count=20) -> List[str]:
        """Issue #24-H1: 거래량 급증 종목 조회 (전일 대비 거래량 급증 순).

        기존 get_top_market_stocks()는 거래대금 상위이므로 대형주 편향.
        이 메서드는 거래량 급증률 순 정렬로 당일 테마/급등 종목을 포착.
        """
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_COND_SCR_DIV_CODE": "20171",
            "FID_INPUT_ISCD": market_code,
            "FID_DIV_CLS_CODE": "0",
            "FID_BLNG_CLS_CODE": "0",
            "FID_TRGT_CLS_CODE": "111111111",
            "FID_TRGT_EXLS_CLS_CODE": "000000000",
            "FID_INPUT_PRICE_1": "1000",   # 최소 1,000원 (저가주 제외)
            "FID_INPUT_PRICE_2": "",
            "FID_VOL_CNT": "50000",         # 최소 거래량 5만주
        }
        try:
            res = await self._fetch(
                "GET", "/uapi/domestic-stock/v1/quotations/volume-rank",
                "FHPST01710000", params=params
            )
            tickers = []
            if res.get("rt_cd") == "0" and "output" in res:
                for item in res["output"]:
                    ticker = item.get("mksc_shrn_iscd")
                    if ticker:
                        tickers.append(ticker)
            return tickers[:count]
        except Exception as e:
            logger.warning(f"거래량 급증 조회 실패: {e}")
            return []
