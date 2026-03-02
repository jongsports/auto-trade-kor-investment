import logging
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional

import config
from utils.utils import is_market_open, is_trading_time

logger = logging.getLogger("auto_trade.risk_manager")


class RiskManager:
    """동적 리스크 관리 클래스"""

    def __init__(self, api_client):
        """
        Args:
            api_client (KisAPI): 한국투자증권 API 클라이언트
        """
        self.api_client = api_client

        # 기본 리스크 파라미터 초기화
        self.max_position_size = config.MAX_STOCK_RATIO  # 종목당 최대 포지션 비중
        self.max_total_position = config.MAX_INVESTMENT_RATIO  # 총 투자 비중
        self.base_loss_cut_ratio = config.LOSS_CUT_RATIO  # 기본 손절 비율
        self.base_profit_cut_ratio = config.PROFIT_CUT_RATIO  # 기본 익절 비율

        # 리스크 상태 관리
        self.risk_status = "NORMAL"  # NORMAL, CAUTION, RISK
        self.daily_loss_tracking = {}  # 일별 손실 추적
        self.market_condition = "NORMAL"  # BULL, NORMAL, BEAR

        # 위험 관리 로그 저장 디렉토리
        self.log_dir = "data/risk_logs"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # 시장 변동성 판단을 위한 지표
        self.vix_threshold = {
            "NORMAL_TO_CAUTION": 20,  # 정상 -> 주의 전환 VIX 임계값
            "CAUTION_TO_RISK": 30,  # 주의 -> 위험 전환 VIX 임계값
            "RISK_TO_CAUTION": 25,  # 위험 -> 주의 전환 VIX 임계값
            "CAUTION_TO_NORMAL": 15,  # 주의 -> 정상 전환 VIX 임계값
        }

        # ATR 기반 동적 손절 배수
        self.atr_multiplier = {
            "NORMAL": 2.0,  # 정상 시장 상태에서의 ATR 배수
            "CAUTION": 1.5,  # 주의 시장 상태에서의 ATR 배수
            "RISK": 1.0,  # 위험 시장 상태에서의 ATR 배수
        }

        # 리스크 상태에 따른 투자 비중 조정 계수
        self.position_size_multiplier = {
            "NORMAL": 1.0,  # 정상 시장에서 기본 투자 비중
            "CAUTION": 0.7,  # 주의 시장에서 투자 비중 감소
            "RISK": 0.4,  # 위험 시장에서 투자 비중 크게 감소
        }

        # 리스크 상태에 따른 익절 비율 조정
        self.profit_cut_multiplier = {
            "NORMAL": 1.0,  # 정상 시장에서 기본 익절 비율
            "CAUTION": 0.8,  # 주의 시장에서 익절 목표 하향 조정
            "RISK": 0.6,  # 위험 시장에서 익절 목표 크게 하향 조정
        }

        # 일일 손실 한도
        self.max_daily_loss = config.MAX_DAILY_LOSS  # 일일 최대 손실 비율

        # 주문 제한 및 과도한 레버리지 방지
        self.max_orders_per_day = 20  # 일일 최대 주문 횟수
        self.max_orders_per_minute = 3  # 분당 최대 주문 횟수
        self.max_orders_per_ticker = 5  # 동일 종목 일일 최대 주문 횟수
        self.max_leverage = 1.5  # 최대 레버리지 (계좌 자산의 1.5배)

        # 주문 추적
        self.daily_order_count = 0  # 일일 총 주문 횟수
        self.last_order_reset_date = datetime.now().date()
        self.minute_orders = []  # 최근 분당 주문 목록
        self.ticker_order_count = {}  # 종목별 일일 주문 횟수

        # 운영 제약 관련 상태
        self.trading_enabled = True  # 거래 활성화 상태
        self.trading_lock = threading.RLock()  # 스레드 안전성을 위한 락

        # 리스크 모니터링 스레드
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._risk_monitoring_worker)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        # 변동성 기반 포지션 사이징 관련 변수
        self.volatility_lookback = 20  # 변동성 계산 기간
        self.stop_loss_pct = 0.05  # 기본 손절 비율

    def __del__(self):
        """소멸자"""
        self.monitoring_active = False
        if hasattr(self, "monitoring_thread") and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)

    def _risk_monitoring_worker(self):
        """리스크 모니터링 워커 스레드"""
        while self.monitoring_active:
            try:
                # 시장 시간에 따른 거래 가능 상태 업데이트
                self._update_trading_status()

                # 시장 리스크 평가 (1시간마다)
                current_hour = datetime.now().hour
                current_minute = datetime.now().minute
                if current_minute < 5:  # 매 시간 처음 5분 이내
                    self.assess_market_risk()

                # 분당 주문 제한 초기화 (오래된 주문 제거)
                self._clean_minute_orders()

                # 일별 주문 카운터 초기화
                self._reset_daily_counters_if_needed()

                # 10초마다 체크
                time.sleep(10)
            except Exception as e:
                logger.error(f"리스크 모니터링 워커 오류: {str(e)}")
                time.sleep(30)  # 오류 발생 시 30초 대기 후 재시도

    def _update_trading_status(self):
        """거래 시간에 따른 거래 가능 상태 업데이트"""
        with self.trading_lock:
            # 거래 시간이 아니면 거래 비활성화
            trading_status = is_trading_time()

            # 상태 변경 시에만 로그 출력
            if self.trading_enabled != trading_status:
                if trading_status:
                    logger.info("거래 시간 시작: 거래 기능 활성화")
                else:
                    logger.info("거래 시간 종료: 거래 기능 비활성화")

                self.trading_enabled = trading_status

    def _reset_daily_counters_if_needed(self):
        """일별 카운터 초기화 (날짜가 바뀌었을 경우)"""
        current_date = datetime.now().date()
        if current_date != self.last_order_reset_date:
            with self.trading_lock:
                logger.info("일별 주문 카운터 초기화")
                self.daily_order_count = 0
                self.ticker_order_count = {}
                self.last_order_reset_date = current_date

    def _clean_minute_orders(self):
        """분당 주문 제한을 위한 오래된 주문 제거"""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        with self.trading_lock:
            # 1분 이내의 주문만 유지
            self.minute_orders = [
                order for order in self.minute_orders if order > one_minute_ago
            ]

    def assess_market_risk(self):
        """시장 리스크 평가"""
        # KOSPI200 변동성 지수 또는 한국형 VIX 가져오기
        try:
            # 최근 KOSPI 지수 데이터 가져오기
            kospi_data = self.api_client.get_ohlcv("U001", "D", 20)

            if kospi_data is None or kospi_data.empty:
                logger.warning("KOSPI 지수 데이터를 가져올 수 없습니다.")
                return

            # 20일 변동성 계산 (표준 편차)
            returns = kospi_data["close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # 연율화된 변동성(%)

            # 변동성을 VIX와 유사한 스케일로 조정
            adjusted_volatility = volatility * 1.2

            logger.info(f"현재 시장 변동성: {adjusted_volatility:.2f}")

            # 현재 리스크 상태에 따라 다른 임계값 적용
            if (
                self.risk_status == "NORMAL"
                and adjusted_volatility > self.vix_threshold["NORMAL_TO_CAUTION"]
            ):
                self.update_risk_status("CAUTION")
            elif self.risk_status == "CAUTION":
                if adjusted_volatility > self.vix_threshold["CAUTION_TO_RISK"]:
                    self.update_risk_status("RISK")
                elif adjusted_volatility < self.vix_threshold["CAUTION_TO_NORMAL"]:
                    self.update_risk_status("NORMAL")
            elif (
                self.risk_status == "RISK"
                and adjusted_volatility < self.vix_threshold["RISK_TO_CAUTION"]
            ):
                self.update_risk_status("CAUTION")

            # 시장 상태 평가
            ma20 = kospi_data["close"].rolling(window=20).mean().iloc[-1]
            ma60 = kospi_data["close"].rolling(window=60).mean().iloc[-1]
            current_close = kospi_data["close"].iloc[-1]

            if current_close > ma20 and ma20 > ma60:
                self.market_condition = "BULL"
            elif current_close < ma20 and ma20 < ma60:
                self.market_condition = "BEAR"
            else:
                self.market_condition = "NORMAL"

            logger.info(f"현재 시장 상태: {self.market_condition}")

        except Exception as e:
            logger.error(f"시장 리스크 평가 중 오류 발생: {e}")

    def update_risk_status(self, new_status):
        """리스크 상태 업데이트 및 로깅"""
        if new_status == self.risk_status:
            return

        old_status = self.risk_status
        self.risk_status = new_status

        logger.warning(f"리스크 상태 변경: {old_status} -> {new_status}")

        # 리스크 상태 변경 로그 기록
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "old_status": old_status,
            "new_status": new_status,
            "reason": "시장 변동성 변화",
        }

        # 로그 파일에 기록
        log_file = f"{self.log_dir}/risk_status_changes.json"

        log_entries = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    log_entries = json.load(f)
            except:
                pass

        log_entries.append(log_entry)

        with open(log_file, "w") as f:
            json.dump(log_entries, f, indent=2)

    def can_trade(self, ticker, order_type, quantity, price=0):
        """주문 가능 여부 확인 (컴플라이언스 체크)

        Args:
            ticker (str): 종목코드
            order_type (str): 주문 유형 ('buy' 또는 'sell')
            quantity (int): 주문 수량
            price (float): 주문 가격

        Returns:
            tuple: (주문 가능 여부, 사유)
        """
        with self.trading_lock:
            # 거래 시간 체크
            if not self.trading_enabled:
                return False, "거래 시간이 아닙니다"

            # 일일 주문 한도 체크
            if self.daily_order_count >= self.max_orders_per_day:
                return False, f"일일 주문 한도({self.max_orders_per_day}회) 초과"

            # 분당 주문 한도 체크
            if len(self.minute_orders) >= self.max_orders_per_minute:
                return False, f"분당 주문 한도({self.max_orders_per_minute}회) 초과"

            # 종목별 주문 한도 체크
            ticker_count = self.ticker_order_count.get(ticker, 0)
            if ticker_count >= self.max_orders_per_ticker:
                return (
                    False,
                    f"종목별 일일 주문 한도({self.max_orders_per_ticker}회) 초과",
                )

            # 매수인 경우에만 추가 체크
            if order_type.lower() == "buy":
                # 계좌 정보 확인
                account_info = self.api_client.get_account_info()
                if not account_info:
                    return False, "계좌 정보를 가져올 수 없습니다"

                # 레버리지 체크
                total_assets = account_info.get("total_evaluated_amount", 0)
                cash_available = account_info.get("available_amount", 0)

                if price == 0:  # 시장가 주문인 경우 현재가로 추정
                    price = self.api_client.get_current_price(ticker)
                    if not price:
                        return False, "현재가를 가져올 수 없습니다"

                # 예상 주문 금액
                order_amount = price * quantity

                # 현재 포지션 + 신규 주문이 최대 레버리지를 초과하는지 확인
                current_positions_value = sum(
                    pos.get("eval_amount", 0)
                    for pos in account_info.get("positions", [])
                )

                new_total_exposure = current_positions_value + order_amount
                max_allowed_exposure = total_assets * self.max_leverage

                if new_total_exposure > max_allowed_exposure:
                    return False, f"최대 레버리지({self.max_leverage}배) 초과"

                # 현금 여유가 있는지 확인
                if order_amount > cash_available:
                    return False, "계좌 잔고 부족"

                # 리스크 상태에 따른 추가 제한
                if self.risk_status == "RISK":
                    # 위험 상태에서는 추가 매수 제한
                    return False, "리스크 상태(RISK)에서는 신규 매수가 제한됩니다"

                # 리스크 상태가 주의 상태이고, 종목이 고변동성이면 제한
                if self.risk_status == "CAUTION":
                    # 최근 변동성 확인
                    try:
                        ohlcv_data = self.api_client.get_ohlcv(ticker, "D", 20)
                        if ohlcv_data is not None and not ohlcv_data.empty:
                            # 20일 변동성 계산
                            returns = ohlcv_data["close"].pct_change().dropna()
                            volatility = returns.std() * np.sqrt(252) * 100

                            # 변동성이 높은 종목(30% 이상)은 제한
                            if volatility > 30:
                                return (
                                    False,
                                    f"주의 상태에서 고변동성 종목({volatility:.1f}%) 매수 제한",
                                )
                    except Exception as e:
                        logger.warning(f"변동성 체크 중 오류: {str(e)}")

            # 모든 조건 통과
            return True, "주문 가능"

    def record_order(self, ticker, order_type):
        """주문 기록 (제한 관리용)

        Args:
            ticker (str): 종목코드
            order_type (str): 주문 유형 ('buy' 또는 'sell')
        """
        with self.trading_lock:
            # 일별 주문 카운터 증가
            self.daily_order_count += 1

            # 종목별 주문 카운터 증가
            self.ticker_order_count[ticker] = self.ticker_order_count.get(ticker, 0) + 1

            # 분당 주문 목록에 추가
            self.minute_orders.append(datetime.now())

            logger.debug(
                f"주문 기록: {ticker} {order_type}, "
                f"일별 {self.daily_order_count}/{self.max_orders_per_day}, "
                f"분당 {len(self.minute_orders)}/{self.max_orders_per_minute}, "
                f"종목별 {self.ticker_order_count.get(ticker, 0)}/{self.max_orders_per_ticker}"
            )

    def calculate_position_size(self, ticker: str, account_balance: float) -> float:
        """변동성 기반 포지션 사이징 계산"""
        try:
            # 최근 가격 데이터 조회
            price_data = self.api_client.get_historical_prices(
                ticker, self.volatility_lookback
            )
            if not price_data:
                return 0

            # 변동성 계산
            returns = pd.Series(price_data["close"]).pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 연간화된 변동성

            # 변동성에 따른 포지션 크기 조정
            position_size = self.max_position_size * (
                0.2 / volatility
            )  # 20% 변동성을 기준으로 조정
            position_size = min(position_size, self.max_position_size)  # 최대 한도 적용

            # 계좌 잔고 기반 최종 포지션 크기 계산
            final_position_size = account_balance * position_size

            logger.info(
                f"종목 {ticker}의 포지션 크기 계산: {final_position_size:,.0f}원 (변동성: {volatility:.2%})"
            )
            return final_position_size

        except Exception as e:
            logger.error(f"포지션 크기 계산 중 오류 발생: {str(e)}")
            return 0

    def calculate_stop_loss(self, ticker: str, entry_price: float) -> float:
        """동적 손절가 계산"""
        try:
            # 최근 가격 데이터로 변동성 계산
            price_data = self.api_client.get_historical_prices(
                ticker, self.volatility_lookback
            )
            if not price_data:
                return entry_price * (1 - self.stop_loss_pct)

            returns = pd.Series(price_data["close"]).pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)

            # 변동성에 따른 손절 비율 조정
            dynamic_stop_loss = min(self.stop_loss_pct, volatility * 2)
            stop_loss_price = entry_price * (1 - dynamic_stop_loss)

            logger.info(
                f"종목 {ticker}의 손절가 계산: {stop_loss_price:,.0f}원 (변동성: {volatility:.2%})"
            )
            return stop_loss_price

        except Exception as e:
            logger.error(f"손절가 계산 중 오류 발생: {str(e)}")
            return entry_price * (1 - self.stop_loss_pct)

    def check_portfolio_risk(self, positions: List[Dict]) -> bool:
        """포트폴리오 리스크 체크"""
        try:
            total_value = sum(pos["current_value"] for pos in positions)
            total_cost = sum(pos["entry_value"] for pos in positions)

            # 일일 손익률 계산
            daily_pnl = (total_value - total_cost) / total_cost

            # 손실 한도 체크
            if daily_pnl < -self.max_daily_loss:
                logger.warning(f"일일 손실 한도 초과: {daily_pnl:.2%}")
                return False

            return True

        except Exception as e:
            logger.error(f"포트폴리오 리스크 체크 중 오류 발생: {str(e)}")
            return False

    def track_daily_loss(self, portfolio_value, date_str=None):
        """일일 손실 추적

        Args:
            portfolio_value (float): 현재 포트폴리오 가치
            date_str (str): 날짜 문자열 (YYYYMMDD 형식), None인 경우 현재 날짜 사용

        Returns:
            bool: 일일 손실 한도 초과 여부
        """
        # 날짜 설정
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        # 일일 손실 데이터 불러오기
        self._load_daily_loss_data()

        # 새로운 날짜인 경우 초기화
        if date_str not in self.daily_loss_tracking:
            # 전날 포트폴리오 가치 가져오기
            yesterday = (
                datetime.strptime(date_str, "%Y%m%d") - timedelta(days=1)
            ).strftime("%Y%m%d")

            if yesterday in self.daily_loss_tracking:
                prev_value = self.daily_loss_tracking[yesterday]["end_value"]
            else:
                # 이전 데이터가 없으면 현재 값 사용
                prev_value = portfolio_value

            self.daily_loss_tracking[date_str] = {
                "start_value": prev_value,
                "current_value": portfolio_value,
                "end_value": portfolio_value,
                "max_loss_pct": 0.0,
            }
        else:
            # 기존 날짜 데이터 업데이트
            self.daily_loss_tracking[date_str]["current_value"] = portfolio_value
            self.daily_loss_tracking[date_str]["end_value"] = portfolio_value

            # 최대 손실 비율 업데이트
            start_value = self.daily_loss_tracking[date_str]["start_value"]
            loss_pct = (portfolio_value / start_value - 1) * 100

            if loss_pct < self.daily_loss_tracking[date_str]["max_loss_pct"]:
                self.daily_loss_tracking[date_str]["max_loss_pct"] = loss_pct

        # 일일 손실 데이터 저장
        self._save_daily_loss_data()

        # 일일 손실 한도 확인
        current_loss_pct = (
            portfolio_value / self.daily_loss_tracking[date_str]["start_value"] - 1
        ) * 100

        # 일일 손실이 한도를 초과하는지 확인
        if current_loss_pct < -self.max_daily_loss * 100:
            logger.warning(
                f"일일 손실 한도 초과: {current_loss_pct:.2f}% (한도: -{self.max_daily_loss*100:.2f}%)"
            )
            return True

        return False

    def _load_daily_loss_data(self):
        """일일 손실 데이터 로드"""
        loss_file = f"{self.log_dir}/daily_loss_tracking.json"

        if os.path.exists(loss_file):
            try:
                with open(loss_file, "r") as f:
                    self.daily_loss_tracking = json.load(f)
            except Exception as e:
                logger.error(f"일일 손실 데이터 로드 오류: {e}")
                self.daily_loss_tracking = {}
        else:
            self.daily_loss_tracking = {}

    def _save_daily_loss_data(self):
        """일일 손실 데이터 저장"""
        loss_file = f"{self.log_dir}/daily_loss_tracking.json"

        try:
            with open(loss_file, "w") as f:
                json.dump(self.daily_loss_tracking, f, indent=2)
        except Exception as e:
            logger.error(f"일일 손실 데이터 저장 오류: {e}")

    def calculate_optimal_portfolio_allocation(self, candidates, account_balance):
        """최적 포트폴리오 배분 계산

        Args:
            candidates (list): 스크리닝된 종목 리스트 (dict 포맷, 'ticker', 'score' 등 포함)
            account_balance (float): 계좌 잔고

        Returns:
            dict: 종목별 최적 투자 금액 및 비중
        """
        # 포트폴리오에 할당할 최대 자금 계산
        max_portfolio_amount = account_balance * self.max_total_position

        # 리스크 상태에 따른 투자 비중 조정
        max_portfolio_amount *= self.position_size_multiplier[self.risk_status]

        # 후보 종목이 없는 경우
        if not candidates or len(candidates) == 0:
            logger.warning("포트폴리오 배분 계산 위한 후보 종목이 없습니다.")
            return {}

        # 스코어 기반 가중치 계산
        total_score = sum(candidate["score"] for candidate in candidates)

        if total_score == 0:
            # 모든 종목의 점수가 0이면 동일 가중치 적용
            weights = [1.0 / len(candidates) for _ in candidates]
        else:
            weights = [candidate["score"] / total_score for candidate in candidates]

        # 최대 보유 종목 수 제한
        max_stocks = min(config.MAX_STOCK_COUNT, len(candidates))

        # 가중치가 높은 순으로 정렬
        sorted_indices = np.argsort(weights)[::-1][:max_stocks]

        # 선택된 종목들의 가중치 재계산
        selected_weights = [weights[i] for i in sorted_indices]
        total_selected_weight = sum(selected_weights)
        normalized_weights = [w / total_selected_weight for w in selected_weights]

        # 종목별 투자 금액 계산
        allocations = {}

        for idx, i in enumerate(sorted_indices):
            ticker = candidates[i]["ticker"]
            weight = normalized_weights[idx]
            amount = max_portfolio_amount * weight

            # 한 종목당 최대 투자 금액 제한
            max_per_stock = account_balance * self.max_position_size
            amount = min(amount, max_per_stock)

            allocations[ticker] = {
                "amount": amount,
                "weight": weight,
                "score": candidates[i]["score"],
            }

        logger.info(
            f"포트폴리오 최적 배분 계산 완료: {len(allocations)}개 종목, 총 투자 금액: {sum(a['amount'] for a in allocations.values()):,.0f}원"
        )

        return allocations

    def generate_risk_report(self, output_file=None):
        """리스크 상태 리포트 생성"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.log_dir}/risk_report_{timestamp}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== 리스크 상태 리포트 ===\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"현재 리스크 상태: {self.risk_status}\n")
            f.write(f"현재 시장 상태: {self.market_condition}\n\n")

            f.write("== 파라미터 설정 ==\n")
            f.write(f"최대 종목별 투자 비중: {self.max_position_size * 100:.1f}%\n")
            f.write(f"최대 총 투자 비중: {self.max_total_position * 100:.1f}%\n")
            f.write(
                f"현재 리스크 계수: {self.position_size_multiplier[self.risk_status]:.2f}\n\n"
            )

            f.write("== 리스크 상태 변경 이력 ==\n")
            risk_log_file = f"{self.log_dir}/risk_status_changes.json"

            if os.path.exists(risk_log_file):
                try:
                    with open(risk_log_file, "r") as log_f:
                        risk_logs = json.load(log_f)

                        # 최근 5개 로그만 표시
                        for log in risk_logs[-5:]:
                            f.write(
                                f"{log['timestamp']} - {log['old_status']} -> {log['new_status']}: {log['reason']}\n"
                            )
                except:
                    f.write("로그 데이터를 불러올 수 없습니다.\n")
            else:
                f.write("로그 데이터가 없습니다.\n")

            f.write("\n== 일일 손실 추적 ==\n")
            self._load_daily_loss_data()

            # 최근 5일 데이터 표시
            sorted_dates = sorted(self.daily_loss_tracking.keys())[-5:]

            for date in sorted_dates:
                data = self.daily_loss_tracking[date]
                daily_return = (data["end_value"] / data["start_value"] - 1) * 100
                f.write(
                    f"{date}: 일일 수익률 {daily_return:.2f}%, 최대 낙폭 {data['max_loss_pct']:.2f}%\n"
                )

        logger.info(f"리스크 리포트 생성 완료: {output_file}")
        return output_file
