"""
trade_db.py — 매매 데이터 영속화 모듈

PostgreSQL(192.168.45.53:5432 / auto_trade_kor)에 매매 이력을 비동기로 저장합니다.

저장 데이터:
  - trades           : 매수/매도 체결 이력 (핵심)
  - screening_scores : 스크리닝 점수 이력
  - agent_decisions  : 에이전트 매수 결정 로그
  - daily_summary    : 일별 성과 요약

사용법:
  db = TradeDatabase()
  await db.connect()
  await db.save_trade_buy(ticker, name, price, qty, strategy, score, regime, conf)
  await db.save_trade_sell(ticker, name, price, qty, buy_price, pnl_amount, pnl_ratio, reason, buy_trade_id)
  await db.close()
"""

import asyncio
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import asyncpg

logger = logging.getLogger("auto_trade.trade_db")

# DB 접속 정보 (환경변수 → 하드코딩 폴백)
import os as _os
DB_CONFIG = {
    "host": _os.getenv("DB_HOST") or "192.168.45.53",
    "port": int(_os.getenv("DB_PORT") or "5432"),
    "database": _os.getenv("DB_NAME") or "auto_trade_kor",
    "user": _os.getenv("DB_USER") or "trader",
    "password": _os.getenv("DB_PASSWORD") or "trader2024",
}


class TradeDatabase:
    """비동기 PostgreSQL 매매 데이터 저장소."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._cfg = {**DB_CONFIG, **(config or {})}
        self._pool: Optional[asyncpg.Pool] = None
        self._enabled = True  # DB 연결 실패 시 False로 설정해 무시

    # ── 연결 관리 ────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """연결 풀 초기화. 실패해도 예외를 던지지 않음 (선택적 기능)."""
        try:
            self._pool = await asyncpg.create_pool(
                **self._cfg,
                min_size=1,
                max_size=5,
                command_timeout=10,
                server_settings={"application_name": "auto_trade_kor"},
            )
            # 연결 확인
            async with self._pool.acquire() as conn:
                ver = await conn.fetchval("SELECT version()")
                logger.info(f"[DB] 연결 성공: {ver[:40]}")
            return True
        except Exception as e:
            logger.error(f"[DB] 연결 실패 (매매는 계속 진행): {e}")
            self._enabled = False
            return False

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("[DB] 연결 종료")

    def _ok(self) -> bool:
        return self._enabled and self._pool is not None

    # ── 매매 체결 저장 ────────────────────────────────────────────────────

    async def save_trade_buy(
        self,
        ticker: str,
        name: str,
        price: float,
        quantity: int,
        strategy: str = "",
        score: float = 0.0,
        market_regime: str = "",
        agent_confidence: float = 0.0,
    ) -> Optional[int]:
        """매수 체결 기록. 반환값: trade_id (이후 SELL 연결용)."""
        if not self._ok():
            return None
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO trades (
                        ticker, name, action, price, quantity,
                        strategy, score, agent_confidence, market_regime
                    ) VALUES ($1,$2,'BUY',$3,$4,$5,$6,$7,$8)
                    RETURNING id
                    """,
                    ticker, name, int(price), quantity,
                    strategy, float(score), float(agent_confidence), market_regime,
                )
                trade_id = row["id"]
                logger.debug(f"[DB] 매수 저장: {ticker} {quantity}주 @{price:,.0f} (id={trade_id})")
                return trade_id
        except Exception as e:
            logger.error(f"[DB] 매수 저장 실패 {ticker}: {e}")
            return None

    async def save_trade_sell(
        self,
        ticker: str,
        name: str,
        price: float,
        quantity: int,
        buy_price: float,
        pnl_amount: float,
        pnl_ratio: float,
        reason: str = "",
        buy_trade_id: Optional[int] = None,
        market_regime: str = "",
        strategy: str = "",
    ) -> Optional[int]:
        """매도 체결 기록. Bug #5 fix (2026-04-24): 1회 재시도 + 실패 시 loud warning.
        과거 Overnight 버그 시기 16건 SELL 레코드 유실된 전례로 방어 강화."""
        if not self._ok():
            logger.warning(
                f"[DB] 매도 저장 스킵 {ticker}: DB 연결 비활성 "
                f"(pnl={pnl_ratio:+.2%} amount={pnl_amount:+,.0f})"
            )
            return None
        last_err: Optional[str] = None
        for attempt in range(2):   # 최초 시도 + 1회 재시도
            try:
                async with self._pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO trades (
                            ticker, name, action, price, quantity,
                            buy_price, pnl_amount, pnl_ratio,
                            reason, market_regime, strategy, buy_trade_id
                        ) VALUES ($1,$2,'SELL',$3,$4,$5,$6,$7,$8,$9,$10,$11)
                        RETURNING id
                        """,
                        ticker, name, int(price), quantity,
                        int(buy_price), int(pnl_amount), float(pnl_ratio),
                        reason, market_regime, strategy, buy_trade_id,
                    )
                    trade_id = row["id"]
                    logger.info(
                        f"[DB] 매도 저장: {ticker} {pnl_ratio:+.2%} "
                        f"({pnl_amount:+,.0f}원) (id={trade_id}"
                        f"{', 재시도' if attempt else ''})"
                    )
                    return trade_id
            except Exception as e:
                last_err = str(e)
                logger.warning(
                    f"[DB] 매도 저장 실패 {ticker} (시도 {attempt+1}/2): {e}"
                )
                await asyncio.sleep(0.5)
        # 최종 실패 — CRITICAL: 포지션 기록 유실, 통계 왜곡
        logger.error(
            f"[DB] 🚨 매도 저장 최종 실패 {ticker} | pnl={pnl_ratio:+.2%} "
            f"amount={pnl_amount:+,.0f}원 reason={reason} | err={last_err}"
        )
        return None

    # ── 스크리닝 결과 저장 ────────────────────────────────────────────────

    async def save_screening(
        self,
        results: List[Dict],
        session: str = "morning",
    ) -> int:
        """스크리닝 결과 일괄 저장. 반환값: 저장 건수."""
        if not self._ok() or not results:
            return 0
        rows = []
        for r in results:
            rows.append((
                session,
                r.get("ticker") or r.get("code", ""),
                r.get("name", ""),
                float(r.get("total_score", 0) or 0),
                float(r.get("tech_score", 0) or 0),
                float(r.get("volume_score", 0) or 0),
                float(r.get("supply_score", 0) or 0),
                float(r.get("news_score", 0) or 0),
                float(r.get("overnight_bonus", 0) or 0),
                bool(r.get("selected", False)),
            ))
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO screening_scores
                        (session, ticker, name, total_score,
                         tech_score, volume_score, supply_score,
                         news_score, overnight_bonus, selected)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                    """,
                    rows,
                )
            logger.debug(f"[DB] 스크리닝 {len(rows)}건 저장 (세션: {session})")
            return len(rows)
        except Exception as e:
            logger.error(f"[DB] 스크리닝 저장 실패: {e}")
            return 0

    # ── 에이전트 결정 저장 ────────────────────────────────────────────────

    async def save_agent_decision(
        self,
        ticker: str,
        name: str,
        decision: str,
        strategy: str = "",
        alpha_confidence: float = 0.0,
        alpha_score: float = 0.0,
        market_regime: str = "",
        risk_status: str = "",
        skip_reason: str = "",
        executed: bool = False,
        trade_id: Optional[int] = None,
    ) -> None:
        """에이전트 매수/스킵 결정 기록."""
        if not self._ok():
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO agent_decisions
                        (ticker, name, decision, skip_reason,
                         strategy, alpha_confidence, alpha_score,
                         market_regime, risk_status, executed, trade_id)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    """,
                    ticker, name, decision, skip_reason,
                    strategy, float(alpha_confidence), float(alpha_score),
                    market_regime, risk_status, executed, trade_id,
                )
        except Exception as e:
            logger.error(f"[DB] 에이전트 결정 저장 실패 {ticker}: {e}")

    # ── 일별 요약 저장 ────────────────────────────────────────────────────

    async def save_daily_summary(
        self,
        total_trades: int,
        buy_trades: int,
        sell_trades: int,
        win_trades: int,
        loss_trades: int,
        gross_pnl: float,
        market_regime: str = "",
        risk_status: str = "",
        starting_equity: float = 0,
        ending_equity: float = 0,
        screened_count: int = 0,
        selected_count: int = 0,
        trade_date: Optional[date] = None,
    ) -> None:
        """일별 성과 요약 Upsert (당일 재실행 시 업데이트)."""
        if not self._ok():
            return
        td = trade_date or date.today()
        win_rate = round(win_trades / sell_trades, 3) if sell_trades > 0 else 0.0
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO daily_summary
                        (trade_date, total_trades, buy_trades, sell_trades,
                         win_trades, loss_trades, gross_pnl, win_rate,
                         market_regime, risk_status,
                         starting_equity, ending_equity,
                         screened_count, selected_count)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                    ON CONFLICT (trade_date) DO UPDATE SET
                        total_trades   = EXCLUDED.total_trades,
                        buy_trades     = EXCLUDED.buy_trades,
                        sell_trades    = EXCLUDED.sell_trades,
                        win_trades     = EXCLUDED.win_trades,
                        loss_trades    = EXCLUDED.loss_trades,
                        gross_pnl      = EXCLUDED.gross_pnl,
                        win_rate       = EXCLUDED.win_rate,
                        market_regime  = EXCLUDED.market_regime,
                        risk_status    = EXCLUDED.risk_status,
                        starting_equity = EXCLUDED.starting_equity,
                        ending_equity  = EXCLUDED.ending_equity,
                        screened_count = EXCLUDED.screened_count,
                        selected_count = EXCLUDED.selected_count
                    """,
                    td, total_trades, buy_trades, sell_trades,
                    win_trades, loss_trades, int(gross_pnl), win_rate,
                    market_regime, risk_status,
                    int(starting_equity), int(ending_equity),
                    screened_count, selected_count,
                )
            logger.info(f"[DB] 일별 요약 저장: {td} | 매매 {total_trades}건 | 손익 {gross_pnl:+,.0f}원")
        except Exception as e:
            logger.error(f"[DB] 일별 요약 저장 실패: {e}")

    # ── 조회 메서드 ───────────────────────────────────────────────────────

    async def get_today_trades(self) -> List[Dict]:
        """당일 매매 내역 조회."""
        if not self._ok():
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM trades WHERE trade_date = CURRENT_DATE ORDER BY executed_at"
                )
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[DB] 당일 매매 조회 실패: {e}")
            return []

    async def get_recent_performance(self, days: int = 30) -> List[Dict]:
        """최근 N일 일별 성과 조회."""
        if not self._ok():
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM daily_summary
                    WHERE trade_date >= CURRENT_DATE - $1::INTEGER
                    ORDER BY trade_date DESC
                    """,
                    days,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[DB] 성과 조회 실패: {e}")
            return []

    async def get_strategy_stats(self, days: int = 30) -> List[Dict]:
        """전략별 승률/손익 통계."""
        if not self._ok():
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        strategy,
                        COUNT(*) AS total,
                        SUM(CASE WHEN pnl_ratio > 0 THEN 1 ELSE 0 END) AS wins,
                        ROUND(AVG(pnl_ratio)::numeric * 100, 2) AS avg_pnl_pct,
                        SUM(pnl_amount) AS total_pnl
                    FROM trades
                    WHERE action = 'SELL'
                      AND trade_date >= CURRENT_DATE - $1::INTEGER
                      AND strategy IS NOT NULL
                    GROUP BY strategy
                    ORDER BY total_pnl DESC NULLS LAST
                    """,
                    days,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[DB] 전략 통계 조회 실패: {e}")
            return []
