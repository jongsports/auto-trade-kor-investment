"""거래 비용 모델 — 라이브 손익 추정과 백테스트가 함께 쓴다."""
import config


def round_trip_cost(buy_amount: float, sell_amount: float) -> float:
    """매수·매도 수수료와 매도 거래세의 합."""
    return (buy_amount * config.COMMISSION_RATE
            + sell_amount * (config.COMMISSION_RATE + config.SELL_TAX_RATE))


def net_pnl(buy_price: float, sell_price: float, quantity: int) -> tuple:
    """비용을 뺀 손익 (금액, 매수금액 대비 비율)."""
    if buy_price <= 0 or sell_price <= 0 or quantity <= 0:
        return 0.0, 0.0
    buy_amount = buy_price * quantity
    sell_amount = sell_price * quantity
    amount = sell_amount - buy_amount - round_trip_cost(buy_amount, sell_amount)
    return amount, amount / buy_amount
