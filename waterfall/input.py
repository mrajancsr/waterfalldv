from dataclasses import dataclass, field
import math


@dataclass
class PoolInfo:
    """Initial Pool Information that contains loan level detail that assumes monthly prepayment
    num_loans: int
        the number of loans in pool at time 0
    wac: float
        weighted average coupons
    wam: int
        weighted average maturity in months
    init_pool_balance: float
        the initial pool balance at time 0
    expected loss: float
        the expected loss as a percentage of initial pool balance
    cumulative_default_rate: float
        the cumulative account space default rate
    lgd: float
        loss given default
    periodic_coupon: float
        wac / 12 (annualized)
    """

    num_loans: int = 16378
    wac: float = 0.12
    wam: int = 39
    cumulative_default_rate: float = 0.01
    lgd: float = 0.5
    init_balance_per_loan: float = (
        6212.663328855782  # (note balance for two notes / num_loans)
    )
    maturity: int = 60
    periodic_coupon: float = field(init=False)
    expected_loss: float = field(init=False)
    init_pool_balance: float = field(init=False)
    periodic_payment: float = field(init=False)

    # prepayments
    initial_recovery: float = 0.2  # assume 20% recovery
    # before this point, cumulative prepayment curve is rising
    # after this point, it becomes steady
    inflection_point: int = 20
    # the a parameter
    cumulative_prepayment_curve_slope: float = field(init=False)
    periodic_payment: float = field(init=False)
    time_decay: int = 3
    total_loans_prepay_on_recovery: float = field(init=False)

    # Liabilities
    servicing_fee: float = 0.01
    servicing_fee_short_fall_rate: float = 0.001
    class_a_interest: float = 0.03
    class_b_interest: float = 0.06
    # advance rate assume 80%
    alpha: float = 0.8

    # reserve accounts assume 0.5% for both investment adn target reserve percentage
    eligible_investment_rate: float = 0.005
    target_reserve_percentage: float = 0.005

    def __post_init__(self):
        self.periodic_coupon = self.wac / 12
        self.expected_loss = self.cumulative_default_rate * self.num_loans
        self.init_pool_balance = self.num_loans * self.init_balance_per_loan
        self.periodic_payment = (self.init_balance_per_loan * self.periodic_coupon) / (
            1 - math.pow(1 + self.periodic_coupon, -self.maturity)
        )
        self.total_loans_prepay_on_recovery = self.initial_recovery * self.num_loans
        self.cumulative_prepayment_curve_slope = self.total_loans_prepay_on_recovery / (
            math.pow(self.inflection_point, 2) * 0.5
            + (self.maturity - self.inflection_point) * self.inflection_point
        )
