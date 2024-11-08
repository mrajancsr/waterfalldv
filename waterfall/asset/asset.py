from __future__ import annotations
from waterfall.input import PoolInfo
from dataclasses import dataclass, field
import numpy as np
import math
import pandas as pd


@dataclass
class AssetCashFlow:
    pool_info: PoolInfo
    pool_balance: np.ndarray = field(init=False)
    current_loans_remaining: np.ndarray = field(init=False)
    fully_prepaying: np.ndarray = field(init=False)
    scheduled_interest: np.ndarray = field(init=False)
    scheduled_principal: np.ndarray = field(init=False)
    prepaid_principal: np.ndarray = field(init=False)
    total_principal: np.ndarray = field(init=False)
    principal_loan_balance: np.ndarray = field(init=False)
    defaulted_balances: np.ndarray = field(init=False)
    recoveries: np.ndarray = field(init=False)
    available_funds: np.ndarray = field(init=False)
    principal_due: np.ndarray = field(init=False)
    L: np.ndarray = field(init=False)
    nD: np.ndarray = field(init=False)
    l: np.ndarray = field(init=False)

    def __post_init__(self):
        self.pool_balance = np.zeros(self.pool_info.maturity + 1)
        self.current_loans_remaining = np.zeros(self.pool_info.maturity + 1)
        self.fully_prepaying = np.zeros(self.pool_info.maturity + 1)
        self.scheduled_interest = np.zeros(self.pool_info.maturity + 1)
        self.scheduled_principal = np.zeros(self.pool_info.maturity + 1)
        self.prepaid_principal = np.zeros(self.pool_info.maturity + 1)
        self.total_principal = np.zeros(self.pool_info.maturity + 1)
        self.principal_loan_balance = np.zeros(self.pool_info.maturity + 1)
        self.defaulted_balances = np.zeros(self.pool_info.maturity + 1)
        self.recoveries = np.zeros(self.pool_info.maturity + 1)
        self.available_funds = np.zeros(self.pool_info.maturity + 1)
        self.principal_due = np.zeros(self.pool_info.maturity + 1)
        # cumulative non normalized account space loss curve
        self.L = np.zeros(self.pool_info.maturity + 1)
        # marginal_non_normalized_account_space_loss
        self.l = np.zeros(self.pool_info.maturity + 1)
        # marginal_normalized_account_space_loss
        self.nD = np.zeros(self.pool_info.maturity + 1)

    def build_normalized_loss_curves(self):
        self.L[0] = self.credit_loss_cdf(0) * self.pool_info.num_loans

        for t in range(1, self.pool_info.maturity + 1):
            self.L[t] = self.credit_loss_cdf(t) * self.pool_info.num_loans
            self.l[t] = self.L[t] - self.L[t - 1]

        for t in range(1, self.pool_info.maturity + 1):
            self.nD[t] = (
                self.l[t] * self.pool_info.expected_loss / (self.L[-1] - self.L[0])
            )

    def _cumulative_prepayment_curve(self, t: int) -> float:
        """Computes the cumulative prepayment curve G(t)

        Parameters
        ----------
        t : int
            given in months, is the time corresponding to that month

        Returns
        -------
        float
            computes cumulative prepayment curve
        """
        top = self.pool_info.inflection_point
        a = self.pool_info.cumulative_prepayment_curve_slope
        if t < top:
            return a * t * t * 0.5
        if top <= t <= self.pool_info.maturity:
            return a * top * top * 0.5 + (t - top) * a * top

    def credit_loss_cdf(self, t):
        """Cumulative distribution function for credit loss curve
        assumes the curve is a logistic curve.
        parameters obtained from elements from structured finance by sylvaian raynes.

        Parameters
        ----------
        t : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        a = 0.1
        b = 1
        c = 0.1
        t0 = 55

        return a / (1 + b * np.exp(-c * (t - t0)))

    def initialize_pool_balance(self):
        self.pool_balance[0] = self.pool_info.init_pool_balance

    def build_fully_prepaying(self):
        """npt = G(t) - G(t-1)"""
        for t in range(1, self.pool_info.maturity):
            self.fully_prepaying[t] = self._cumulative_prepayment_curve(
                t
            ) - self._cumulative_prepayment_curve(t - 1)

    def initialize_current_loan_remaining(self):
        self.current_loans_remaining[0] = self.pool_info.num_loans

    def build_pool_balance(self):
        self.initialize_pool_balance()
        for t in range(1, self.pool_info.maturity + 1):
            self.pool_balance[t] = (
                self.pool_balance[t - 1]
                - self.defaulted_balances[t]
                - self.prepaid_principal[t]
                - self.scheduled_principal[t]
            )

    def build_current_loans_remaining(self):
        self.initialize_current_loan_remaining()
        for t in range(1, self.pool_info.maturity + 1):
            self.current_loans_remaining[t] = (
                self.current_loans_remaining[t - 1]
                - self.nD[t]
                - self.fully_prepaying[t]
            )

    def compute_beginning_balance(self, t: int):
        """computes the beginning balance at beginning of each period

        Parameters
        ----------
        t : int
            _description_

        Returns
        -------
        _type_
            _description_
        """
        m = self.pool_info.periodic_payment
        r = self.pool_info.periodic_coupon
        maturity = self.pool_info.maturity
        return (m / r) * (1 - np.pow(1 + r, t - maturity))

    def build_balance_and_recoveries(self):
        m = self.pool_info.periodic_payment
        r = self.pool_info.periodic_coupon
        wam = self.pool_info.wam
        self.principal_loan_balance[0] = self.compute_beginning_balance(0)
        for t in range(1, self.pool_info.maturity + 1):
            self.principal_loan_balance[t] = self.compute_beginning_balance(t)
            self.defaulted_balances[t] = (
                self.nD[t] * (m / r) * (1 - math.pow(1 + r, t - 1 - wam))
            )
        # assume recoveries are delayed until the 4th period
        for t in range(4, self.pool_info.maturity + 1):
            self.recoveries[t] = (1 - self.pool_info.lgd) * self.defaulted_balances[
                t - 3
            ]

    def build_scheduled_interest_and_principal(self):
        r = self.pool_info.periodic_coupon
        m = self.pool_info.periodic_payment
        wam = self.pool_info.wam
        for t in range(1, self.pool_info.maturity + 1):
            self.scheduled_interest[t] = (
                (self.current_loans_remaining[t - 1] - self.nD[t])
                * r
                * self.principal_loan_balance[t - 1]
            )
            self.scheduled_principal[t] = (
                (self.current_loans_remaining[t - 1] - self.nD[t])
                * (m / r)
                * (math.pow(1 + r, t - wam) - math.pow(r + 1, t - 1 - wam))
            )
            self.prepaid_principal[t] = (self.fully_prepaying[t] * m / r) * (
                1 - math.pow(1 + r, t - wam)
            )
            self.total_principal[t] = (
                self.prepaid_principal[t] + self.scheduled_principal[t]
            )

    def build_current_collections(self):
        for t in range(1, self.pool_info.maturity + 1):
            self.available_funds[t] = (
                self.scheduled_interest[t]
                + self.total_principal[t]
                + self.recoveries[t]
            )
            self.principal_due[t] = (
                self.scheduled_principal[t]
                + self.prepaid_principal[t]
                + self.defaulted_balances[t]
            )

    def build_asset_side_cashflow(self):
        self.build_fully_prepaying()
        self.build_normalized_loss_curves()
        self.build_balance_and_recoveries()
        self.build_current_loans_remaining()
        self.build_scheduled_interest_and_principal()
        self.build_pool_balance()
        self.build_current_collections()
        data = {
            "Pool Balance": self.pool_balance,
            "Current Loans Remaining": self.current_loans_remaining,
            "Fully Prepaying": self.fully_prepaying,
            "Scheduled Interest": self.scheduled_interest,
            "Scheduled Principal": self.scheduled_principal,
            "Prepaid Principal": self.prepaid_principal,
            "Total Principal": self.total_principal,
            "Principal Loan Balance": self.principal_loan_balance,
            "Defaulted Balances": self.defaulted_balances,
            "Recoveries": self.recoveries,
            "Available Funds": self.available_funds,
            "Principal Due": self.principal_due,
        }
        self.asset = pd.DataFrame(data)


def main():

    pd.set_option("display.float_format", lambda x: "%.3f" % x)

    acf = AssetCashFlow(PoolInfo())
    acf.build_asset_side_cashflow()
    acf.asset


if __name__ == "__main__":
    main()
