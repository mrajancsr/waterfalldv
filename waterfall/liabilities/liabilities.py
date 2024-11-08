from __future__ import annotations
from waterfall.input import PoolInfo
from waterfall.asset.asset import AssetCashFlow
from dataclasses import dataclass, field
import numpy as np

import pandas as pd


class NotFinishedException(Exception):
    pass


@dataclass
class LiabilitiesCashFlow:
    asset_cf: AssetCashFlow
    pool_info: PoolInfo = field(init=False)
    class_a_beginning_principal_balance: float = field(init=False)
    class_b_beginning_principal_balance: float = field(init=False)
    available_funds: np.ndarray = field(init=False)
    total_principal_due: np.ndarray = field(init=False)
    cumulative_principal_due: np.ndarray = field(init=False)
    sf_amount_due: np.ndarray = field(init=False)
    sf_amount_paid: np.ndarray = field(init=False)
    sf_short_fall: np.ndarray = field(init=False)
    sf_available_funds: np.ndarray = field(init=False)
    class_a_interest_due: np.ndarray = field(init=False)
    class_a_interest_paid: np.ndarray = field(init=False)
    class_a_interest_short_fall: np.ndarray = field(init=False)
    class_a_interest_remaining_available_funds: np.ndarray = field(init=False)
    class_a_principal_due: np.ndarray = field(init=False)
    class_a_principal_paid: np.ndarray = field(init=False)
    class_a_principal_short_fall: np.ndarray = field(init=False)
    class_a_ending_principal_balance: np.ndarray = field(init=False)
    class_a_principal_remaining_available_funds: np.ndarray = field(init=False)
    class_b_principal_due: np.ndarray = field(init=False)
    class_b_principal_paid: np.ndarray = field(init=False)
    class_b_principal_short_fall: np.ndarray = field(init=False)
    class_b_ending_principal_balance: np.ndarray = field(init=False)
    class_b_principal_remaining_available_funds: np.ndarray = field(init=False)

    # reserve account beginning balance at month end
    ra_beg_balance: np.ndarray = field(init=False)
    ra_end_balance: np.ndarray = field(init=False)
    ra_account_draw: np.ndarray = field(init=False)
    ra_reserve_contribution_amount: np.ndarray = field(init=False)
    ra_target_reseserve_amount: np.ndarray = field(init=False)
    ra_collection_account_balance_after_class_b_principal: np.ndarray = field(
        init=False
    )
    finish: bool = False

    def __post_init__(self):
        self.pool_info = self.asset_cf.pool_info
        self.class_a_beginning_principal_balance = 85340000
        self.class_b_beginning_principal_balance = 16411000
        self.available_funds = self.asset_cf.available_funds.copy()
        self.total_principal_due = self.asset_cf.principal_due.copy()
        self.cumulative_principal_due = self.asset_cf.principal_due.cumsum()
        self.sf_amount_due = np.zeros(self.pool_info.maturity + 1)
        self.sf_amount_paid = np.zeros(self.pool_info.maturity + 1)
        self.sf_short_fall = np.zeros(self.pool_info.maturity + 1)
        self.sf_available_funds = np.zeros(self.pool_info.maturity + 1)

        # class a cashflows
        self.class_a_interest_due = np.zeros(self.pool_info.maturity + 1)
        self.class_a_interest_paid = np.zeros(self.pool_info.maturity + 1)
        self.class_a_interest_short_fall = np.zeros(self.pool_info.maturity + 1)
        self.class_a_interest_remaining_available_funds = np.zeros(
            self.pool_info.maturity + 1
        )
        self.class_a_principal_due = np.zeros(self.pool_info.maturity + 1)
        self.class_a_principal_paid = np.zeros(self.pool_info.maturity + 1)
        self.class_a_principal_short_fall = np.zeros(self.pool_info.maturity + 1)
        self.class_a_ending_principal_balance = np.zeros(self.pool_info.maturity + 1)
        self.class_a_principal_remaining_available_funds = np.zeros(
            self.pool_info.maturity + 1
        )

        # class b cashflows
        self.class_b_interest_due = np.zeros(self.pool_info.maturity + 1)
        self.class_b_interest_paid = np.zeros(self.pool_info.maturity + 1)
        self.class_b_interest_short_fall = np.zeros(self.pool_info.maturity + 1)
        self.class_b_interest_remaining_available_funds = np.zeros(
            self.pool_info.maturity + 1
        )
        self.class_b_principal_due = np.zeros(self.pool_info.maturity + 1)
        self.class_b_principal_paid = np.zeros(self.pool_info.maturity + 1)
        self.class_b_principal_short_fall = np.zeros(self.pool_info.maturity + 1)
        self.class_b_ending_principal_balance = np.zeros(self.pool_info.maturity + 1)
        self.class_b_principal_remaining_available_funds = np.zeros(
            self.pool_info.maturity + 1
        )

        # reserve accounts
        self.ra_beg_balance = np.zeros(self.pool_info.maturity + 1)
        self.ra_end_balance = np.zeros(self.pool_info.maturity + 1)
        self.ra_account_draw = np.zeros(self.pool_info.maturity + 1)
        self.ra_reserve_contribution_amount = np.zeros(self.pool_info.maturity + 1)
        self.ra_target_reseserve_amount = np.zeros(self.pool_info.maturity + 1)
        self.ra_collection_account_balance_after_class_b_principal = np.zeros(
            self.pool_info.maturity + 1
        )

    def initialize_ending_principal_balance(self):
        """Initializes the ending principal balance for both tranches at time t = 0"""
        self.class_a_ending_principal_balance[0] = (
            self.class_a_beginning_principal_balance
        )
        self.class_b_ending_principal_balance[0] = (
            self.class_b_beginning_principal_balance
        )

    def initialize_target_reserve_amount(self):
        raise NotImplementedError("Yet to Implement")

    def build_waterfall_engine(self):
        """Builds the WaterFall Engine based on the prospectus"""
        sf = self.pool_info.servicing_fee
        sr = self.pool_info.servicing_fee_short_fall_rate

        # reserve account rate
        re = self.pool_info.eligible_investment_rate
        rp = self.pool_info.target_reserve_percentage

        # interest rate corresponding to the two notes
        ra = self.pool_info.class_a_interest
        rb = self.pool_info.class_b_interest

        self.initialize_ending_principal_balance()
        # self.initialize_target_reserve_amount()

        for t in range(1, self.pool_info.maturity + 1):
            # servicing Fee calculations
            self.sf_amount_due[t] = (sf / 12) * self.asset_cf.pool_balance[
                t - 1
            ] + self.sf_short_fall[t - 1] * (1 + sr / 12)
            self.sf_amount_paid[t] = min(
                self.asset_cf.available_funds[t], self.sf_amount_due[t]
            )
            self.sf_short_fall[t] = self.sf_amount_due[t] - self.sf_amount_paid[t]
            self.class_b_interest_short_fall[t - 1] = (
                self.class_b_interest_due[t - 1] - self.class_b_interest_paid[t - 1]
            )
            self.class_b_interest_due[t] = (
                rb / 12
            ) * self.class_b_ending_principal_balance[
                t - 1
            ] + self.class_b_interest_short_fall[
                t - 1
            ] * (
                1 + rb / 12
            )

            self.class_b_interest_remaining_available_funds[t - 1] = (
                self.class_a_interest_remaining_available_funds[t - 1]
                - self.class_b_interest_paid[t - 1]
            )

            self.class_b_interest_paid[t] = min(
                self.class_a_interest_remaining_available_funds[t],
                self.class_b_interest_due[t],
            )
            # class a principal account calc
            self.class_a_principal_paid[t] = min(
                self.class_b_interest_remaining_available_funds[t],
                self.class_a_principal_due[t],
            )
            self.class_a_principal_remaining_available_funds[t - 1] = (
                self.class_b_interest_remaining_available_funds[t - 1]
                - self.class_a_principal_paid[t - 1]
            )
            self.class_a_principal_due[t] = min(
                self.class_a_ending_principal_balance[t - 1],
                self.class_a_principal_short_fall[t - 1] + self.total_principal_due[t],
            )
            self.class_a_ending_principal_balance[t] = (
                self.class_a_ending_principal_balance[t - 1]
                - self.class_a_principal_paid[t]
            )
            self.class_a_principal_paid[t] = min(
                self.class_b_interest_remaining_available_funds[t],
                self.class_a_principal_due[t],
            )

            # class b principal account calc
            self.class_b_principal_remaining_available_funds[t - 1] = (
                self.class_a_principal_remaining_available_funds[t - 1]
                - self.class_b_principal_paid[t - 1]
            )
            self.class_b_ending_principal_balance[t] = (
                self.class_b_ending_principal_balance[t - 1]
                - self.class_b_principal_paid[t]
            )
            self.class_b_principal_due[t] = min(
                self.class_b_ending_principal_balance[t - 1],
                max(
                    0,
                    self.cumulative_principal_due[t]
                    - max(
                        self.class_a_beginning_principal_balance,
                        self.cumulative_principal_due[t - 1],
                    ),
                )
                + self.class_b_principal_short_fall[t - 1],
            )
            self.class_b_principal_paid[t] = min(
                self.class_a_principal_remaining_available_funds[t],
                self.class_b_principal_due[t],
            )
            # reserve accounts
            self.ra_account_draw[t - 1] = max(
                0,
                self.ra_beg_balance[t - 1]
                - self.class_b_principal_remaining_available_funds[t - 1],
            )
            self.ra_reserve_contribution_amount[t - 1] = min(
                self.ra_collection_account_balance_after_class_b_principal[t - 1],
                self.ra_target_reseserve_amount[t - 1]
                - self.ra_beg_balance[t - 1]
                + self.ra_account_draw[t - 1],
            )
            self.ra_end_balance[t - 1] = (
                self.ra_beg_balance[t - 1]
                - self.ra_account_draw[t - 1]
                + self.ra_reserve_contribution_amount[t - 1]
            )
            self.ra_beg_balance[t] = self.ra_end_balance[t - 1] * (1 + re / 12)
            # servicng fee
            self.sf_available_funds[t] = (
                self.asset_cf.available_funds[t]
                - self.sf_amount_paid[t]
                + self.ra_beg_balance[t]
            )

        self.finish = True

    def build_waterfall_df(self):
        if not self.finish:
            raise NotFinishedException(
                "The WaterFall building process has not been completed"
            )
        # Building the pandas dataframe to display results
        data = {
            "Available Funds": self.available_funds,
            "Total Principal Due": self.total_principal_due,
            "Class A Beginning Principal Balance": self.class_a_beginning_principal_balance,
            "Class B Beginning Principal Balance": self.class_b_beginning_principal_balance,
            "Cumulative Principal Due": self.cumulative_principal_due,
        }
        df = pd.DataFrame(data)
        df.loc[1:, "Class A Beginning Principal Balance"] = None
        df.loc[1:, "Class B Beginning Principal Balance"] = None
        self.loan_info = df.round(3)

        index = pd.MultiIndex.from_tuples(
            [
                ("Servicing Fee", "Amount Due"),
                ("Servicing Fee", "Amount Paid"),
                ("Servicing Fee", "ShortFall"),
                ("Servicing Fee", "Available Funds"),
                ("Class A Interest", "Interest Due"),
                ("Class A Interest", "Interest Paid"),
                ("Class A Interest", "Interest ShortFall"),
                ("Class A Interest", "Remaining Available Funds"),
                ("Class B Interest", "Interest Due"),
                ("Class B Interest", "Interest Paid"),
                ("Class B Interest", "Interest ShortFall"),
                ("Class B Interest", "Remaining Available Funds"),
                ("Class A Principal", "Principal Due"),
                ("Class A Principal", "Principal Paid"),
                ("Class A Principal", "Principal ShortFall"),
                ("Class A Principal", "Ending Principal Balance"),
                ("Class A Principal", "Remaining Available Funds"),
                ("Class B Principal", "Principal Due"),
                ("Class B Principal", "Principal Paid"),
                ("Class B Principal", "Principal ShortFall"),
                ("Class B Principal", "Ending Principal Balance"),
                ("Class B Principal", "Remaining Available Funds"),
                ("Reserve Account", "Beg Balance(at month end)"),
                (
                    "Reserve Account",
                    "Collection Account Balance after class B Principal",
                ),
                ("Reserve Account", "Account Draw(in current period)"),
                ("Reserve Account", "Target Reserve Amount"),
                ("Reserve Account", "Reserve Contribution Amount"),
                ("Reserve Account", "Ending Reserve Balance"),
            ]
        )
        df = pd.DataFrame(
            np.column_stack(
                [
                    self.sf_amount_due,
                    self.sf_amount_paid,
                    self.sf_short_fall,
                    self.sf_available_funds,
                    self.class_a_interest_due,
                    self.class_a_interest_paid,
                    self.class_a_interest_short_fall,
                    self.class_a_interest_remaining_available_funds,
                    self.class_b_interest_due,
                    self.class_b_interest_paid,
                    self.class_b_interest_short_fall,
                    self.class_b_interest_remaining_available_funds,
                    self.class_a_principal_due,
                    self.class_a_principal_paid,
                    self.class_a_principal_short_fall,
                    self.class_a_ending_principal_balance,
                    self.class_a_principal_remaining_available_funds,
                    self.class_b_principal_due,
                    self.class_b_principal_paid,
                    self.class_b_principal_short_fall,
                    self.class_b_ending_principal_balance,
                    self.class_b_principal_remaining_available_funds,
                    self.ra_beg_balance,
                    self.ra_collection_account_balance_after_class_b_principal,
                    self.ra_account_draw,
                    self.ra_target_reseserve_amount,
                    self.ra_reserve_contribution_amount,
                    self.ra_end_balance,
                ]
            ),
            columns=index,
        )
        self.waterfall = df.round(3)


def main():

    np.set_printoptions(suppress=True)
    acf = AssetCashFlow(PoolInfo())
    acf.build_asset_side_cashflow()
    lcf = LiabilitiesCashFlow(acf)
    lcf.build_waterfall_engine()
    lcf.build_waterfall_df()
    acf.asset = acf.asset.round(3)
    acf.asset.index.name = "t"
    acf.asset.to_csv("asset.csv")
    lcf.loan_info.to_csv("loan_info.csv")
    lcf.waterfall.to_csv("pro-rata-liabilities.csv")


if __name__ == "__main__":
    main()
