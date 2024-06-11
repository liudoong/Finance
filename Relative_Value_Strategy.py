#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:53:41 2024

@author: dliu
"""

class CDS:
    def __init__(self, issuer, spread, recovery_rate):
        self.issuer = issuer
        self.spread = spread
        self.recovery_rate = recovery_rate

class RelativeValueCreditStrategy:
    def __init__(self):
        self.portfolio = []

    def buy_cds(self, cds):
        self.portfolio.append(cds)

    def sell_cds(self, issuer):
        self.portfolio = [cds for cds in self.portfolio if cds.issuer != issuer]

    def execute_strategy(self, cds_list):
        # Sort CDS by spread
        sorted_cds = sorted(cds_list, key=lambda cds: cds.spread, reverse=True)

        # Buy the CDS with the highest spread
        self.buy_cds(sorted_cds[0])

        # If portfolio is too large, sell the CDS with the lowest spread
        if len(self.portfolio) > 10:
            self.sell_cds(sorted(self.portfolio, key=lambda cds: cds.spread)[0].issuer)

        # Consider wrong way risk
        for cds in self.portfolio:
            if cds.recovery_rate < 0.4:
                self.sell_cds(cds.issuer)

# Example usage:
strategy = RelativeValueCreditStrategy()
cds_list = [CDS('Issuer1', 100, 0.5), CDS('Issuer2', 200, 0.6), CDS('Issuer3', 150, 0.3)]
for cds in cds_list:
    strategy.execute_strategy(cds_list)