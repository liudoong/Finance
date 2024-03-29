% Consider a cap with these parameters

Settle     = 'Jan-21-2008'
Maturity   = 'mar-21-2011'
Strike     = 0.0690
Reset      = 4
Pricipal   = 1000
Basis      = 0

% The caplets for this example would fall in

capletDates = cfdates(Settle, Maturity, Reset, Basis);
datestr(capletDates)

MarketStrike = [0.0590; 0.0790];

% In the best case, lookup the market volatilities for caplets with a
% strike = 0.0690, and maturities in each reset date listed, but the
% likelihood of finding these exact instruments is low. As a consequence,
% use data that is available in the market and interpolate to find
% appropriate values for the caplets.

MarketMat = [datetime(2008,3,21) ; datetime(2008,6,21) ; datetime(2008,9,21) ; datetime(2008,12,21) ; datetime(2009,3,21) ; ...
datetime(2009,6,21) ; datetime(2009,9,21) ; datetime(2009,12,21); datetime(2010,3,21) ; ...
datetime(2010,6,21); datetime(2010,9,21) ; datetime(2010,12,21) ; datetime(2011,3,21)]; 

MarketVol = [0.1533 0.1731 0.1727 0.1752 0.1809 0.1800 0.1805 0.1802 0.1735 0.1757 ... 
             0.1755 0.1755 0.1726; % First row in table corresponding to Strike1 
             0.1526 0.1730 0.1726 0.1747 0.1808 0.1792 0.1797 0.1794 0.1733 0.1751 ... 
             0.1750 0.1745 0.1719]; % Second row in table corresponding to Strike2