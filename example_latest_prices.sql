-- This returns each ticker's latest stored adjusted close.
SELECT ticker, date, close AS adjusted_close
FROM `stockpulse-ai-511816.stockpulse.prices`
QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
ORDER BY ticker;
