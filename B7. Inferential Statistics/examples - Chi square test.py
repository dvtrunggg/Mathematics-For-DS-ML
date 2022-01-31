"""============================================================================
   Slides #99
      - H0: X và Y độc lập
============================================================================"""
from scipy.stats import chi2_contingency

table1 = [[10, 20, 30], [6, 9, 17]]
table2 = [[91, 104, 235], 
          [39,  73,  48],
          [18,  31, 161]]

stat, p, df, expected = chi2_contingency(table1)
print('statistic = %.3f' %stat)
print('p-value   = %.3f' %p)
print('df        = %d' %df)

