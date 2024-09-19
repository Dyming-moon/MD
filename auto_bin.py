# utf8

import numpy as np
import pandas as pd
import scipy
from scipy import stats


class AutoBins:

    def __init__(self, frame, y):
        self._frame = frame.copy()
        self._y = y

    def _column_qcut(self, column):

        _, bins = pd.qcut(self._frame[column], q=20, retbins=True, duplicates="drop")

        bins = list(bins)
        bins.insert(0, -float("inf"))
        bins[-1]= float("inf")

        self._frame[column+"_qcut"] = pd.cut(self._frame[column], bins=bins)

        init_counts = list(self._frame[column+"_qcut"].value_counts(sort=False))
    
        if init_counts[0] < (len(self._frame)/50):
            bins.pop(1)
        if init_counts[-1] < (len(self._frame)/50):
            bins.pop(-2)


        self._frame[column+"_qcut"] = pd.cut(self._frame[column], bins=bins)
     
        inf_init_bins = self._frame.groupby([column+"_qcut", self._y])[self._y].count().unstack(fill_value=0)
       
        num_bins = [*zip(bins, bins[1:], inf_init_bins[0], inf_init_bins[1])]
        return num_bins

    def _merge_zero_bins(self, num_bins):
       
        idx = 0
        while idx < len(num_bins):
           
            if 0 in num_bins[0][2:]:
                num_bins = self._merger_bins(num_bins, idx)
                continue
            else:
              
                if 0 in num_bins[idx][2:]:
                    num_bins = self._merger_bins(num_bins, idx-1)
                    continue
                else:
                  
                    idx += 1
        return num_bins

    def _merger_bins(self, num_bins, x):
      
        num_bins[x: x+2] = [(
            num_bins[x][0],
            num_bins[x+1][1],
            num_bins[x][2]+num_bins[x+1][2],
            num_bins[x][3]+num_bins[x+1][3]
        )]
        return num_bins

   
    def _get_iv(self, woe_df):
        rate = ((woe_df.count_0/woe_df.count_0.sum()) -
                (woe_df.count_1/woe_df.count_1.sum()))
        iv = np.sum(rate * woe_df.woe)
        return iv


   
    def _get_woe(self, num_bins):
      
        columns = ["min", "max", "count_0", "count_1"]
        df = pd.DataFrame(num_bins, columns=columns)

        df["total"] = df.count_0 + df.count_1
        df["percentage"] = df.total / df.total.sum()
        df["bad_rate"] = df.count_1 / df.total
        df["woe"] = np.log(
            (df.count_0 / df.count_0.sum()) /
            (df.count_1 / df.count_1.sum())
            )
        return df

    def _chi2_merge(self, num_bins):
        p_values = []
     
        for i in range(len(num_bins)-1):
            x1 = num_bins[i][2:]
            x2 = num_bins[i+1][2:]
          
            pv = stats.chi2_contingency([x1, x2])[1]
          
            p_values.append(pv)

       
        idx = p_values.index(max(p_values))
        num_bins = self._merger_bins(num_bins, idx)
        return num_bins

    def auto_bins(self, column, n=2, show_iv=True):
        if show_iv:
            print("Binning {} columns: ".format(column))
       
        num_bins = self._column_qcut(column)
      
        num_bins = self._merge_zero_bins(num_bins)
       
        while len(num_bins) > n:
            num_bins = self._chi2_merge(num_bins)
            woe_df = self._get_woe(num_bins)
            iv = self._get_iv(woe_df)
            if show_iv:
                print("Number of group: {:02d} \tiv: {}".format(len(num_bins),iv))

        woe_df = self._get_woe(num_bins)
        iv = self._get_iv(woe_df)
        if show_iv:
            print("\nBinning outcome: ")
            print("Number of group: {:02d} \tiv: {}".format(len(num_bins),iv))
            print("\nwoe ：")
            print(woe_df)
        return num_bins, woe_df, iv

