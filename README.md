# Same Company, Same Signal:\\ The Role of Identity in Earnings Call Transcripts

This repository is dedicated to the research paper: "Same Company, Same Signal: The Role of Identity in Earnings Call Transcripts." 

## About the Paper
Post-earnings volatility prediction is critical for investors, with previous works often leveraging earnings call transcripts under the assumption that their rich semantics contribute significantly. To further investigate how transcripts impact volatility, we introduce DEC, a dataset featuring accurate volatility calculations enabled by the previously overlooked \texttt{beforeAfterMarket} attribute and dense ticker coverage. Unlike established benchmarks, where each ticker has only around two earnings, DEC provides 20 earnings records per ticker. Using DEC, we reveal that post-earnings volatility undergoes significant shifts, with each ticker displaying a distinct volatility distribution. To leverage historical post-earnings volatility and capture ticker-specific patterns, we propose two training-free baselines: \textit{Post-earnings Volatility} (PEV) and \textit{Same-ticker Post-earnings Volatility} (STPEV). These baselines surpass all transcripts-based models on DEC as well as on established benchmarks. Additionally, we demonstrate that current transcript representations predominantly capture ticker identity rather than offering financially meaningful insights specific to each earnings. This is evidenced by two key observations: earnings representations from the same ticker exhibit significantly higher similarity compared to those from different tickers, and predictions from transcript-based models show strong correlations with prior post-earnings volatility.

## Code and Dataset

The code and dataset used in this study are integral to reproducing the results and furthering the research. We are currently preparing both for release and anticipate making them available soon.

### What to Expect

- **Code**: The complete source code used for analysis and experiments in the paper.
- **Dataset**: Access to the datasets utilized or created during our research.
- **Documentation**: Guidelines on how to set up and use the code and data.

## Usage

Instructions on how to use the code and dataset will be provided here once they are made available.

## Citation

If you use our research, code, or dataset in your work, please cite our paper. The citation will be provided here once the paper is published.

## Contact

For any questions or further information, please reach out to dyu18@ur.rochester.edu.

---
Repository maintained by Ding Yu
