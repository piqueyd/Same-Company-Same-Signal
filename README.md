# Same Company, Same Signal

This repository is dedicated to the paper: ["Same Company, Same Signal: The Role of Identity in Earnings Call Transcripts"](https://aclanthology.org/2025.findings-acl.946/) (Findings of ACL 2025). A preprint is also available on [arXiv](https://arxiv.org/abs/2412.18029).

## Abstract
Post-earnings volatility prediction is critical for investors, with previous works often leveraging earnings call transcripts under the assumption that their rich semantics contribute significantly. To further investigate how transcripts impact volatility, we introduce DEC, a dataset featuring accurate volatility calculations enabled by the previously overlooked beforeAfterMarket attribute and dense ticker coverage. Unlike established benchmarks, where each ticker has only around two earnings, DEC provides 20 earnings records per ticker. 

Using DEC, we reveal that post-earnings volatility undergoes significant shifts, with each ticker displaying a distinct volatility distribution. To leverage historical post-earnings volatility and capture ticker-specific patterns, we propose two training-free baselines: Post-earnings Volatility (PEV) and Same-ticker Post-earnings Volatility (STPEV). These baselines surpass all transcripts-based models on DEC as well as on established benchmarks. 

Additionally, we demonstrate that current transcript representations predominantly capture ticker identity rather than offering financially meaningful insights specific to each earnings. This is evidenced by two key observations: earnings representations from the same ticker exhibit significantly higher similarity compared to those from different tickers, and predictions from transcript-based models show strong correlations with prior post-earnings volatility.


## Dense Earnings Call Dataset
You can access the DEC (Dense Earnings Call) dataset via [Google Drive](https://drive.google.com/drive/folders/1BZp5rzwdVtLSwUFwAUNcU6EHPNcZldJI?usp=drive_link). The dataset is provided in two formats:

1. **Individual earnings call transcripts** – each in a `.txt` file, located in the `DEC/` subdirectory.
2. **Aggregated JSON format** – all earnings calls combined into a single file: `DEC.json`.


We also provide three types of embeddings for reproducing experiments:
1. **OpenAI Vanilla Transcripts**  
2. **Random Ticker**  
3. **Random All**
   
Embedding-level analysis can be found in `SS.ipynb`.

DEC is released for academic research use. Please see [License](#license) below before redistributing or building on the dataset.

## Code Organization
```text
Same Company, Same Signal/
│
├── README.md       # Project overview and documentation
├── SameCompanySameSignal/           # Code for running experiments with both transcript-based models and time-series-based models
└── SS.ipynb        # Introduction to the DEC dataset, access instructions, and analysis
```

## Code Usage
```text
# Run transcript-based model (MLP)
# Please download the embeddings and place them under SameCompanySameSignal/dataset/
bash ./scripts/TMLP_DEC.sh

# Run time-series-based model (TSMixer)
bash ./scripts/TSMixer_DEC.sh
```

## License

This project is released under two separate licenses.

**Code** — everything in this repository — is licensed under the MIT License. See [`LICENSE`](LICENSE).

**The DEC dataset** is split by origin, because it combines our own work with transcript text authored by third parties. Full terms are in [`LICENSE-DATA`](LICENSE-DATA); in summary:

| Component | Terms |
|---|---|
| Volatility calculations, `beforeAfterMarket` annotations, ticker/earnings selection, splits, metadata, embeddings, and `DEC.json` structure | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to share and adapt with attribution |
| Earnings call transcript text (`DEC/*.txt`, transcript fields of `DEC.json`) | Third-party material collected from publicly accessible sources. Copyright remains with the respective owners; we make no copyright grant over it and provide it to support academic research and reproduction of our results. |

**Using DEC in academic work — including unpublished theses, dissertations, and postgraduate research reports — is permitted.** Please cite the paper below. If you intend to redistribute the transcript text or use it commercially, you are responsible for obtaining any permissions required.

Rights holders with concerns about specific material may contact dyu18@ur.rochester.edu and it will be removed.

## Acknowledgements

We would like to express our gratitude to the following projects and contributors:

1. **[Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library)** – Parts of the code in this project are adapted from their implementation.
2. **[What You Say and How You Say It Matters: Predicting Stock Volatility Using Verbal and Vocal Cues](https://github.com/GeminiLn/EarningsCall_Dataset)** – For providing the EC dataset.
3. **[MAEC: A Multimodal Aligned Earnings Conference Call Dataset for Financial Risk Prediction](https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction)** – For providing the MAEC dataset.
4. **[KeFVP](https://github.com/hankniu01/KeFVP/tree/main)** – For supplying the price data associated with the EC and MAEC datasets.

## Citation
If you use our research, code, or dataset in your work, please cite our paper. 
```bibtex
@inproceedings{yu-etal-2025-company,
    title = "Same Company, Same Signal: The Role of Identity in Earnings Call Transcripts",
    author = "Yu, Ding  and
      Liu, Zhuo  and
      He, Hangfeng",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.946/",
    pages = "18403--18422",
}
```

## Contact

For any questions or further information, please reach out to dyu18@ur.rochester.edu.

---
Repository maintained by Ding Yu
