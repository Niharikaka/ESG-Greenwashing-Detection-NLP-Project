# ESG Greenwashing Detection: NLP Patent-Report Alignment Analysis

**Course:** AIB551 Natural Language Processing | **Student:** Niharika Karanam | **SUSS ID:** B2511473 | **Submission Date:** 03/11/2025

---

## A. Introduction and Problem Statement

### Project Overview

Companies are increasingly publishing detailed and ambitious Environmental, Social and Governance Disclosures (ESG) Reports that claim ambitious environmental commitments like "carbon-neutral products," "closed-loop manufacturing," or "100 % recycled materials" to attract investors, consumers and satisfy the regulators. However, many of these claims are vague or unverified, leading to accusations of greenwashing.

This project uses a Natural Language processing framework to assess the authenticity of corporate ESG Reports with their actual Research and development in the sustainability space by checking their patent data. The study uses Apple's 2024 ESG Report and 2023-2024 Sustainability patents as a case study. Through the use of embedding and comparing both texts using domain-specific transformer models (e.g., ClimateBERT, ESG-BERT), Apple's ESG Statements and patent data can be quantified to see if there is semantic alignment. A high semantic overlap would indicate that Apple's ESG report is backed by significant proof of research and innovation while persistent gaps could flag possible greenwashing.

### Background and Existing Research

Recent research in 2020-2025 in ESG related NLP have leveraged transformer models to analyze sustainability disclosures, detect sentiment, classify ESG topics and classify action/no action claims. Researchers like Calamai et al. (2025) and Schimanski et al. (2023) have shown that domain-specific BERT variants can capture nuanced ESG language, while other researchers like Zhao et al. (2023) and Chang et al. (2024) demonstrated how discrepancies between ESG claims and public sentiment may signal greenwashing. Santarlasci et al. (2025) analysed 12 million "green" patents and found only ≈20 % contained genuine eco-innovation.

As shared by Lublóy, Á. (2025) these are the notable limitations of the current approach:

- They are heavily dependent on labelled training data, which remains scarce in greenwashing contexts.
- Many rely on supervised classification using small, domain-specific datasets, limiting generalizability.
- Few studies are able to incorporate external validation sources such as emissions, patents, or product life-cycle data.
- No unified definition of greenwashing, and as a result current systems largely detect linguistic patterns rather than being able to verify factual consistency.

In summary, as noted by Lublóy, Á. (2025), most of the past research has examined ESG disclosures or patent data separately, but not together.

### Problem Statement

This project aims to bridge this gap between talk vs walk using Apple Inc. as a case study and NLP techniques to measure the semantic consistency to explore how well a company's sustainability claims in ESG Reports align with its real technological developments, particularly patent data.

### Case Study: Apple Inc.

Apple being a significant global tech leader, also markets itself as a climate leader, pledging carbon neutrality across its supply chain and products by 2030, and labeling the Apple Watch Series 9 as its "first carbon-neutral product" (Apple, 2023). The company emphasizes recycled materials, clean energy, and eco-design changes such as removing wall chargers from iPhones to reduce e-waste (Wired, 2020). Many other companies followed Apple's footsteps and stopped providing wall chargers, earbuds signalling their position as a change maker in sustainability practices and still being publicly favoured.

Yet these claims have attracted skepticism, suggesting possible greenwashing. In 2025, a German court ruled that Apple's "carbon-neutral" product label was misleading, citing insufficient verification of the carbon offsets used (ESG Dive, 2025).

At the same time, Apple is one of the world's top patent filers. In 2023 alone, it had 2,536 U.S patents including many related to recycling systems, renewable materials, and energy-efficient designs (IFI Claims, 2024).

### Significance of the Problem

Greenwashing has become a widespread issue across industries:
- **40–57%** of corporate environmental claims are misleading (ACCC, 2022)
- **94%** of investors believe many companies exaggerate their ESG performance (PwC, 2023)

---

## B. Methodology

### Data Collection and Sources

#### (a) ESG Datasets

Apple's Environmental Progress Report 2024 and GRI Content Index 2024 were downloaded from Apple's environment portal:
- 📄 https://www.apple.com/environment/pdf/Apple_Environmental_Progress_Report_2024.pdf
- 📊 https://www.apple.com/ca/environment/reports/gri/

The main report supplied the text for sustainability claims by Apple, while the GRI Index relevant to the Environmental Progress Report was used as a thematic dictionary defining ten GRI-aligned categories (Energy and Emissions, Materials and Circular Economy, Waste and Recycling, Product Responsibility, etc.).

Both a focused subset (sections Smarter Chemistry and Product Longevity) and the full report were analysed to test whether smaller, product-specific text improved ESG patent alignment through cosine similarity.

#### (b) Patent Data

179 active U.S. patents assigned to Apple Inc. and published between 1 January 2023 and 31 December 2024 were retrieved from Lens.org using green-technology CPC codes (C02F, F24F, A61K, C08L, B65D, Y02A, Y02E, Y02W, Y02P, etc.):
- 🔬 https://www.lens.org/

Each patent abstract was rewritten in DWPI style using GPT-4 Mini to simplify technical wording and align words with ESG report language, improving the lexical and semantic comparability between patents and sustainability claims in ESG reports.

### Data Cleaning and Pre-processing

Text extraction used PyPDF2 for PDFs and pandas for CSV patent data. Pre-processing pipeline:
- Text normalisation: lower-casing, removal of punctuation, URLs, and stop words (nltk)
- Tokenisation and lemmatisation: spaCy pipeline to unify word forms
- Duplicate and null removal
- Sentence segmentation from report paragraphs (≥ 25 characters per sentence)
- Exploratory EDA: token-length distribution, word-frequency plots, and TF-IDF word clouds

After cleaning, **1,661 ESG sentences** and **177 patent abstracts** remained.

**Figure 1: Distribution of Token Counts (ESG Sentences)**

![Distribution of Token Counts](figures/figure_01_cell13.png)

**Figure 2: ESG Themes Word Cloud**

![ESG Themes Word Cloud](figures/figure_02_cell13.png)

**Figure 3: Patent Abstracts Word Cloud**

![Patent Abstracts Word Cloud](figures/figure_03_cell13.png)

**Figure 4: Top 20 Words in ESG Reports**

![Top 20 Words in ESG Reports](figures/figure_04_cell13.png)

**Figure 5: Explained Variance by TruncatedSVD Components**

![Explained Variance](figures/figure_05_cell13.png)

**Figure 6: Top 20 Words in Patents**

![Top 20 Words in Patents](figures/figure_06_cell14.png)

**Figure 7: TF-IDF Top Keywords Comparison**

![TF-IDF Keywords](figures/figure_07_cell14.png)

**Figure 8: ESG vs Patent Token Length Distribution**

![Token Length Distribution](figures/figure_08_cell14.png)

**Figure 9: ESG vs Patent Token Count Distribution**

![Token Count Distribution](figures/figure_09_cell15.png)

**Figure 10: ESG Theme Word Overlap**

![Theme Word Overlap](figures/figure_10_cell16.png)

**Figure 11: TF-IDF Cosine Similarity Heatmap (ESG Themes vs Patents)**

![TF-IDF Heatmap](figures/figure_11_cell17.png)

### Machine Learning and NLP Techniques

#### (a) Entity Recognition

A fine-tuned DistilBERT NER model (ExponentialScience/ESG-DLT-NER) identified ESG entities such as carbon neutral, recycled material, and renewable energy. Evaluation metrics (precision = 0.86, recall = 0.81, **F1 = 0.83**) were computed with seqeval.

**Figure 12: Most Frequent Sustainability Entities — ESG vs Patents**

![ESG vs Patent Entities](figures/figure_12_cell20.png)

**Figure 13: Distribution of ESG–Patent Semantic Similarity (MiniLM)**

![ESG Patent Semantic Similarity](figures/figure_13_cell28.png)

**Figure 14: Top 15 ESG Entities Extracted**

![Top 15 ESG Entities](figures/figure_14_cell29.png)

**Figure 15: NER Training Performance — F1 Score per Epoch**

![NER Training](figures/figure_15_cell29.png)

**Figure 16: Entity Label Distribution**

![Entity Label Distribution](figures/figure_16_cell29.png)

**Appendix A1 — Carbon-Neutral Mentions: ESG vs Patent Texts**

Following the German court ruling that deemed "carbon neutral" a misleading claim, this figure highlights its prevalence in Apple's 2024 ESG Report, with over 120 mentions.

![Carbon Neutral Mentions](figures/figure_17_cell33.png)

#### (b) ESG–Patent Alignment: Focused Subset vs Full Report

**Figure 3 (in report): ESG–Patent Alignment Comparison**

Using the SentenceTransformer MiniLM-L6-v2, the full ESG report showed higher lexical overlap (0.125 vs 0.083) and semantic similarity (0.062 vs 0.047) with patents than the 2-PDF subset, so it was used for further analysis.

![ESG Patent Alignment Focused vs Full](figures/figure_18_cell42.png)

#### (c) Text Embedding and Semantic Similarity

Multiple transformers (MiniLM, MPNet, ESG-BERT, ClimateBERT, OpenAI Ada-002) generated sentence embeddings. Cosine similarity measured alignment between each ESG sentence and its closest patent.

**Figure 9: Distribution of ESG–Patent Similarities by Model**

![Model Similarity Distribution](figures/figure_19_cell46.png)

![Model Similarity Distribution 2](figures/figure_20_cell46.png)

**Average Similarity Scores by Model**

![Average Similarity by Model](figures/figure_21_cell47.png)

**Per-Theme Semantic Similarity (EnvironmentalBERT)**

![Per Theme EnvironmentalBERT](figures/figure_22_cell48.png)

**Per-Theme Semantic Similarity (Hybrid: EnvironmentalBERT + MiniLM)**

![Per Theme Hybrid](figures/figure_23_cell55.png)

**Figure 5: Distribution of ESG–Patent Similarities (MiniLM)**

![Similarity Distribution MiniLM](figures/figure_24_cell59.png)

**Cosine Similarity Distribution (All pairs)**

![Cosine Similarity All Pairs](figures/figure_25_cell62.png)

**Figure 9: Embedding Model Similarity Comparison**

MiniLM produced low, varied similarity scores, while EnvironmentalBERT gave unrealistically high values. Ada-002 achieved balanced, consistent results, making it the most reliable model for ESG–patent alignment analysis.

![Embedding Comparison](figures/figure_26_cell64.png)

**Figure 13: Distribution of ESG–Patent Semantic Similarity (Ada-002)**

![Ada-002 Similarity Distribution](figures/figure_27_cell69.png)

**Ada-002 Similarity by Action vs None Classification**

![Ada Action vs None](figures/figure_28_cell69.png)

**Ada-002 Action Classification Analysis**

In addition to embedding comparisons, the ESGBERT/EnvironmentalBERT-action classifier was applied to label ESG sentences as Action or None. Action labelled sentences had slightly higher similarity (~0.77) with patents.

![Ada Action Analysis](figures/figure_29_cell70.png)

#### (d) Keyword and Theme Extraction

**Figure 6: Top TF-IDF Keywords**

![TF-IDF Keywords](figures/figure_30_cell74.png)

**Figure 7: GRI Aligned ESG Theme Distribution (Zero-Shot)**

![GRI Theme Distribution](figures/figure_31_cell75.png)

**GRI Theme Distribution (Bar Chart)**

![GRI Theme Bar](figures/figure_32_cell75.png)

**Zero-Shot Theme Classification Results**

![Zero-Shot Themes](figures/figure_33_cell76.png)

**Zero-Shot Theme Distribution (Pie)**

![Zero-Shot Pie](figures/figure_34_cell76.png)

**Figure 8: ClimateBERT Environmental Claims — Patent Alignment by Theme**

ClimateBERT was used to classify ESG report sentences as environmental claims or not. A two-sample t-test comparing their similarity to patent texts showed that only the Waste and Recycling theme exhibited a statistically significant difference (p = 0.00025).

![ClimateBERT Claims](figures/figure_35_cell78.png)

**Figure 7 (Appendix A7): Patent Alignment by GRI Theme and Environmental Claim Status (Ada-002)**

![GRI Claim Alignment](figures/figure_36_cell79.png)

#### (e) Ada-002 vs SBERT Model Comparison

**Figure 10: Ada-002 vs BERT Normalized Similarities**

The scatter plot shows a moderate correlation (r = 0.48) between the two models, indicating that both capture similar alignment patterns but with different scaling.

![Ada vs SBERT Scatter](figures/figure_37_cell91.png)

**Ada vs SBERT — Normalized Comparison**

![Ada vs SBERT Comparison](figures/figure_38_cell91.png)

**Figure 11: Mean Normalized ESG–Patent Similarity by GRI Theme (Heatmap)**

The heatmap highlights similar results for Energy & Emissions and Materials & Circular Economy across both models.

![GRI Theme Heatmap](figures/figure_39_cell92.png)

**Figure 12: Mean Normalized ESG–Patent Similarity per GRI Theme (Bar Chart)**

Although SBERT produced slightly higher similarity scores in several themes, statistical testing (t = -0.877, p = 0.388) showed no significant difference between the two models after normalization.

![GRI Theme Bar Chart](figures/figure_40_cell92.png)

**Figure 11 (Appendix A6): Average ESG–Patent Alignment by Model and GRI Theme**

Compares embedding models (Ada-002, EnvBERT, MiniLM) — EnvBERT has highest alignment but Ada-002 provides more balanced, reliable results.

![Model Theme Heatmap](figures/figure_41_cell95.png)

**Average Alignment by GRI Theme and Model (Heatmap)**

![Alignment Heatmap 2](figures/figure_42_cell95.png)

**Figure 6 (Appendix A6): Average ESG-Patent Alignment by Model and GRI Theme (Bar)**

![Alignment Bar](figures/figure_43_cell95.png)

---

## C. Results and Discussion

### 5. Alignment and Greenwashing Risk Scoring

**Alignment Score Formula:**

```
Alignment Score = 0.60 × Semantic Similarity
                + 0.25 × Claim–Action Consistency
                + 0.15 × Theme Coverage

Greenwashing Risk Score = 1 − Alignment Score
```

**Figure 16: Average ESG–Patent Alignment by Theme (Ada-002)**

Average ESG-Patent Alignment by Theme shows that Circularity (0.776) and Climate (0.769) achieved the highest mean similarity, while Resources (0.753) scored lowest.

![Alignment by Theme](figures/figure_44_cell100.png)

**Figure 17: Average ESG–Patent Alignment by Zero-Shot Theme**

Zero-Shot Themes confirms this trend, with Materials and Circular Economy (0.781) showing the strongest patent backing, whereas Water and Resource Use (0.755) remains least supported.

![Zero Shot Alignment](figures/figure_45_cell101.png)

**Appendix A9 — Top ESG–Patent Pairs (Cosine Similarity Heatmap)**

The top 20 ESG-patent pairings show close thematic and technical correspondence, confirming that the model's highest-similarity matches are logically valid.

![Top Pairs Heatmap](figures/figure_46_cell104.png)

**Appendix A2 — Word Cloud of ESG Entities**

![ESG Entity Wordcloud](figures/figure_47_cell105.png)

**Appendix A4 — Word Cloud of Patent Abstracts**

![Patent Wordcloud](figures/figure_48_cell105.png)

**Figure 16 (Ada-002): Average ESG-Patent Alignment by GRI Theme**

![Ada Theme Alignment](figures/figure_49_cell106.png)

**Figure 17 (Zero-Shot): Average ESG-Patent Alignment by Theme**

![Zero-Shot Theme Alignment](figures/figure_50_cell107.png)

**Appendix A11 — Correlation: Semantic Similarity vs ESG-BERT & ClimateBERT Indicators**

Shows weak correlations between similarity and classifier outputs, meaning patent alignment and classifiers capture different aspects of ESG reporting.

![Correlation Heatmap](figures/figure_51_cell108.png)

**Figure 19: Distribution of Predicted Greenwashing Risk**

Most ESG statements fall in a moderate-risk range (0.3–0.7), with relatively few either fully aligned (≈ 0) or highly risky (≈ 1).

![Risk Distribution](figures/figure_44_cell100.png)

**Appendix A6 — Average ESG–Patent Alignment by Model and GRI Theme**

![Model Alignment Comparison](figures/figure_53_cell108.png)

**Appendix A7 — Patent Alignment by GRI Theme and Environmental Claim Status**

![Claim Status Alignment](figures/figure_54_cell108.png)

**Appendix A5 — Distribution of ESG–Patent Similarities (Ada-002)**

Shows a normal distribution of cosine similarity (mean ≈ 0.73), indicating consistent moderate alignment.

![Final Similarity Distribution](figures/figure_55_cell108.png)

### Interpretations and Implications

The results indicate that Apple's ESG data are partially supported by patent data, strongly in materials, circularity and energy themes and weak links in water-use and supply chain topics.

Sentences flagged as high risk mainly involve future goals, certifications, or partner initiatives (e.g., AWS water stewardship or "100 % recycled materials by 2025"), which depend on policy and supplier compliance rather than Apple's own patents.

These findings suggest that the model is capable of detecting communication and innovation gaps rather than purposeful greenwashing. The alignment score and classifier together can provide a quantitative method to evaluate ESG greenwashing.

Overall, Apple's sustainability reporting appears largely credible, with clear patent evidence for energy and material efficiency innovations but less substantiation for resource and governance claims which is typical for a tech company.

---

## D. Conclusion and Recommendation

### Summary of Findings

The project applied NLP and machine learning to explore how well a company's sustainability claims in ESG Reports align with its real technological developments, particularly patent data. By using Apple's ESG report along with its 2023-2024 patent data, Ada-002 embeddings, ESG-BERT and ClimateBERT classifiers, BERTopic Model, and a Random Forest model, the framework quantified how closely reported commitments align with research and innovation.

Findings show that Apple's ESG communication is partially but meaningfully supported by patent data, with strong alignment in materials, circularity, and energy efficiency themes, and weaker links in water-use and supply-chain topics. Sentences identified as high risk largely represent future commitments, certifications, or partner initiatives, suggesting ambition rather than deceptive language.

### Impact and Practical Application

The machine learning solution contributes to the growing need for automated, scalable ESG verification systems. By linking ESG Report to patent data, the framework offers better transparency:
- Integrating patent-based verification in ESG data platforms
- Supporting investor confidence and simpler sustainability auditing
- Assisting regulators in identifying unsubstantiated or misleading claims before they spread in the market

### Limitations

- Context mismatch between ESG narrative language and patent technical terms
- Lack of labelled greenwashing data — heuristic labelling was used
- ESG commitments often describe future targets not yet reflected in patent filings
- The framework's accuracy relies on embedding quality (Ada-002)

### Future Work

- Human ESG experts for re-labelling patent data and expand datasets to include emissions or supplier data
- Track how ESG-patent alignment evolves over time as companies move from commitments to action
- Combine patent verification with emissions, financial, and media data for holistic ESG authenticity scoring
- Apply the framework to various sectors to identify industry-wide greenwashing patterns

In conclusion, this framework offers a practical way to verify sustainability statements with real patent innovations and spot possible greenwashing behaviour by companies. While Apple's ESG claims are mostly credible, some gaps remain between what is written in ESG Reports and what is achieved, highlighting the need for clearer, evidence-based ESG reporting.

---

## APA References

Calamai, A., Hübner, J., & Webersinke, N. (2025). Corporate greenwashing detection in text: A survey. *arXiv preprint arXiv:2501.12345*.

Chang, T., Jin, G. Z., & Luca, M. (2024). Greenwashing in earnings calls: Evidence from FinBERT. *Sustainable Finance Alliance*.

Lublóy, Á., de Koning, D., & Rousseau, R. (2025). Quantifying firm-level greenwashing: A systematic review. *Journal of Environmental Management*, 330, 117122.

Santarlasci, L., Webersinke, N., & Hübner, J. (2025). Seeing through green: NLP for identifying true green patents. *arXiv preprint arXiv:2502.07455*.

Schimanski, S., Lehmann, J., & Weber, M. (2023). Bridging the gap in ESG measurement: Using NLP to quantify E, S, G communication. *SSRN Electronic Journal*.

Zhao, L., Liu, W., & Qian, Z. (2023). Detecting greenwashing in ESG domains via NLP: A case study in pharma. *KDIR Conference Proceedings*.

Apple. (2023). Apple unveils its first carbon neutral products. https://www.apple.com/newsroom

Apple Inc. (2024a). *Apple environmental progress report 2024*. https://www.apple.com/environment/pdf/Apple_Environmental_Progress_Report_2024.pdf

Apple Inc. (2024b). *GRI content index 2024*. https://www.apple.com/ca/environment/reports/gri/

Australian Competition and Consumer Commission. (2022, March). *Greenwashing internet sweep: Findings report*. Government of Australia.

BloombergNEF. (2024, May). ESG assets exceed $30 trillion as investors push for sustainability transparency. https://about.bnef.com/

IFI Claims. (2024). Top 50 U.S. Patent Assignees 2023. https://www.ificlaims.com

Lens.org. (2025). Patent search: Apple Inc. (active 2023–2024, sustainable CPC classes). https://www.lens.org/

PricewaterhouseCoopers. (2023, November). *2023 Global investor survey*. https://www.pwc.com/

The Verge. (2023). Apple's first 'carbon neutral' products are a red herring. https://www.theverge.com

Wired. (2020). The iPhone 12 ships without a charger. Will it curb e-waste? https://www.wired.com

---

**Declaration:** I acknowledge that AI tools like ChatGPT were used as part of the preparation and development of this project.

