# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Data Science: Final Project Ideas

> Sample Datasets for Final Projects

Below, we have sourced some suggested datasets suitable for final projects, including a range of selections - some are more complicated and may require a substantial amount of effort. 

## Dataset Suggestions
| | 
| --- | 
|*Intermediate Datasets* |
| [Option 1: Amazon Pricing Data](#option1) |
| [Option 2: Insurance Marketplace Data](#option2) | 
| [Option 3: Loan Data](#option3) | 
| *Advanced Datasets* |
| [Option 4: Fuel Economy Data](#option4) | 
| [Option 5: Craft Beer Data](#option5) | 
| [Option 6: Policing Data](#option6) | 
| [Option 8: Choose Your Own](#option7) |

> Note: Alternatively, instructors may allow students to choose their own final project topic and dataset. Students should clearly indicate this preference and discuss their choice(s) with instructors ahead of time.

---

<a name="option1"></a>
## Option 1: Amazon Pricing Data

From ProPublica:

> ProPublica reporters examined Amazon’s shopping algorithm. We scraped data from the company's website to examine listings for 250 best-selling products across a wide range of categories, from electronics to household supplies, over a period of several weeks during summer 2016. We compared pricing and shipping costs for products offered by multiple vendors, including those sold by Amazon and sellers in the "Fulfilled by Amazon" program. In total, we examined 6,973 vendor listings.

**Possible Areas to Examine**

- How does vendor affect price?
- How does rank affect an item's price? Or vice versa?

[Download Here](https://www.propublica.org/datastore/dataset/amazon-pricing-algorithm-data-set)

---

<a name="option2"></a>
### Option 2: Health Insurance Marketplace

From Kaggle:

> The Health Insurance Marketplace Public Use Files contain data on health and dental plans offered to individuals and small businesses through the U.S. Health Insurance Marketplace.

**Possible Areas to Examine**

- How do plan rates and benefits vary across states?
- How do plan benefits relate to plan rates?
- How do plan rates vary by age?
- How do plans vary across insurance network providers?

> Note: This data set is quite large, and taking a random 10-percent sample by state (with justification) is acceptable.

[Download Here](https://www.kaggle.com/hhs/health-insurance-marketplace)

The Public Use Files receive regular updates from the Centers for Medicare & Medicaid Services and can be used in place of or as a supplemental addition to the kaggle data. Please note that these files are very large, so consider additional sampling or a targeted question. Datasets may be downloaded [here](https://www.cms.gov/cciio/resources/data-resources/marketplace-puf.html).

---

<a name="option3"></a>
### Option 3: Loan Data

From Kaggle:

> These files contain complete loan data for all loans issued through the 2007–2015, including the current loan status (current, late, fully paid, etc.) and latest payment information. The file containing loan data through the "present" contains complete loan data for all loans issued through the previous completed calendar quarter. Additional features include credit scores, number of finance inquiries, address including zip codes and state, and collections among others. The file is a matrix of about 890,000 observations and 75 variables. A data dictionary is provided in a separate file.

**Possible Areas to Examine:**

- What factors most explain on-time loan repayments?
- As a bank, what evaluation metric would you opt for to determine if a given load should be given? As a consumer, what evaluation metric would you opt for to determine if a given loan should be given? How does this affect your model's performance?

[Download Here](https://www.kaggle.com/wordsforthewise/lending-club)

---

## Advanced Data Sets (Optional)

These "stretch" data sets are ordered by an estimated feasibility of analysis for your final project. The reason they are considered optional is denoted for each item. Even if you are unable to complete work with one these data sets, know that additional practice will unlock the skills necessary to complete a final project on any of the following suggestions.

<a name="option4"></a>
### Option 4: Fuel Economy Data

The U.S. Environmental Protection Agency publishes miles-per-gallon data following tests of vehicles on the road in the United States. Data sets from the years 2000–2018 are readily accessible for all make and models.

**Possible Areas to Examine**

- Predict the miles per gallon of a given brand and car type.
- Create a cross-sectional look at how car efficiencies have improved over the years. Predict future miles per gallon for models, brands, or car types.
- Predict the carbon impact for a given brand or car type.

**Difficulty Assessment**

The multi-year nature of the data set may prove to be a data-wrangling challenge. Nonetheless, the overall size of the set across years makes it approachable — perhaps even more so than prior options.

> Moreover, forecasting is not explicitly covered in this portion of the course, so model selection may be a panel OLS by year or a decision tree regressor.

[Download Here](https://www.kaggle.com/epa/fuel-economy)

---

<a name="option5"></a>
### Option 5: Craft Beers

Craftcans.com maintains a continuously updated data set of (as of July 2017) 2,962 craft beers. The data contains the beer name, brewery, location, style, size, ABV, and IBUs.

**Possible Areas to Examine.**

- Predict the ABV based on other factors.
- Cross-reference this data with the [ratebeers.com API](https://www.ratebeer.com/api.asp) to predict quality of beer using the other factors provided.

**Difficulty Assessment**

The multi-year nature of the data set may prove to be a data-wrangling challenge. Nonetheless, the overall size of the set across years makes it approachable — perhaps even more so than prior options.

> Moreover, forecasting is not explicitly covered in this portion of the course, so model selection may be a panel OLS by year or a decision tree regressor.

[Download Here](https://www.kaggle.com/nickhould/craft-cans)

---

<a name="option6"></a>
### Option 6:  Police Stops

The Open Policing Project is a useful store of many interesting datasets collected from varios police departments across the United States and details information about the person pulled over, why they were stopped, and different outcomes from event.

**Possible Areas to Examine.**

 - Impact of different personal characteristics on policing outcomes
 - Differences in importance of different features for different outcomes (frisked, searched, arrested)

There is a cleaned version of this data which combines police stops across all of North Carolina that is easy to get started with.

**Difficulty Assessment**

This dataset is rather large (4.5 million records), and it is difficult to tell what events happened when, so determining what columns to include to properly study an outcome can be important for studying results correctly.

[Download Here](https://www.kaggle.com/jonathanbechtel/north-carolina-master-dataset-of-arrests/version/1)

<a name="option7"></a>
### Option 7: Choose Your Own Dataset

If none of the datasets above look interesting to you, feel free to find a different one. Just be sure to clearly indicate this preference and discuss your choice with your instructor.
