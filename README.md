# Exploring Species Population Change with Machine Learning

This project explores historical animal population data to understand how population trends can be grouped into different risk categories using machine learning.

I used Python to clean and analyse the data, create population-change features, compare classification models, and explore how uneven data collection can affect the interpretation of biodiversity patterns.

The project is an educational analysis, not a validated system for predicting real-world extinction risk.


-------------------------------------------------------------------------------------------------------------------------------------

# Link
https://predicting-global-species-extinction-risk-using-machine-learni.streamlit.app/


### Demo 1
![Demo 1](demos/demo1.gif)

---

### Demo 2
![Demo 2](demos/demo2.gif)

---

### Demo 3
![Demo 3](demos/demo3.gif)

---

### Demo 4
![Demo 4](demos/demo4.gif)

------------------------------------------------------------------------------------------------------------------------------------------------------


## Project Links

- **GitHub repository:** [View the code](https://github.com/amnafatimaa6-ops/Predicting-Global-Species-Extinction-Risk-Using-Machine-Learning)
- **Jupyter notebook:** [Endangered_species.ipynb](https://github.com/amnafatimaa6-ops/Predicting-Global-Species-Extinction-Risk-Using-Machine-Learning/blob/main/Endangered_species.ipynb)
- **Interactive dashboard:** [Open the Streamlit app](https://predicting-global-species-extinction-risk-using-machine-learni.streamlit.app/)
- **Demo screenshots:** [View the demos folder](https://github.com/amnafatimaa6-ops/Predicting-Global-Species-Extinction-Risk-Using-Machine-Learning/tree/main/demos)

**Important:** The notebook and the live dashboard are separate parts of this repository. The notebook contains the machine learning experiments. The dashboard uses live biodiversity occurrence records from GBIF and does not currently use the notebook's trained model.

## 1. The Question Behind the Project

Species populations can change over time, but studying these changes is not always straightforward. Historical datasets may contain missing values, uneven geographical coverage, and different amounts of monitoring across countries.

I wanted to explore this question:

**Can historical population changes be used to classify species into different population-trend risk categories, and what limitations appear when interpreting the results?**

## 2. Dataset and Preparation

The notebook uses a file named `LPD_2024_public.csv`.

After removing an irrelevant column, standardising column names, and checking missing values, I reshaped the yearly population data from wide format into long format.

The processed dataset contained:

- **420,356 records**
- **5,176 unique species**
- **202 countries**

Missing population observations were removed. I then averaged population values across locations for each species and year, allowing me to compare population values for 1970 and 2020.

These figures describe the processed dataset before the final filtering used for model training.

The CSV file is not included in the repository. To reproduce the notebook, you need to obtain the dataset separately and update the file path to match your environment.

## 3. Creating Population-Change Features

I calculated the relative population change between 1970 and 2020:

\[
\text{Change Rate}=\frac{P_{2020}-P_{1970}}{P_{1970}}
\]

I used this value to create three project-defined categories:

- **Stable:** change rate greater than -0.2
- **Vulnerable:** change rate greater than -0.5 and up to -0.2
- **Endangered:** change rate of -0.5 or lower

These are simplified categories created for this project. They are **not official IUCN Red List classifications**.

I also created features representing absolute population change, growth ratio, and log-transformed population change.

A minimum baseline population of 10 was applied, and invalid or infinite values were removed before model training.

## 4. Machine Learning Approach

I compared three classification models using a stratified 80/20 train-test split with `random_state=42`:

- Logistic Regression
- Random Forest
- Gradient Boosting

The notebook used five input features:

- Population in 1970
- Population in 2020
- Absolute population change
- Growth ratio
- Log-transformed population change

The target was the project-defined risk category. I used `LabelEncoder` to convert the category labels into numerical values.

`StandardScaler` was fitted on the training features and then applied to the test features.

## 5. Results

The notebook reported the following test-set accuracy:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 91.67% |
| Random Forest | 95.83% |
| Gradient Boosting | 95.83% |

Random Forest was selected as the final model in the notebook. Feature-importance analysis highlighted growth ratio and log change as influential features.

The Random Forest classification report showed an overall macro F1-score of approximately 0.88. The least-represented class had only two test samples, with a recall of 0.50, so the overall accuracy needs to be interpreted carefully.

### An important limitation of these results

The reported accuracy is not evidence that the model can independently predict future extinction risk.

The target categories were created from the population values for 1970 and 2020. The model then received those same values and features calculated from them. This creates a circular relationship between the inputs and the target: the model can learn the rule used to assign the categories.

The test score therefore measures how well the model reproduces these project-defined categories on a small held-out sample. It does not establish reliable forecasting performance on unseen species or future population data.

## 6. Exploring Geographical Patterns and Data Bias

The notebook also examines the distribution of species records across countries and continents.

In this separate analysis, I used a simplified growth-ratio rule to group records by population change and then compared the resulting country counts.

The notebook found that some countries, including the United States and the United Kingdom, contributed relatively high counts within this dataset-defined group. These counts do not establish that those countries have the highest real-world extinction rates.

Differences in monitoring effort, historical records, and reporting coverage may influence the patterns. The analysis raises the question of whether areas with more available data are more visible in the results.

The geographical findings should therefore be treated as exploratory, not as a global ranking of extinction severity.

## 7. Interactive Biodiversity Dashboard

The repository also includes a Streamlit application that explores live biodiversity occurrence records.

The app currently provides:

- Species search and comparison
- Geographic distribution maps
- Occurrence heatmaps
- Observation trends over time
- Country-level record summaries
- A simple observation-trend indicator
- External NASA Earth-data map layers for night lights and vegetation index
- A map overlay based on the distribution of occurrence records

The dashboard retrieves occurrence records from the GBIF API and uses the number and distribution of records to create some of its visual indicators.

**The dashboard's risk indicators are simplified proxies, not predictions from the notebook's Random Forest model.** Changes in recorded observations can reflect changes in data collection as well as changes in nature. The satellite layers are map imagery; they are not used as input features by the notebook's trained classifier.

## 8. Tools Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Plotly
- PyDeck
- GBIF API

## 9. What I Learned

This project helped me practise cleaning a large dataset, reshaping time-based data, creating features, training classification models, comparing evaluation metrics, and interpreting visualisations.

One of my main takeaways was that a high accuracy score is not enough to establish that a model answers the scientific question we intended to ask. The way a target is defined, the features supplied to a model, the number of test samples, and the quality of the underlying data all matter.

I also learned that geographical patterns need context. More records in a country do not automatically mean greater biodiversity loss there.

## 10. Possible Next Steps

- Build a more independent evaluation using features that do not directly determine the target labels.
- Test on species and time periods kept completely separate from model development.
- Compare the project-defined categories with independently assessed conservation status, where suitable data are available.
- Investigate missing data and uneven monitoring across regions.
- Connect the dashboard to a properly validated model if the two parts are developed into one system.

## Conclusion

This project is an exploration of historical population trends, classification methods, and the limitations of biodiversity data. Its main value is not a claim of near-perfect extinction prediction, but the practical experience of building a data-analysis workflow and examining how modelling choices affect scientific conclusions.
