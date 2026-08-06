# Book Scraping and Data Analysis using Scrapy

## Student Information

**Name:** *Bhumi Halatwala*
**Student ID:** *202618038*

---

## Project Overview

This project demonstrates a complete data pipeline using **Python** and **Scrapy** to collect book information from the **Books to Scrape** website. The scraped data was cleaned, transformed, analyzed, and visualized to extract meaningful insights.

The project covers the complete workflow from web scraping to data preprocessing, feature engineering, visualization, and interpretation of results.

---

## Objective

The objective of this project is to:

* Scrape book information from the Books to Scrape website using Scrapy.
* Clean and preprocess the collected data.
* Perform feature engineering for better analysis.
* Generate visualizations to understand patterns in the dataset.
* Interpret the findings using descriptive statistics and visual analysis.

---

## Website

https://books.toscrape.com/

---

## Technologies Used

* Python
* Scrapy
* Pandas
* NumPy
* Matplotlib
* Seaborn
* WordCloud
* Jupyter Notebook

---

## Dataset

A total of **100 books** were scraped from the first **five catalogue pages**.

The following attributes were collected for every book:

* Title
* Category
* Price
* Rating
* Availability
* Product Description
* UPC
* Number of Reviews
* Product URL

---

## Project Structure

```text
bookscraper/
│
├── scrapy.cfg
├── raw_books.csv
├── cleaned_books.csv
├── analysis.ipynb
│
└── bookscraper/
    ├── settings.py
    └── spiders/
        └── books_spider.py
```

---

## Task 1 – Data Scraping

The Scrapy spider:

* Crawls the first five catalogue pages.
* Visits every individual book page.
* Extracts all required fields.
* Exports the collected data into a CSV file.

The project also reports:

* Total records scraped
* Missing values
* Duplicate UPC values

---

## Task 2 – Data Preprocessing

The preprocessing stage includes:

* Removing extra spaces and inconsistent text
* Removing duplicate books using UPC
* Handling missing descriptions
* Converting prices to numeric values
* Mapping ratings from text to integers
* Extracting available stock count

### Engineered Features

* Description Word Count
* Price Band
* Recommended

---

## Task 3 – Visualization and Analysis

The following visualizations were created:

* Price Distribution
* Rating Distribution
* Average Price by Category
* Price vs Rating Relationship
* Word Cloud from Book Descriptions

Summary statistics were also generated to analyze:

* Category distribution
* Average prices
* Average ratings
* Stock availability
* Missing values

---

## Key Findings

* The dataset contains **100 books** with no missing values or duplicate UPCs.
* Sequential Art is the most represented category.
* Historical Fiction has the highest average price among the scraped categories.
* Most books have ratings between **2 and 3**.
* No strong relationship was observed between book price and rating.
* Books marked as **Recommended** provide better value by combining higher ratings with relatively lower prices.

---

## Limitations

* The analysis is based on only the first **100 books** from five catalogue pages.
* The Books to Scrape website is a practice dataset and all books contain **0 customer reviews**, limiting review-based analysis.
* The results represent only the scraped sample and should not be generalized to the complete catalogue.

---

## How to Run the Project

### 1. Install the required libraries

```bash
pip install scrapy pandas numpy matplotlib seaborn wordcloud jupyter notebook
```

### 2. Navigate to the project directory

```bash
cd bookscraper
```

### 3. Run the Scrapy spider

```bash
scrapy crawl books -O raw_books.csv
```

### 4. Open the notebook

Run all cells in:

```text
202618038_Lab_1.ipynb
```

This will:

* Clean the dataset
* Generate engineered features
* Produce visualizations
* Export the cleaned dataset
* Display the final analysis and insights

---

## Author

**Name:** *Bhumi Halatwala*
**Student ID:** *202618038*
