# Satellite-Imagery-Based-Property-Valuation

This repository contains a **Multimodal Regression Pipeline** designed to predict property market values by integrating traditional tabular data with high-dimensional embeddings extracted from satellite imagery. By programmatically acquiring visual data based on coordinates, the model captures environmental contexts—such as green cover and road density—that traditional pricing models often miss.

---

## 🚀 Overview

The core objective is to move beyond standard house-price modeling by utilizing a **late-fusion architecture** that combines structured tabular data and unstructured visual data into a unified XGBoost framework.

### Key Features

* **Multimodal Integration**: Fuses tabular features (sqft, bedrooms, etc.) with satellite image embeddings.


* **Transfer Learning**: Utilizes a pre-trained **EfficientNet-B4** CNN as a feature extractor to recognize neighborhood "curb appeal".


* **Visual Compression**: Employs **Principal Component Analysis (PCA)** to reduce 1,792 raw image features to the 12 most significant components, retaining 40% variance while preventing overfitting.


* **Advanced Feature Engineering**: Includes target encoding for zipcodes, temporal decomposition of sale dates, and "Effective House Age" calculations based on renovation data.



---

## 📂 Project Structure

The project is organized into modular notebooks for clear reproducibility:

* **`01_basic_feature_engineering.ipynb`**: Initial data cleaning and basic feature creation.
* **`02_advanced_feature_engineering.ipynb`**: Implementation of target encoding and complex temporal features.


* **`03_hyperparameter_tuning_without_img_encodings.ipynb`**: Baseline model optimization using only tabular data.
* **`04_img_fetcher.ipynb`**: Script to programmatically acquire satellite images using latitude/longitude coordinates.


* **`05_img_to_encoding.ipynb`**: Extracting high-dimensional vectors (embeddings) via Transfer Learning.


* **`06_hyperparameter_tuning_with_img_encodings.ipynb`**: Fine-tuning the multimodal XGBoost framework.
* **`07_final_model_training.ipynb`**: Final training loop and generation of the prediction submission file.

---

## 📊 Performance Summary

We compared a tabular-only baseline against our multimodal approach using Cross-Validation (CV) on training data.

| Model Version | Features Used | RMSE |  Score |
| --- | --- | --- | --- |
| **Baseline** | Tabular Only | 107,575.09 | 0.895 |
| **Multimodal** | Tabular + PCA Image Embeddings | 111,736.39 | 0.885 |

---

## 🛠️ Tech Stack

* **Data Handling**: Pandas, NumPy
* **Machine Learning**: XGBoost, Scikit-learn
* **Deep Learning**: CNN (EfficientNet-B4) via Transfer Learning
* **Visualizations**: Matplotlib, Seaborn, Plotly

---

## 🏁 Conclusion & Future Work

This pipeline provides a holistic valuation framework by fusing visual environmental metrics with traditional real estate data. Future iterations will focus on fine-tuning the CNN backbone specifically on satellite-view datasets to further improve the granularity of visual feature extraction.
