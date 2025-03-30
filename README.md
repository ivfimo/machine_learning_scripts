# Here is the ML training for (Deutsch et al., 2024 https://pubmed.ncbi.nlm.nih.gov/39314370/)
# **TensorFlow-Keras Binary Classification Model**
-   1. Create a Google Colab file https://colab.research.google.com/#create=true (Google Account required)
    2. Ensure you have the following Python libraries installed !pip install pandas numpy matplotlib scikit-learn tensorflow shap
    3. Copy the Python script from the TF-K_script.md file and paste it on your Google Colab file
    5. Run the TF-K_script using the TF-K_dataset.csv file as input ("/content/TF-K_dataset.csv") and view the results
-      See README_detailed_ML_approaches_description.md for more details on the approach
-	   See the SHAP plot showing the top variables of importance
-      The reader can expect an AUC in the range 0.66 to 0.68 depending on the random selection of test/train 
# **Binary Classification using MLPClassifier**
-   1. Create a Google Colab file https://colab.research.google.com/#create=true or use the Google Colab file you already created
    2. Ensure you have the following Python libraries installed !pip install numpy pandas scikit-learn matplotlib shap
    3. Copy the Python script from the MLPClassifier_script.md file and paste it on your Google Colab file
    4. Upload the MLPClassifier_dataset.csv file to your Google Colab environment
    5. Run the script using the MLPClassifier_dataset.csv file as input ("/content/MLPClassifier_dataset.csv") and view the results
-      See README_detailed_ML_approaches_description.md for more details on the approach
-	   See the variable importance plot and the SHAP plot showing the top variables of importance
-      The reader can expect an AUC in the range 0.68 to 0.69 depending on the random selection of test/train
