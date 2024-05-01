# Project Documentation

# Enhancing Data Security in Healthcare with Synthetic Data Generation: An Auto-Encoder and Variational Auto-Encoder Approach

### Abstract
The advent of machine learning and artificial intelligence (AI) in healthcare has revolutionized data analysis and patient care. This thesis addresses these challenges by exploring the application of Auto-Encoders (AEs) and Variational Auto-Encoders (VAEs) in synthetic healthcare data generation, enhancing data security and producing synthetic data that upholds privacy while retaining utility for AI applications.

### Installation Instructions
**Prerequisites:**
- **Operating System**: Windows 10 (64-bit), macOS, or Linux
- **Python Version**: 3.8 or higher

**Environment Setup:**
1. **Install Python**: Download from [Python's official website](https://www.python.org/downloads/).
2. **Install Jupyter Notebook**: Run `pip install notebook`.

**Clone the Repository**:
```bash
git clone https://github.com/Kelechiede/Enhancing_Data_Security_with_AE_and_VAE_Synthetic_Data_Generation.git
cd Enhancing_Data_Security_with_AE_and_VAE_Synthetic_Data_Generation
Usage
Navigate to the notebooks/ directory:

bash
Copy code
cd notebooks
Launch Jupyter Notebook:

bash
Copy code
jupyter notebook
Open the respective .ipynb notebook files to view the demonstrations and analysis for each dataset.

Project Structure
data/: Contains raw and processed datasets used in the project.
src/: Source code for the Autoencoder models and privacy risk evaluations.
notebooks/: Jupyter notebooks that demonstrate the project workflow and results.
Ensuring Privacy Preservation and Data Utility
This project utilizes autoencoders to generate synthetic data, minimizing the risk of exposing original data entries. Our methodology ensures that the synthetic data maintains utility for downstream tasks while significantly reducing privacy risks.

Control Dataset
The control dataset helps benchmark privacy leakage and ensures that our evaluations accurately reflect the privacy-preserving capabilities of our synthetic data generation methods.

### About the Anonymeter Tool
For more information on the Anonymeter tool, including installation and usage, please visit the authors and owners website and github page via the links below:
-[Anonymeter Official Website](https://www.anonos.com/blog/presenting-anonymeter-the-tool-for-assessing-privacy-risks-in-synthetic-datasets)
-[Anonymeter GitHub Repository](https://github.com/statice/anonymeter/commits?author=MatteoGiomi)

Author of the above named project: Enhancing_Data_Security_with_AE_and_VAE_Synthetic_Data_Generation
Kelechukwu Innocent Ede

Email: kelechukwuede@gmail.com
License
This project is open source and available under the MIT License.

Acknowledgments
Special thanks to the authors and contributors of the Anonymeter tool for their open-source contributions to privacy technology.
