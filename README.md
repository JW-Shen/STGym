# STGym: A Modular Benchmark for Spatio-Temporal Networks

STGym is a framework designed for the development, exploration, and evaluation of spatio-temporal networks. The modular design enhances understanding of model composition while enabling seamless adoption and extension of existing methods. It ensures reproducible and scalable experiments through a standardized training and evaluation pipeline, promoting fair comparisons across models.



## ✨ **Key Features**
### 🔧 **Modular Design**
Effortlessly explore various model architectures, simplifying the adoption and extension of existing methods.

### 🧪 **Standardized Pipelines**
Guarantee reproducible and scalable experiments with a standardized pipeline for training and evaluation, enabling fair comparisons across models and datasets.

### ⚙️ **Flexible Configuration**
Leverage **Hydra** for dynamic configuration, allowing easy command-line overrides to speed up experimentation — no need for managing multiple configuration files.

### 📊 **Automatic Tracking & Logging**
Integrates with **Weights & Biases (W&B)** for efficient tracking, logging, and recording of experiment results.



## 🚀 **Getting Started**

### 1️⃣ **Installing Dependencies**

#### 🐍 **Python**
- Python 3.7 or higher is required.

#### 🔥 **PyTorch**
- Install PyTorch according to your Python version and CUDA version.

#### 📦 **Other Dependencies**
Install all required dependencies via:
```
pip install -r requirements.txt
```
#### 💡 **Example Setups**
- Example 1: Python 3.8 + PyTorch 1.13.1 + CUDA 11.6
  ```
  conda create -n STGym python=3.8
  conda activate STGym
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  pip install -r requirements.txt
  ```
- Example 2: Python 3.11 + PyTorch 2.4.0 + CUDA 12.4
  ```
  conda create -n STGym python=3.11
  conda activate STGym
  pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
  pip install -r requirements.txt
  ```

### 2️⃣ **Downloading Datasets**
1. Download the dataset `raw.zip` from [Google Drive](https://drive.google.com/file/d/1-C8E9bJNbqAqjJF97LUpFRWQRG8n5g8p/view?usp=share_link).
2. Extract the files into the `./data/` directory:
  ```
  cd /path/to/STGym
  unzip /path/to/raw.zip -d data/
  ```

### 3️⃣ **Training & Evaluating Models**

#### 🛠️ **Train Your Own Model**

1. **Define Your Model**
    - Implement your model and place it in the `./modeling/sotas` directory.

2. **Define Model Configuration**
    - Create a configuration file (`.yaml`) to set the model parameters and save it in the `./config/model` directory.
    - Use the Hydra-based configuration system to override parameters such as scaler, optimizer, loss, and other hyperparameters from the command line.
    - Default settings can be found in the `./config/` directory.

3. **Train & Evaluate**
    - Run the following command to train and evaluate your model:
      ```
      python -m tools.main model=<MODEL_NAME> data=<DATASET_NAME>
      ```
      - Replace `<MODEL_NAME>` with your model's name.
      - Replace `<DATASET_NAME>` with any supported dataset or your own dataset (see instructions below for using custom datasets).

---

#### 📊 **Using Custom Datasets**
1. **Prepare Your Dataset**
    - Format your dataset to match the structure expected by STGym. Once you have downloaded the `raw.zip` file from [Google Drive](https://drive.google.com/file/d/1-C8E9bJNbqAqjJF97LUpFRWQRG8n5g8p/view?usp=share_link), you can refer to the data in `./raw` for examples.
    - Place your dataset in the `./data/` directory.
2. **Add Dataset Configuration**
    - Add a new configuration file for your dataset in the `./config/data/` directory.
    - Specify preprocessing steps, data paths, and any dataset-specific parameters.

---

#### 📄 **Reproducing Built-in Models**
To reproduce results for pre-built models, use:
```
python -m tools.run_sotas model=<MODEL_NAME> data=<DATASET_NAME>
```
Replace `<MODEL_NAME>` and `<DATASET_NAME>` with the desired built-in model and dataset.



## 📂 **Built-in Datasets and Baselines**

### **Built-in Datasets**

#### **Multi-step Datasets**

| **Dataset Name**  | **Task Type**   | **Nodes** | **Time Steps** | **Rate** | **Time Span** | **Data Splitting** | **Data Link** |
|-------------------|-----------------|-----------|----------------|----------|---------------|--------------------|---------------|
| METR-LA           | Traffic Speed   | 207       | 34272          | 5 min    | 4 months      | 7:1:2              | [Link](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph) |
| PEMS-BAY          | Traffic Speed   | 325       | 52116          | 5 min    | 6 months      | 7:1:2              | [Link](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph) |
| PEMS03            | Traffic Flow    | 358       | 26208          | 5 min    | 3 months      | 6:2:2              | [Link](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |
| PEMS04            | Traffic Flow    | 307       | 16992          | 5 min    | 2 months      | 6:2:2              | [Link](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |
| PEMS07            | Traffic Flow    | 883       | 28224          | 5 min    | 4 months      | 6:2:2              | [Link](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |
| PEMS08            | Traffic Flow    | 170       | 17856          | 5 min    | 2 months      | 6:2:2              | [Link](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |

#### **Single-step Datasets**

| **Dataset Name**  | **Task Type**           | **Nodes** | **Time Steps** | **Rate** | **Time Span** | **Data Splitting** | **Data Link** |
|-------------------|-------------------------|-----------|----------------|----------|---------------|--------------------|---------------|
| Electricity       | Electricity Consumption | 321       | 26304          | 1 hour   | 3 years       | 6:2:2              | [Link](https://github.com/laiguokun/multivariate-time-series-data) |
| Solar Energy      | Solar Power Production  | 137       | 52560          | 10 min   | 1 year        | 6:2:2              | [Link](https://github.com/laiguokun/multivariate-time-series-data) |
| Traffic           | Road Occupancy Rates    | 862       | 17544          | 1 hour   | 2 years       | 6:2:2              | [Link](https://github.com/laiguokun/multivariate-time-series-data) |
| Exchange Rate     | Exchange Rate           | 8         | 7588           | 1 day    | 27 years      | 6:2:2              | [Link](https://github.com/laiguokun/multivariate-time-series-data) |

---

### Baseline Models

#### **Multi-step Forecasting Models**

| **Model Name** | **Year** | **Venue** | **Paper Title**                                               | **Code**                                 |
|----------------|----------|-----------|---------------------------------------------------------------|------------------------------------------|
| DCRNN          | 2018     | ICLR      | [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](http://arxiv.org/abs/1707.01926) | [Link](http://github.com/liyaguang/DCRNN) |
| STGCN          | 2018     | IJCAI     | [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875) | [Link](https://github.com/VeritasYin/STGCN_IJCAI-18) |
| GWNet          | 2019     | IJCAI     | [Graph WaveNet for Deep Spatial-Temporal Graph Modeling](https://arxiv.org/abs/1906.00121) | [Link](https://github.com/nnzhan/Graph-WaveNet) |
| MTGNN          | 2020     | KDD       | [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/abs/2005.11650) | [Link](https://github.com/nnzhan/MTGNN) |
| STSGCN         | 2020     | AAAI      | [Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/5438) | [Link](https://github.com/Davidham3/STSGCN) |
| AGCRN          | 2020     | NeurIPS   | [Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting](https://arxiv.org/abs/2007.02842) | [Link](https://github.com/LeiBAI/AGCRN) |
| GMAN           | 2020     | AAAI      | [GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://arxiv.org/abs/1911.08415) | [Link](https://github.com/zhengchuanpan/GMAN) |
| DGCRN          | 2021     | TKDD      | [Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution](https://arxiv.org/abs/2104.14917) | [Link](https://github.com/tsinghua-fib-lab/Traffic-Benchmark) |
| GTS            | 2021     | ICML      | [Discrete Graph Structure Learning for Forecasting Multiple Time Series](https://arxiv.org/abs/2101.06861) | [Link](https://github.com/chaoshangcs/GTS) |
| STNorm         | 2021     | KDD       | [ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting](https://dl.acm.org/doi/10.1145/3447548.3467330) | [Link](https://github.com/JLDeng/ST-Norm) |
| STID           | 2022     | CIKM      | [Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting](https://arxiv.org/abs/2208.05233)  | [Link](https://github.com/GestaltCogTeam/STID) |
| SCINet         | 2022     | NeurIPS   | [SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction](https://arxiv.org/abs/2106.09305) | [Link](https://github.com/cure-lab/SCINet) |
| STAEformer     | 2023     | CIKM      | [STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting](https://arxiv.org/abs/2308.10425) | [Link](https://github.com/XDZhelheim/STAEformer) |
| MegaCRN        | 2023     | AAAI      | [Spatio-Temporal Meta-Graph Learning for Traffic Forecasting](https://arxiv.org/abs/2211.14701) | [Link](https://github.com/deepkashiwa20/MegaCRN) |

#### **Single-step Forecasting Models**

| **Model Name** | **Year** | **Venue** | **Paper Title**                                               | **Code**                                 |
|----------------|----------|-----------|---------------------------------------------------------------|------------------------------------------|
| LST-Skip       | 2018     | SIGIR     | [Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks](https://arxiv.org/abs/1703.07015) | [Link](https://github.com/laiguokun/LSTNet) |
| TPA-LSTM       | 2019     | ECML/PKDD | [Temporal Pattern Attention for Multivariate Time Series Forecasting](https://arxiv.org/abs/1809.04206) | [Link](https://github.com/shunyaoshih/TPA-LSTM) |
| MTGNN          | 2020     | KDD       | [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/abs/2005.11650) | [Link](https://github.com/nnzhan/MTGNN) |
| SCINet         | 2022     | NeurIPS   | [SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction](https://arxiv.org/abs/2106.09305) | [Link](https://github.com/cure-lab/SCINet) |
| Linear         | 2023     | AAAI      | [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) | [Link](https://github.com/cure-lab/LTSF-Linear) |
| NLinear        | 2023     | AAAI      | [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) | [Link](https://github.com/cure-lab/LTSF-Linear) |
| DLinear        | 2023     | AAAI      | [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) | [Link](https://github.com/cure-lab/LTSF-Linear) |


## 📈 **Main Results**

The detailed results and performance comparisons can be found [here]("./result/README.md").



## 💌 **Feedback & Support**

Feel free to <a href="https://github.com/JiangJiaWei1103/STGym/issues">open an issue</a> if you have any questions.