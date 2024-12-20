# RCRank

This is the official PyTorch code for RCRank.

We propose RCRank, the first method to utilize a multimodal approach for identifying and ranking the root causes of slow queries by estimating their impact. We employ a pre-training method to align the multimodal information of queries, enhancing the performance and training speed of root cause impact estimated. Based on the aligned pre-trained embedding module, we use Cross-modal fusion of feature modalities to ultimately estimate the impact of root causes, identifying and ranking the root causes of slow queries.

## Installation
First clone the repository using Git.

Please download the Bert model and parameters.
```bash
git lfs install
git clone https://huggingface.co/google-bert/bert-base-uncased
```

The project dependencies can be installed by executing the following commands in the root of the repository:
```bash
conda create --name RCRank python=3.9
conda activate RCRank
pip install -r requirements.txt
```

## Run

Our method is divided into two stages. The first stage is the pre-training stage. For convenience, we provide **pre-trained checkpoint**, which can be downloaded via this [link](https://drive.google.com/file/d/1ar52Ih9ADbB4TX2NXE4PNfi4HdFfyv-t/view?usp=drive_link).
After downloading the **pre-trained checkpoint**, you can place it in the `./pretrain/` directory.

The second stage is the training and inference stage, **datasets** can be downloaded from this [link](https://drive.google.com/file/d/1u9Ne2fqSzzeQ1Nd24DeEeWUxyFddhexW/view?usp=sharing). Then run follow script to place the data files into the `data` directory.
```bash
mkdir data
mv train_data.zip ./data/
unzip train_data.zip
```

### TPC-H dataset
Run the following script to obtain the results for the TPC-H dataset.
```bash
python main.py --device cuda:0 --dataset tpc_h
```
The execution results of this script correspond to the TPC-H results in Table 3 in the paper.

### TPC-C dataset
Run the following script to obtain the results for the TPC-C dataset.
```bash
python main.py --device cuda:0 --dataset tpc_c
```
The execution results of this script correspond to the TPC-C results in Table 3 in the paper.

### TPC-DS dataset
Run the following script to obtain the results for the TPC-DS dataset.
```bash
python main.py --device cuda:0 --dataset tpc_ds
```
The execution results of this script correspond to the TPC-DS results in Table 4 in the paper.

The final results will be saved in the `res` directory.

### Hologres dataset

The Hologres dataset uses the same code as the public datasets. However, since it is a private dataset, we have only open-sourced the TPC-C, TPC-H, and TPC-DS datasets.

## Detailed guide

#### Pre-train

If you need to pre-train from scratch, you can execute the following method. Please download the pre-training data from this [link](https://drive.google.com/file/d/1ZkVLYl9gV5GnkD_Uv3G3a9VA-WrgRNhT/view?usp=drive_link), place it in the `./pretrain/` directory, and execute the following script. The checkpoint will be saved in the `./pretrain/save` directory.
```bash
python pretrain/pretrain.py
```

#### Config
Configuration File: config.py - Training Parameters

- **batch_size**: Number of samples per batch during training.
- **lr**: Learning rate.
- **epoch**: Number of training iterations.
- **device**: Device for computation, either GPU or CPU.
- **opt_threshold**: Threshold for valid root causes.
- **model_name**: Name of the model.
- **use_fuse_model**: Whether to use a fused model.
- **use_threshold_loss**: Whether to use validity and orderliness loss.
- **model_path**: Path to save the model.
- **margin_loss_type**: Margin loss type: "MarginLoss"、"ListnetLoss"、"ListMleLoss".
- **use_margin_loss**: Whether to use margin loss.
- **use_label_loss**: Whether to predict root cause type.
- **use_weight_loss**: Whether to assign sample weights.
- **use_log**: Whether to use log information.
- **use_metrics**: Whether to use metrics information.
- **embed_size**: "emb_size" in QueryFormer.
- **embed_size**: "emb_size" in QueryFormer.
- **pred_hid**: "pred_hid" in QueryFormer.
- **ffn_dim**: "ffn_dim" in QueryFormer.
- **head_size**: "head_size" in QueryFormer.
- **n_layers**: "n_layers" in QueryFormer.
- **dropout**: "dropout" in QueryFormer.
- **input_emb**: "dropout" in QueryFormer.
- **use_sample**: "use_sample" in QueryFormer.
- **ts_weight**: Weight for validity and orderliness.
- **mul_label_weight**: Weight for type of root cause.
- **pred_type**: Type of task.

Complete Directory Structure
```bash
RCRank
├─ data  # TPC-C, TPC-H, TPC-DS datasets
│  ├─ tpc_c.csv  # TPC-C dataset
│  ├─ tpc_ds.csv  # TPC-DS dataset
│  ├─ tpc_h.csv  # TPC-H dataset
├─ bert-base-uncased # Bert model and parameters directory
├─ model  # model, loss function, train and test
│  ├─ __init__.py
│  ├─ loss
│  │  └─ loss.py
│  ├─ modules
│  │  ├─ FuseModel
│  │  │  ├─ Attention.py
│  │  │  └─ CrossTransformer.py
│  │  ├─ LogModel
│  │  │  ├─ __init__.py
│  │  │  └─ log_model.py
│  │  ├─ QueryFormer
│  │  │  ├─ QueryFormer.py
│  │  │  ├─ datasetQF.py
│  │  │  └─ utils.py
│  │  ├─ TSModel
│  │  │  └─ ts_model.py
│  │  ├─ __init__.py
│  │  ├─ rcrank_model.py
│  └─ train_test.py  # train and test
├─ pretrain  # pretrain function and pretrain parameters
│  └─ pretrain.py  
│  └─ pretrain.pth  
├─ res  # Test result save directory
│  └─ GatePretrainModel tpc_c confidence eta0.07
│     └─ res.txt
├─ utils
│   ├─ config.py
│   ├─ data_tensor.py
│   ├─ evaluate.py
│   ├─ load_data.py
│   └─ plan_encoding.py
├─ main.py  # Root cause rank
├─ README.md
├─ requirements.txt
```

