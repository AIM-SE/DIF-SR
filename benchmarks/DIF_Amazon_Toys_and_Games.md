## DIF_Amazon_Beauty

A notebook to benchmark DIF-SR on Amazon_Toys_and_Games dataset.

Author: Peilin Zhou, Upstage


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

    ```bash
    CPU: AMD Ryzen Threadripper 3970X 32-Core Processor
    RAM: 256
    GPU: NVIDIA GeForce RTX 3090
    ```
+ Software

    ```python
    python: 3.8.0
    pytorch 1.8.1
    numpy: 1.20.3
    scipy: 1.6.3
    tensorboard: 2.7.0
    ```

### Dataset
Following commonly used pre-processing steps in SR,
we remove all items and users that occur less than five times in
these datasets. All the interactions are regarded as implicit feedback.

You could download datasets from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or their [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). 
### Code

Our implementation of DIF-SR is based on Recbole. You can run the experiments using following commands:
```bash
python main.py --config_file amazon_config.ini
```

### Results
Recall@20 = 0.1382

NDCG@20 = 0.0597

### Logs
```bash
18 Jan 11:18    INFO  
General Hyper Parameters:
gpu_id = 2
use_gpu = True
seed = 212
state = INFO
reproducibility = True
data_path = dataset/Amazon_Toys_and_Games
show_progress = True
save_dataset = False
save_dataloaders = False
benchmark_filename = None

Training Hyper Parameters:
checkpoint_dir = saved/Amazon_Toys_and_Games/SASRecD_3_2_['categories']_[128]_concat_[5]_212
epochs = 200
train_batch_size = 2048
learner = adam
learning_rate = 0.0001
eval_step = 2
stopping_step = 10
clip_grad_norm = None
weight_decay = 0.0
loss_decimal_place = 4

Evaluation Hyper Parameters:
eval_args = {'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'full'}
metrics = ['Recall', 'NDCG']
topk = [5, 10, 20]
valid_metric = Recall@10
valid_metric_bigger = True
eval_batch_size = 256
metric_decimal_place = 4

Dataset Hyper Parameters:
field_separator = 	
seq_separator =  
USER_ID_FIELD = user_id
ITEM_ID_FIELD = item_id
RATING_FIELD = rating
TIME_FIELD = timestamp
seq_len = None
LABEL_FIELD = label
threshold = None
NEG_PREFIX = neg_
load_col = {'inter': ['user_id', 'item_id', 'rating', 'timestamp'], 'item': ['item_id', 'title', 'sales_rank', 'price', 'brand', 'categories', 'sales_type']}
unload_col = None
unused_col = None
additional_feat_suffix = None
rm_dup_inter = None
val_interval = None
filter_inter_by_user_or_item = True
user_inter_num_interval = [5,inf)
item_inter_num_interval = [5,inf)
alias_of_user_id = None
alias_of_item_id = None
alias_of_entity_id = None
alias_of_relation_id = None
preload_weight = None
normalize_field = None
normalize_all = None
ITEM_LIST_LENGTH_FIELD = item_length
LIST_SUFFIX = _list
MAX_ITEM_LIST_LENGTH = 7
POSITION_FIELD = position_id
HEAD_ENTITY_ID_FIELD = head_id
TAIL_ENTITY_ID_FIELD = tail_id
RELATION_ID_FIELD = relation_id
ENTITY_ID_FIELD = entity_id

Other Hyper Parameters: 
neg_sampling = None
repeatable = True
n_layers = 3
n_heads = 2
hidden_size = 256
inner_size = 256
hidden_dropout_prob = 0.5
attn_dropout_prob = 0.3
hidden_act = gelu
layer_norm_eps = 1e-12
initializer_range = 0.02
selected_features = ['categories']
pooling_mode = sum
loss_type = CE
MODEL_TYPE = ModelType.SEQUENTIAL
attribute_hidden_size = [128]
weight_sharing = not
fusion_type = concat
lamdas = [5]
attribute_predictor = linear
predictor_source = item
behavior_feature = not
MODEL_INPUT_TYPE = InputType.POINTWISE
eval_type = EvaluatorType.RANKING
device = cuda
train_neg_sample_args = {'strategy': 'none'}
eval_neg_sample_args = {'strategy': 'full', 'distribution': 'uniform'}


18 Jan 11:19    INFO  Amazon_Toys_and_Games
The number of users: 19413
Average actions of users: 8.633680197815783
The number of items: 11925
Average actions of items: 14.055434417980543
The number of inters: 167597
The sparsity of the dataset: 99.92760389550713%
Remain Fields: ['user_id', 'item_id', 'rating', 'timestamp', 'title', 'price', 'sales_type', 'sales_rank', 'brand', 'categories']
18 Jan 11:19    INFO  [Training]: train_batch_size = [2048] negative sampling: [None]
18 Jan 11:19    INFO  [Evaluation]: eval_batch_size = [256] eval_args: [{'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'full'}]
18 Jan 11:19    INFO  SASRecD(
  (item_embedding): Embedding(11925, 256, padding_idx=0)
  (position_embedding): Embedding(7, 256)
  (feature_embed_layer_list): ModuleList(
    (0): FeatureSeqEmbLayer()
  )
  (trm_encoder): DeTransformerEncoder(
    (layer): ModuleList(
      (0): DeTransformerLayer(
        (multi_head_attention): DeMultiHeadAttention(
          (query): Linear(in_features=256, out_features=256, bias=True)
          (key): Linear(in_features=256, out_features=256, bias=True)
          (value): Linear(in_features=256, out_features=256, bias=True)
          (query_p): Linear(in_features=256, out_features=256, bias=True)
          (key_p): Linear(in_features=256, out_features=256, bias=True)
          (query_layers): ModuleList(
            (0): Linear(in_features=128, out_features=128, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=128, out_features=128, bias=True)
          )
          (fusion_layer): Linear(in_features=21, out_features=7, bias=True)
          (attn_dropout): Dropout(p=0.3, inplace=False)
          (dense): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.5, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=256, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
      )
      (1): DeTransformerLayer(
        (multi_head_attention): DeMultiHeadAttention(
          (query): Linear(in_features=256, out_features=256, bias=True)
          (key): Linear(in_features=256, out_features=256, bias=True)
          (value): Linear(in_features=256, out_features=256, bias=True)
          (query_p): Linear(in_features=256, out_features=256, bias=True)
          (key_p): Linear(in_features=256, out_features=256, bias=True)
          (query_layers): ModuleList(
            (0): Linear(in_features=128, out_features=128, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=128, out_features=128, bias=True)
          )
          (fusion_layer): Linear(in_features=21, out_features=7, bias=True)
          (attn_dropout): Dropout(p=0.3, inplace=False)
          (dense): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.5, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=256, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
      )
      (2): DeTransformerLayer(
        (multi_head_attention): DeMultiHeadAttention(
          (query): Linear(in_features=256, out_features=256, bias=True)
          (key): Linear(in_features=256, out_features=256, bias=True)
          (value): Linear(in_features=256, out_features=256, bias=True)
          (query_p): Linear(in_features=256, out_features=256, bias=True)
          (key_p): Linear(in_features=256, out_features=256, bias=True)
          (query_layers): ModuleList(
            (0): Linear(in_features=128, out_features=128, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=128, out_features=128, bias=True)
          )
          (fusion_layer): Linear(in_features=21, out_features=7, bias=True)
          (attn_dropout): Dropout(p=0.3, inplace=False)
          (dense): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.5, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=256, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
      )
    )
  )
  (ap): ModuleList(
    (0): Linear(in_features=256, out_features=730, bias=True)
  )
  (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (loss_fct): CrossEntropyLoss()
  (attribute_loss_fct): BCEWithLogitsLoss()
)
Trainable parameters: 4924328
SASRecD
['configs/config_d_Amazon_Toys_and_Games.yaml', 'configs/config_training.yaml', 'configs/config_m_SASRecD.yaml']

Train     0:   0%|                                                           | 0/54 [00:00<?, ?it/s]total_loss:12.974088668823242	item_loss:9.444479942321777	attribute_categories_loss:0.7059217095375061

Train     0:   0%|                                  | 0/54 [00:00<?, ?it/s, GPU RAM: 1.40 G/23.70 G]
Train     0:   2%|▍                         | 1/54 [00:00<00:26,  2.01it/s, GPU RAM: 1.40 G/23.70 G]total_loss:12.973123550415039	item_loss:9.445937156677246	attribute_categories_loss:0.7054372429847717

Train     0:   2%|▍                         | 1/54 [00:00<00:26,  2.01it/s, GPU RAM: 1.84 G/23.70 G]
Train     0:   4%|▉                         | 2/54 [00:00<00:15,  3.41it/s, GPU RAM: 1.84 G/23.70 G]total_loss:12.96281623840332	item_loss:9.438323020935059	attribute_categories_loss:0.7048985958099365

Train     0:   4%|▉                         | 2/54 [00:00<00:15,  3.41it/s, GPU RAM: 1.84 G/23.70 G]
Train     0:   6%|█▍                        | 3/54 [00:00<00:10,  4.73it/s, GPU RAM: 1.84 G/23.70 G]total_loss:12.964868545532227	item_loss:9.442314147949219	attribute_categories_loss:0.7045109868049622

Train     0:   6%|█▍                        | 3/54 [00:00<00:10,  4.73it/s, GPU RAM: 1.84 G/23.70 G]
Train     0:   7%|█▉                        | 4/54 [00:00<00:09,  5.29it/s, GPU RAM: 1.84 G/23.70 G]total_loss:12.960132598876953	item_loss:9.439766883850098	attribute_categories_loss:0.7040732502937317

Train     0:   7%|█▉                        | 4/54 [00:00<00:09,  5.29it/s, GPU RAM: 1.84 G/23.70 G]total_loss:12.960346221923828	item_loss:9.443694114685059	attribute_categories_loss:0.7033304572105408

Train     0:   7%|█▉                        | 4/54 [00:01<00:09,  5.29it/s, GPU RAM: 1.84 G/23.70 G]
Train     0:  11%|██▉                       | 6/54 [00:01<00:07,  6.64it/s, GPU RAM: 1.84 G/23.70 G]total_loss:12.941118240356445	item_loss:9.427778244018555	attribute_categories_loss:0.7026680111885071

Train     0:  11%|██▉                       | 6/54 [00:01<00:07,  6.64it/s, GPU RAM: 1.84 G/23.70 G]
Train     0:  13%|███▎                      | 7/54 [00:01<00:06,  7.25it/s, GPU RAM: 1.84 G/23.70 G]total_loss:12.952522277832031	item_loss:9.440581321716309	attribute_categories_loss:0.7023882865905762

Train     0:  13%|███▎                      | 7/54 [00:01<00:06,  7.25it/s, GPU RAM: 1.84 G/23.70 G]
Train     0:  15%|███▊                      | 8/54 [00:01<00:06,  7.42it/s, GPU RAM: 1.84 G/23.70 G]total_loss:12.941692352294922	item_loss:9.434090614318848	attribute_categories_loss:0.7015202641487122

Train     0:  15%|███▊                      | 8/54 [00:01<00:06,  7.42it/s, GPU RAM: 1.84 G/23.70 G]
Train     0:  17%|████▎                     | 9/54 [00:01<00:05,  7.78it/s, GPU RAM: 1.84 G/23.70 G]total_loss:12.953174591064453	item_loss:9.448530197143555	attribute_categories_loss:0.7009289860725403
...
Evaluate   :  93%|███████████████████████▎ | 71/76 [00:09<00:00,  7.47it/s, GPU RAM: 1.85 G/23.70 G]
Evaluate   :  93%|███████████████████████▎ | 71/76 [00:09<00:00,  7.47it/s, GPU RAM: 1.85 G/23.70 G]
Evaluate   :  95%|███████████████████████▋ | 72/76 [00:09<00:00,  6.83it/s, GPU RAM: 1.85 G/23.70 G]
Evaluate   :  95%|███████████████████████▋ | 72/76 [00:09<00:00,  6.83it/s, GPU RAM: 1.85 G/23.70 G]
Evaluate   :  96%|████████████████████████ | 73/76 [00:09<00:00,  6.90it/s, GPU RAM: 1.85 G/23.70 G]
Evaluate   :  96%|████████████████████████ | 73/76 [00:09<00:00,  6.90it/s, GPU RAM: 1.85 G/23.70 G]
Evaluate   :  97%|████████████████████████▎| 74/76 [00:09<00:00,  6.91it/s, GPU RAM: 1.85 G/23.70 G]
Evaluate   :  97%|████████████████████████▎| 74/76 [00:09<00:00,  6.91it/s, GPU RAM: 1.85 G/23.70 G]
Evaluate   :  99%|████████████████████████▋| 75/76 [00:09<00:00,  6.87it/s, GPU RAM: 1.85 G/23.70 G]
Evaluate   :  99%|████████████████████████▋| 75/76 [00:09<00:00,  6.87it/s, GPU RAM: 1.85 G/23.70 G]
Evaluate   : 100%|█████████████████████████| 76/76 [00:09<00:00,  6.95it/s, GPU RAM: 1.85 G/23.70 G]
Evaluate   : 100%|█████████████████████████| 76/76 [00:09<00:00,  7.70it/s, GPU RAM: 1.85 G/23.70 G]
18 Jan 12:06    INFO  best valid : {'recall@5': 0.082, 'recall@10': 0.1207, 'recall@20': 0.1632, 'ndcg@5': 0.0466, 'ndcg@10': 0.0591, 'ndcg@20': 0.0698}
18 Jan 12:06    INFO  test result: {'recall@5': 0.0683, 'recall@10': 0.1013, 'recall@20': 0.1382, 'ndcg@5': 0.0397, 'ndcg@10': 0.0504, 'ndcg@20': 0.0597}

```