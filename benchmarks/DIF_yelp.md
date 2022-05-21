## DIF_Amazon_Beauty

A notebook to benchmark DIF-SR on yelp dataset.

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
Recall@20 = 0.1003

NDCG@20 = 0.0496

### Logs
```bash
20 Jan 21:56    INFO  
General Hyper Parameters:
gpu_id = 2
use_gpu = True
seed = 212
state = INFO
reproducibility = True
data_path = dataset/yelp
show_progress = True
save_dataset = False
save_dataloaders = False
benchmark_filename = None

Training Hyper Parameters:
checkpoint_dir = saved/yelp/SASRecD_4_2_['categories']_[64]_concat_[10]_212
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
ITEM_ID_FIELD = business_id
RATING_FIELD = stars
TIME_FIELD = date
seq_len = None
LABEL_FIELD = label
threshold = None
NEG_PREFIX = neg_
load_col = {'inter': ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'date'], 'item': ['business_id', 'item_name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'item_stars', 'item_review_count', 'is_open', 'categories']}
unload_col = None
unused_col = None
additional_feat_suffix = None
rm_dup_inter = None
val_interval = {'date': '[1546264800,1577714400]'}
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
MAX_ITEM_LIST_LENGTH = 10
POSITION_FIELD = position_id
HEAD_ENTITY_ID_FIELD = head_id
TAIL_ENTITY_ID_FIELD = tail_id
RELATION_ID_FIELD = relation_id
ENTITY_ID_FIELD = entity_id

Other Hyper Parameters: 
neg_sampling = None
repeatable = True
n_layers = 4
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
attribute_hidden_size = [64]
weight_sharing = not
fusion_type = concat
lamdas = [10]
attribute_predictor = linear
predictor_source = item
behavior_feature = not
MODEL_INPUT_TYPE = InputType.POINTWISE
eval_type = EvaluatorType.RANKING
device = cuda
train_neg_sample_args = {'strategy': 'none'}
eval_neg_sample_args = {'strategy': 'full', 'distribution': 'uniform'}


20 Jan 21:57    INFO  yelp
The number of users: 30500
Average actions of users: 10.399750811502017
The number of items: 20069
Average actions of items: 15.805361769982062
The number of inters: 317182
The sparsity of the dataset: 99.94818172387231%
Remain Fields: ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'date', 'item_name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'item_stars', 'item_review_count', 'is_open', 'categories']
20 Jan 21:57    INFO  [Training]: train_batch_size = [2048] negative sampling: [None]
20 Jan 21:57    INFO  [Evaluation]: eval_batch_size = [256] eval_args: [{'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'full'}]
20 Jan 21:57    INFO  SASRecD(
  (item_embedding): Embedding(20069, 256, padding_idx=0)
  (position_embedding): Embedding(10, 256)
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
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (fusion_layer): Linear(in_features=30, out_features=10, bias=True)
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
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (fusion_layer): Linear(in_features=30, out_features=10, bias=True)
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
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (fusion_layer): Linear(in_features=30, out_features=10, bias=True)
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
      (3): DeTransformerLayer(
        (multi_head_attention): DeMultiHeadAttention(
          (query): Linear(in_features=256, out_features=256, bias=True)
          (key): Linear(in_features=256, out_features=256, bias=True)
          (value): Linear(in_features=256, out_features=256, bias=True)
          (query_p): Linear(in_features=256, out_features=256, bias=True)
          (key_p): Linear(in_features=256, out_features=256, bias=True)
          (query_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (fusion_layer): Linear(in_features=30, out_features=10, bias=True)
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
    (0): Linear(in_features=256, out_features=1603, bias=True)
  )
  (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (loss_fct): CrossEntropyLoss()
  (attribute_loss_fct): BCEWithLogitsLoss()
)
Trainable parameters: 7696667
SASRecD
['configs/config_d_yelp.yaml', 'configs/config_training.yaml', 'configs/config_m_SASRecD.yaml']

Train     0:   0%|                                                          | 0/111 [00:00<?, ?it/s]total_loss:16.978126525878906	item_loss:9.921550750732422	attribute_categories_loss:0.7056576609611511

Train     0:   0%|                                 | 0/111 [00:00<?, ?it/s, GPU RAM: 2.79 G/23.70 G]
Train     0:   1%|▏                        | 1/111 [00:00<01:31,  1.21it/s, GPU RAM: 2.79 G/23.70 G]total_loss:16.96388053894043	item_loss:9.912309646606445	attribute_categories_loss:0.7051570415496826

Train     0:   1%|▏                        | 1/111 [00:01<01:31,  1.21it/s, GPU RAM: 3.84 G/23.70 G]
Train     0:   2%|▍                        | 2/111 [00:01<00:49,  2.19it/s, GPU RAM: 3.84 G/23.70 G]total_loss:16.973562240600586	item_loss:9.925098419189453	attribute_categories_loss:0.7048463225364685

Train     0:   2%|▍                        | 2/111 [00:01<00:49,  2.19it/s, GPU RAM: 3.84 G/23.70 G]
Train     0:   3%|▋                        | 3/111 [00:01<00:38,  2.80it/s, GPU RAM: 3.84 G/23.70 G]total_loss:16.96383285522461	item_loss:9.91990852355957	attribute_categories_loss:0.7043923139572144

Train     0:   3%|▋                        | 3/111 [00:01<00:38,  2.80it/s, GPU RAM: 3.84 G/23.70 G]
Train     0:   4%|▉                        | 4/111 [00:01<00:35,  3.05it/s, GPU RAM: 3.84 G/23.70 G]total_loss:16.9639892578125	item_loss:9.924993515014648	attribute_categories_loss:0.703899621963501

Train     0:   4%|▉                        | 4/111 [00:01<00:35,  3.05it/s, GPU RAM: 3.84 G/23.70 G]
Train     0:   5%|█▏                       | 5/111 [00:01<00:33,  3.19it/s, GPU RAM: 3.84 G/23.70 G]total_loss:16.951736450195312	item_loss:9.918646812438965	attribute_categories_loss:0.7033089399337769

Train     0:   5%|█▏                       | 5/111 [00:02<00:33,  3.19it/s, GPU RAM: 3.84 G/23.70 G]
Train     0:   5%|█▎                       | 6/111 [00:02<00:30,  3.45it/s, GPU RAM: 3.84 G/23.70 G]total_loss:16.93708038330078	item_loss:9.911049842834473	attribute_categories_loss:0.7026031613349915

Train     0:   5%|█▎                       | 6/111 [00:02<00:30,  3.45it/s, GPU RAM: 3.84 G/23.70 G]
Train     0:   6%|█▌                       | 7/111 [00:02<00:25,  4.14it/s, GPU RAM: 3.84 G/23.70 G]total_loss:16.948577880859375	item_loss:9.926965713500977	attribute_categories_loss:0.7021611332893372
...
Evaluate   :  96%|██████████████████████ | 115/120 [00:21<00:01,  3.85it/s, GPU RAM: 3.87 G/23.70 G]
Evaluate   :  96%|██████████████████████ | 115/120 [00:21<00:01,  3.85it/s, GPU RAM: 3.87 G/23.70 G]
Evaluate   :  97%|██████████████████████▏| 116/120 [00:21<00:00,  4.01it/s, GPU RAM: 3.87 G/23.70 G]
Evaluate   :  97%|██████████████████████▏| 116/120 [00:22<00:00,  4.01it/s, GPU RAM: 3.87 G/23.70 G]
Evaluate   :  98%|██████████████████████▍| 117/120 [00:22<00:00,  3.76it/s, GPU RAM: 3.87 G/23.70 G]
Evaluate   :  98%|██████████████████████▍| 117/120 [00:22<00:00,  3.76it/s, GPU RAM: 3.87 G/23.70 G]
Evaluate   :  98%|██████████████████████▌| 118/120 [00:22<00:00,  3.81it/s, GPU RAM: 3.87 G/23.70 G]
Evaluate   :  98%|██████████████████████▌| 118/120 [00:22<00:00,  3.81it/s, GPU RAM: 3.87 G/23.70 G]
Evaluate   :  99%|██████████████████████▊| 119/120 [00:22<00:00,  3.72it/s, GPU RAM: 3.87 G/23.70 G]
Evaluate   :  99%|██████████████████████▊| 119/120 [00:22<00:00,  3.72it/s, GPU RAM: 3.87 G/23.70 G]
Evaluate   : 100%|███████████████████████| 120/120 [00:22<00:00,  4.45it/s, GPU RAM: 3.87 G/23.70 G]
Evaluate   : 100%|███████████████████████| 120/120 [00:22<00:00,  5.25it/s, GPU RAM: 3.87 G/23.70 G]
20 Jan 23:14    INFO  best valid : {'recall@5': 0.0485, 'recall@10': 0.0726, 'recall@20': 0.1065, 'ndcg@5': 0.0337, 'ndcg@10': 0.0414, 'ndcg@20': 0.0499}
20 Jan 23:14    INFO  test result: {'recall@5': 0.0477, 'recall@10': 0.0698, 'recall@20': 0.1003, 'ndcg@5': 0.0348, 'ndcg@10': 0.0419, 'ndcg@20': 0.0496}


```