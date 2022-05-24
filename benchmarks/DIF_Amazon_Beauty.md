## DIF_Amazon_Beauty

A notebook to benchmark DIF-SR on Amazon_Beauty dataset.

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
python run_recbole.py --model=SASRecD --dataset='Amazon_Beauty' --config_files='configs/Amazon_Beauty.yaml'
```

### Results
Recall@20 = 0.1283

NDCG@20 = 0.0539

### Logs
```bash
04 Apr 23:58    INFO  
General Hyper Parameters:
gpu_id = 0
use_gpu = True
seed = 212
state = INFO
reproducibility = True
data_path = dataset/Amazon_Beauty
show_progress = True
save_dataset = False
save_dataloaders = False
benchmark_filename = None

Training Hyper Parameters:
checkpoint_dir = saved/Amazon_Beauty/SASRecD_4_8_['categories']_[64]_gate_[10]_50_2048_256_0.0001_212
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
MAX_ITEM_LIST_LENGTH = 50
POSITION_FIELD = position_id
HEAD_ENTITY_ID_FIELD = head_id
TAIL_ENTITY_ID_FIELD = tail_id
RELATION_ID_FIELD = relation_id
ENTITY_ID_FIELD = entity_id

Other Hyper Parameters: 
neg_sampling = None
repeatable = True
n_layers = 4
n_heads = 8
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
fusion_type = gate
lamdas = [10]
attribute_predictor = linear
predictor_source = item
behavior_feature = not
MODEL_INPUT_TYPE = InputType.POINTWISE
eval_type = EvaluatorType.RANKING
device = cuda
train_neg_sample_args = {'strategy': 'none'}
eval_neg_sample_args = {'strategy': 'full', 'distribution': 'uniform'}


04 Apr 23:58    INFO  Amazon_Beauty
The number of users: 22364
Average actions of users: 8.876358270357287
The number of items: 12102
Average actions of items: 16.403768283612923
The number of inters: 198502
The sparsity of the dataset: 99.92665707018277%
Remain Fields: ['user_id', 'item_id', 'rating', 'timestamp', 'title', 'sales_type', 'sales_rank', 'categories', 'price', 'brand']
04 Apr 23:59    INFO  [Training]: train_batch_size = [2048] negative sampling: [None]
04 Apr 23:59    INFO  [Evaluation]: eval_batch_size = [256] eval_args: [{'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'full'}]
04 Apr 23:59    INFO  SASRecD(
  (item_embedding): Embedding(12102, 256, padding_idx=0)
  (position_embedding): Embedding(50, 256)
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
          (fusion_layer): VanillaAttention(
            (projection): Sequential(
              (0): Linear(in_features=50, out_features=50, bias=True)
              (1): ReLU(inplace=True)
              (2): Linear(in_features=50, out_features=1, bias=True)
            )
          )
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
          (fusion_layer): VanillaAttention(
            (projection): Sequential(
              (0): Linear(in_features=50, out_features=50, bias=True)
              (1): ReLU(inplace=True)
              (2): Linear(in_features=50, out_features=1, bias=True)
            )
          )
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
          (fusion_layer): VanillaAttention(
            (projection): Sequential(
              (0): Linear(in_features=50, out_features=50, bias=True)
              (1): ReLU(inplace=True)
              (2): Linear(in_features=50, out_features=1, bias=True)
            )
          )
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
          (fusion_layer): VanillaAttention(
            (projection): Sequential(
              (0): Linear(in_features=50, out_features=50, bias=True)
              (1): ReLU(inplace=True)
              (2): Linear(in_features=50, out_features=1, bias=True)
            )
          )
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
    (0): Linear(in_features=256, out_features=355, bias=True)
  )
  (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (loss_fct): CrossEntropyLoss()
  (attribute_loss_fct): BCEWithLogitsLoss()
)
Trainable parameters: 5355783
SASRecD
['configs/config_d_Amazon_Beauty.yaml', 'configs/config_training.yaml', 'configs/config_m_SASRecD.yaml']

Train     0:   0%|                                                           | 0/65 [00:00<?, ?it/s]total_loss:16.50344467163086	item_loss:9.4447603225708	attribute_categories_loss:0.7058684229850769

Train     0:   0%|                                 | 0/65 [00:00<?, ?it/s, GPU RAM: 14.05 G/23.70 G]
Train     0:   2%|▍                        | 1/65 [00:00<00:25,  2.50it/s, GPU RAM: 14.05 G/23.70 G]total_loss:16.51369857788086	item_loss:9.466225624084473	attribute_categories_loss:0.7047472596168518

Train     0:   2%|▍                        | 1/65 [00:00<00:25,  2.50it/s, GPU RAM: 14.05 G/23.70 G]
Train     0:   3%|▊                        | 2/65 [00:00<00:23,  2.70it/s, GPU RAM: 14.05 G/23.70 G]total_loss:16.489206314086914	item_loss:9.4512300491333	attribute_categories_loss:0.7037975788116455

Train     0:   3%|▊                        | 2/65 [00:01<00:23,  2.70it/s, GPU RAM: 14.05 G/23.70 G]
Train     0:   5%|█▏                       | 3/65 [00:01<00:22,  2.77it/s, GPU RAM: 14.05 G/23.70 G]total_loss:16.482175827026367	item_loss:9.451865196228027	attribute_categories_loss:0.7030311226844788

Train     0:   5%|█▏                       | 3/65 [00:01<00:22,  2.77it/s, GPU RAM: 14.05 G/23.70 G]
Train     0:   6%|█▌                       | 4/65 [00:01<00:21,  2.81it/s, GPU RAM: 14.05 G/23.70 G]total_loss:16.463132858276367	item_loss:9.445528984069824	attribute_categories_loss:0.7017603516578674

Train     0:   6%|█▌                       | 4/65 [00:01<00:21,  2.81it/s, GPU RAM: 14.05 G/23.70 G]
Train     0:   8%|█▉                       | 5/65 [00:01<00:21,  2.83it/s, GPU RAM: 14.05 G/23.70 G]total_loss:16.46879005432129	item_loss:9.458840370178223	attribute_categories_loss:0.7009949684143066

Train     0:   8%|█▉                       | 5/65 [00:02<00:21,  2.83it/s, GPU RAM: 14.05 G/23.70 G]
Train     0:   9%|██▎                      | 6/65 [00:02<00:20,  2.84it/s, GPU RAM: 14.05 G/23.70 G]total_loss:16.456663131713867	item_loss:9.45612907409668	attribute_categories_loss:0.7000533938407898

Train     0:   9%|██▎                      | 6/65 [00:02<00:20,  2.84it/s, GPU RAM: 14.05 G/23.70 G]
Train     0:  11%|██▋                      | 7/65 [00:02<00:20,  2.82it/s, GPU RAM: 14.05 G/23.70 G]total_loss:16.438268661499023	item_loss:9.453808784484863	attribute_categories_loss:0.6984460353851318
...
Evaluate   :  90%|█████████████████████▌  | 79/88 [00:01<00:00, 45.88it/s, GPU RAM: 14.07 G/23.70 G]
Evaluate   :  97%|███████████████████████▏| 85/88 [00:01<00:00, 47.73it/s, GPU RAM: 14.07 G/23.70 G]
Evaluate   :  97%|███████████████████████▏| 85/88 [00:01<00:00, 47.73it/s, GPU RAM: 14.07 G/23.70 G]
Evaluate   :  97%|███████████████████████▏| 85/88 [00:01<00:00, 47.73it/s, GPU RAM: 14.07 G/23.70 G]
Evaluate   :  97%|███████████████████████▏| 85/88 [00:01<00:00, 47.73it/s, GPU RAM: 14.07 G/23.70 G]
Evaluate   : 100%|████████████████████████| 88/88 [00:01<00:00, 48.58it/s, GPU RAM: 14.07 G/23.70 G]
05 Apr 00:44    INFO  best valid : {'recall@5': 0.0774, 'recall@10': 0.1142, 'recall@20': 0.1583, 'ndcg@5': 0.0442, 'ndcg@10': 0.056, 'ndcg@20': 0.0671}
05 Apr 00:44    INFO  test result: {'recall@5': 0.0577, 'recall@10': 0.089, 'recall@20': 0.1283, 'ndcg@5': 0.0339, 'ndcg@10': 0.044, 'ndcg@20': 0.0539}
```