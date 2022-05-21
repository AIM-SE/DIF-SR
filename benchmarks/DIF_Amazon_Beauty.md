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
python main.py --config_file amazon_config.ini
```

### Results
Recall@20 = 0.1284

NDCG@20 = 0.0541

### Logs
```bash
19 Jan 18:11    INFO  
General Hyper Parameters:
gpu_id = 1
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
checkpoint_dir = saved/Amazon_Beauty/SASRecD_4_4_['categories']_[256]_gate_[10]_212
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
metrics = ['Recall',  'NDCG']
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
n_layers = 4
n_heads = 4
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
attribute_hidden_size = [256]
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


19 Jan 18:11    INFO  Amazon_Beauty
The number of users: 22364
Average actions of users: 8.876358270357287
The number of items: 12102
Average actions of items: 16.403768283612923
The number of inters: 198502
The sparsity of the dataset: 99.92665707018277%
Remain Fields: ['user_id', 'item_id', 'rating', 'timestamp', 'title', 'sales_type', 'sales_rank', 'categories', 'price', 'brand']
19 Jan 18:11    INFO  [Training]: train_batch_size = [2048] negative sampling: [None]
19 Jan 18:11    INFO  [Evaluation]: eval_batch_size = [256] eval_args: [{'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'full'}]
19 Jan 18:11    INFO  SASRecD(
  (item_embedding): Embedding(12102, 256, padding_idx=0)
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
            (0): Linear(in_features=256, out_features=256, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=256, out_features=256, bias=True)
          )
          (fusion_layer): VanillaAttention(
            (projection): Sequential(
              (0): Linear(in_features=7, out_features=7, bias=True)
              (1): ReLU(inplace=True)
              (2): Linear(in_features=7, out_features=1, bias=True)
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
            (0): Linear(in_features=256, out_features=256, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=256, out_features=256, bias=True)
          )
          (fusion_layer): VanillaAttention(
            (projection): Sequential(
              (0): Linear(in_features=7, out_features=7, bias=True)
              (1): ReLU(inplace=True)
              (2): Linear(in_features=7, out_features=1, bias=True)
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
            (0): Linear(in_features=256, out_features=256, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=256, out_features=256, bias=True)
          )
          (fusion_layer): VanillaAttention(
            (projection): Sequential(
              (0): Linear(in_features=7, out_features=7, bias=True)
              (1): ReLU(inplace=True)
              (2): Linear(in_features=7, out_features=1, bias=True)
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
            (0): Linear(in_features=256, out_features=256, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=256, out_features=256, bias=True)
          )
          (fusion_layer): VanillaAttention(
            (projection): Sequential(
              (0): Linear(in_features=7, out_features=7, bias=True)
              (1): ReLU(inplace=True)
              (2): Linear(in_features=7, out_features=1, bias=True)
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
Trainable parameters: 5827683
SASRecD
['configs/config_d_Amazon_Beauty.yaml', 'configs/config_training.yaml', 'configs/config_m_SASRecD.yaml']

Train     0:   0%|                                                           | 0/65 [00:00<?, ?it/s]total_loss:16.517316818237305	item_loss:9.458878517150879	attribute_categories_loss:0.7058438062667847

Train     0:   0%|                                  | 0/65 [00:00<?, ?it/s, GPU RAM: 1.57 G/23.70 G]
Train     0:   2%|▍                         | 1/65 [00:00<00:37,  1.71it/s, GPU RAM: 1.57 G/23.70 G]total_loss:16.50027847290039	item_loss:9.452102661132812	attribute_categories_loss:0.7048174738883972

Train     0:   2%|▍                         | 1/65 [00:00<00:37,  1.71it/s, GPU RAM: 1.58 G/23.70 G]
Train     0:   3%|▊                         | 2/65 [00:00<00:20,  3.15it/s, GPU RAM: 1.58 G/23.70 G]total_loss:16.50355339050293	item_loss:9.461633682250977	attribute_categories_loss:0.7041919231414795

Train     0:   3%|▊                         | 2/65 [00:00<00:20,  3.15it/s, GPU RAM: 1.58 G/23.70 G]
Train     0:   5%|█▏                        | 3/65 [00:00<00:16,  3.87it/s, GPU RAM: 1.58 G/23.70 G]total_loss:16.488117218017578	item_loss:9.455796241760254	attribute_categories_loss:0.7032319903373718

Train     0:   5%|█▏                        | 3/65 [00:01<00:16,  3.87it/s, GPU RAM: 1.58 G/23.70 G]
Train     0:   6%|█▌                        | 4/65 [00:01<00:12,  5.01it/s, GPU RAM: 1.58 G/23.70 G]total_loss:16.488605499267578	item_loss:9.464791297912598	attribute_categories_loss:0.7023813128471375

Train     0:   6%|█▌                        | 4/65 [00:01<00:12,  5.01it/s, GPU RAM: 1.58 G/23.70 G]
Train     0:   8%|██                        | 5/65 [00:01<00:11,  5.36it/s, GPU RAM: 1.58 G/23.70 G]total_loss:16.45383071899414	item_loss:9.44739055633545	attribute_categories_loss:0.7006440162658691

Train     0:   8%|██                        | 5/65 [00:01<00:11,  5.36it/s, GPU RAM: 1.58 G/23.70 G]
Train     0:   9%|██▍                       | 6/65 [00:01<00:10,  5.90it/s, GPU RAM: 1.58 G/23.70 G]total_loss:16.457260131835938	item_loss:9.455888748168945	attribute_categories_loss:0.7001371383666992

Train     0:   9%|██▍                       | 6/65 [00:01<00:10,  5.90it/s, GPU RAM: 1.58 G/23.70 G]
Train     0:  11%|██▊                       | 7/65 [00:01<00:09,  6.05it/s, GPU RAM: 1.58 G/23.70 G]total_loss:16.436485290527344	item_loss:9.448519706726074	attribute_categories_loss:0.6987965106964111

Train     0:  11%|██▊                       | 7/65 [00:01<00:09,  6.05it/s, GPU RAM: 1.58 G/23.70 G]
Train     0:  12%|███▏                      | 8/65 [00:01<00:09,  6.06it/s, GPU RAM: 1.58 G/23.70 G]total_loss:16.431352615356445	item_loss:9.452569961547852	attribute_categories_loss:0.6978783011436462

Train     0:  12%|███▏                      | 8/65 [00:01<00:09,  6.06it/s, GPU RAM: 1.58 G/23.70 G]
Train     0:  14%|███▌                      | 9/65 [00:01<00:08,  6.31it/s, GPU RAM: 1.58 G/23.70 G]total_loss:16.41012954711914	item_loss:9.446404457092285	attribute_categories_loss:0.6963725686073303

Train     0:  14%|███▌                      | 9/65 [00:01<00:08,  6.31it/s, GPU RAM: 1.58 G/23.70 G]
Train     0:  15%|███▊                     | 10/65 [00:01<00:08,  6.46it/s, GPU RAM: 1.58 G/23.70 G]total_loss:16.386985778808594	item_loss:9.441742897033691	attribute_categories_loss:0.6945242285728455

Train     0:  15%|███▊                     | 10/65 [00:02<00:08,  6.46it/s, GPU RAM: 1.58 G/23.70 G]
Train     0:  17%|████▏                    | 11/65 [00:02<00:08,  6.36it/s, GPU RAM: 1.58 G/23.70 G]total_loss:16.377216339111328	item_loss:9.442288398742676	attribute_categories_loss:0.6934927105903625
...
Evaluate   :  94%|███████████████████████▌ | 83/88 [00:09<00:00,  7.99it/s, GPU RAM: 1.60 G/23.70 G]
Evaluate   :  94%|███████████████████████▌ | 83/88 [00:09<00:00,  7.99it/s, GPU RAM: 1.60 G/23.70 G]
Evaluate   :  97%|████████████████████████▏| 85/88 [00:09<00:00,  8.46it/s, GPU RAM: 1.60 G/23.70 G]
Evaluate   :  97%|████████████████████████▏| 85/88 [00:09<00:00,  8.46it/s, GPU RAM: 1.60 G/23.70 G]
Evaluate   :  98%|████████████████████████▍| 86/88 [00:09<00:00,  7.31it/s, GPU RAM: 1.60 G/23.70 G]
Evaluate   :  98%|████████████████████████▍| 86/88 [00:09<00:00,  7.31it/s, GPU RAM: 1.60 G/23.70 G]
Evaluate   :  99%|████████████████████████▋| 87/88 [00:09<00:00,  7.11it/s, GPU RAM: 1.60 G/23.70 G]
Evaluate   :  99%|████████████████████████▋| 87/88 [00:09<00:00,  7.11it/s, GPU RAM: 1.61 G/23.70 G]
Evaluate   : 100%|█████████████████████████| 88/88 [00:09<00:00,  8.90it/s, GPU RAM: 1.61 G/23.70 G]
19 Jan 18:45    INFO  best valid : {'recall@5': 0.0772, 'recall@10': 0.1139, 'recall@20': 0.1561, 'ndcg@5': 0.0443, 'ndcg@10': 0.0561, 'ndcg@20': 0.0668}
19 Jan 18:45    INFO  test result: {'recall@5': 0.0569, 'recall@10': 0.0908, 'recall@20': 0.1284, 'ndcg@5': 0.0337, 'ndcg@10': 0.0446, 'ndcg@20': 0.0541}


```