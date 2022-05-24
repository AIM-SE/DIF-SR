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
python run_recbole.py --model=SASRecD --dataset='Amazon_Sports_and_Outdoors' --config_files='configs/Amazon_Sports_and_Outdoors.yaml'
```

### Results
Recall@20= 0.0807

NDCG@20 = 0.0322

### Logs
```bash
06 Apr 17:39    INFO  
General Hyper Parameters:
gpu_id = 1
use_gpu = True
seed = 212
state = INFO
reproducibility = True
data_path = dataset/Amazon_Sports_and_Outdoors
show_progress = True
save_dataset = False
save_dataloaders = False
benchmark_filename = None

Training Hyper Parameters:
checkpoint_dir = saved/Amazon_Sports_and_Outdoors/SASRecD_3_8_['categories']_[64]_gate_[5]_50_2048_256_0.0001_212
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
n_layers = 3
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
lamdas = [5]
attribute_predictor = linear
predictor_source = item
behavior_feature = not
MODEL_INPUT_TYPE = InputType.POINTWISE
eval_type = EvaluatorType.RANKING
device = cuda
train_neg_sample_args = {'strategy': 'none'}
eval_neg_sample_args = {'strategy': 'full', 'distribution': 'uniform'}


06 Apr 17:39    INFO  Amazon_Sports_and_Outdoors
The number of users: 35599
Average actions of users: 8.324540704533963
The number of items: 18358
Average actions of items: 16.142997221768262
The number of inters: 296337
The sparsity of the dataset: 99.95465569853714%
Remain Fields: ['user_id', 'item_id', 'rating', 'timestamp', 'title', 'price', 'brand', 'categories', 'sales_type', 'sales_rank']
06 Apr 17:39    INFO  [Training]: train_batch_size = [2048] negative sampling: [None]
06 Apr 17:39    INFO  [Evaluation]: eval_batch_size = [256] eval_args: [{'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'full'}]
06 Apr 17:40    INFO  SASRecD(
  (item_embedding): Embedding(18358, 256, padding_idx=0)
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
    )
  )
  (ap): ModuleList(
    (0): Linear(in_features=256, out_features=1876, bias=True)
  )
  (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (loss_fct): CrossEntropyLoss()
  (attribute_loss_fct): BCEWithLogitsLoss()
)
Trainable parameters: 6809935
SASRecD
['configs/config_d_Amazon_Sports_and_Outdoors.yaml', 'configs/config_training.yaml', 'configs/config_m_SASRecD.yaml']

Train     0:   0%|                                                           | 0/93 [00:00<?, ?it/s]total_loss:13.417797088623047	item_loss:9.889311790466309	attribute_categories_loss:0.7056969404220581

Train     0:   0%|                                 | 0/93 [00:00<?, ?it/s, GPU RAM: 11.30 G/23.70 G]
Train     0:   1%|▎                        | 1/93 [00:00<00:35,  2.57it/s, GPU RAM: 11.30 G/23.70 G]total_loss:13.399545669555664	item_loss:9.873366355895996	attribute_categories_loss:0.7052357792854309

Train     0:   1%|▎                        | 1/93 [00:00<00:35,  2.57it/s, GPU RAM: 12.57 G/23.70 G]
Train     0:   2%|▌                        | 2/93 [00:00<00:31,  2.90it/s, GPU RAM: 12.57 G/23.70 G]total_loss:13.395700454711914	item_loss:9.870609283447266	attribute_categories_loss:0.705018162727356

Train     0:   2%|▌                        | 2/93 [00:01<00:31,  2.90it/s, GPU RAM: 12.57 G/23.70 G]
Train     0:   3%|▊                        | 3/93 [00:01<00:30,  2.99it/s, GPU RAM: 12.57 G/23.70 G]total_loss:13.406332015991211	item_loss:9.883359909057617	attribute_categories_loss:0.7045944333076477

Train     0:   3%|▊                        | 3/93 [00:01<00:30,  2.99it/s, GPU RAM: 12.57 G/23.70 G]
Train     0:   4%|█                        | 4/93 [00:01<00:28,  3.07it/s, GPU RAM: 12.57 G/23.70 G]total_loss:13.398996353149414	item_loss:9.878922462463379	attribute_categories_loss:0.7040147185325623

Train     0:   4%|█                        | 4/93 [00:01<00:28,  3.07it/s, GPU RAM: 12.57 G/23.70 G]
Train     0:   5%|█▎                       | 5/93 [00:01<00:28,  3.12it/s, GPU RAM: 12.57 G/23.70 G]total_loss:13.396454811096191	item_loss:9.87670612335205	attribute_categories_loss:0.7039497494697571

Train     0:   5%|█▎                       | 5/93 [00:01<00:28,  3.12it/s, GPU RAM: 12.57 G/23.70 G]
Train     0:   6%|█▌                       | 6/93 [00:01<00:27,  3.15it/s, GPU RAM: 12.57 G/23.70 G]total_loss:13.394804954528809	item_loss:9.876762390136719	attribute_categories_loss:0.7036085724830627
...
Evaluate   :  95%|████████████████████▉ | 133/140 [00:02<00:00, 52.14it/s, GPU RAM: 12.58 G/23.70 G]
Evaluate   :  95%|████████████████████▉ | 133/140 [00:02<00:00, 52.14it/s, GPU RAM: 12.58 G/23.70 G]
Evaluate   :  95%|████████████████████▉ | 133/140 [00:02<00:00, 52.14it/s, GPU RAM: 12.58 G/23.70 G]
Evaluate   :  95%|████████████████████▉ | 133/140 [00:02<00:00, 52.14it/s, GPU RAM: 12.59 G/23.70 G]
Evaluate   : 100%|██████████████████████| 140/140 [00:02<00:00, 56.03it/s, GPU RAM: 12.59 G/23.70 G]
Evaluate   : 100%|██████████████████████| 140/140 [00:02<00:00, 46.96it/s, GPU RAM: 12.59 G/23.70 G]
06 Apr 18:40    INFO  best valid : {'recall@5': 0.0443, 'recall@10': 0.0685, 'recall@20': 0.0985, 'ndcg@5': 0.0241, 'ndcg@10': 0.0319, 'ndcg@20': 0.0394}
06 Apr 18:40    INFO  test result: {'recall@5': 0.0355, 'recall@10': 0.0548, 'recall@20': 0.0807, 'ndcg@5': 0.0194, 'ndcg@10': 0.0256, 'ndcg@20': 0.0322}


```