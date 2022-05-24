# Amazon Beauty
python run_recbole.py --gpu_id=0 --model=SASRecD --dataset='Amazon_Beauty' --config_files='configs/Amazon_Beauty.yaml' > log/beauty.log 2>&1 &
# Amazon_Sports_and_Outdoors
python run_recbole.py --gpu_id=1 --model=SASRecD --dataset='Amazon_Sports_and_Outdoors' --config_files='configs/Amazon_Sports_and_Outdoors.yaml' > log/sports.log 2>&1 &
# Amazon_Toys_and_Games
python run_recbole.py --gpu_id=2 --model=SASRecD --dataset='Amazon_Toys_and_Games' --config_files='configs/Amazon_Toys_and_Games.yaml' > log/toys.log 2>&1 &
# yelp
python run_recbole.py --gpu_id=3 --model=SASRecD --dataset='yelp' --config_files='configs/yelp.yaml' > log/yelp.log 2>&1 &
