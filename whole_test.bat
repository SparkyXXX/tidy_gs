@echo ==========Running train script==========
python train.py -s data/Hub -m data/Hub/output --eval --iterations 30000 --optimizer_type sparse_adam --test_iterations 10000 20000 30000 --save_iterations 30000
@echo ==========Running render script==========
python render.py -m ./data/Hub/output --skip_train
@echo ==========Running metrics script==========
python metrics.py -m ./data/Hub/output
@echo ==========Done, 你真厉害!==========